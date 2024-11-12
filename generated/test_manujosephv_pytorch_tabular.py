
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


import random


import pandas as pd


from sklearn.datasets import make_classification


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


from sklearn.model_selection import train_test_split


import time


import torch


from sklearn.datasets import fetch_california_housing


from sklearn.preprocessing import PowerTransformer


from typing import Dict


from typing import List


from typing import Optional


import torch.nn as nn


import re


from typing import Any


from typing import Iterable


from collections import defaultdict


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


import warnings


from abc import ABCMeta


from abc import abstractmethod


from functools import partial


from typing import Callable


from typing import Tuple


from typing import Type


from typing import Union


import numpy as np


from pandas import DataFrame


from torch import Tensor


from torch.optim import Optimizer


import math


from torch import nn


from torch.autograd import Variable


from torch.distributions import Categorical


import torch.nn.functional as F


from torch.autograd import Function


from torch.jit import script


from warnings import warn


from torch import einsum


from collections import OrderedDict


from collections import namedtuple


from enum import Enum


from pandas import DatetimeTZDtype


from pandas import to_datetime


from pandas.tseries import offsets


from pandas.tseries.frequencies import to_offset


from sklearn.base import copy


from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import QuantileTransformer


from sklearn.preprocessing import StandardScaler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import inspect


from sklearn.model_selection import BaseCrossValidator


from sklearn.model_selection import KFold


from sklearn.model_selection import StratifiedKFold


from pandas import Series


from sklearn.cluster import KMeans


from sklearn.datasets import make_regression


from typing import IO


import copy


from scipy.stats import uniform


from sklearn.metrics import r2_score


class GBN(nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)


class BatchNorm1d(nn.Module):
    """BatchNorm1d with Ghost Batch Normalization."""

    def __init__(self, num_features, virtual_batch_size=None):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        if self.virtual_batch_size is None:
            self.bn = nn.BatchNorm1d(self.num_features)
        else:
            self.bn = GBN(self.num_features, self.virtual_batch_size)

    def forward(self, x):
        return self.bn(x)


class SharedEmbeddings(nn.Module):
    """Enables different values in a categorical feature to share some embeddings across."""

    def __init__(self, num_embed: 'int', embed_dim: 'int', add_shared_embed: 'bool'=False, frac_shared_embed: 'float'=0.25):
        super().__init__()
        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, :shared_embed.shape[1]] = shared_embed
        return out

    @property
    def weight(self):
        w = self.embed.weight.detach()
        if self.add_shared_embed:
            w += self.shared_embed
        else:
            w[:, :self.shared_embed.shape[1]] = self.shared_embed
        return w


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == 'kaiming_uniform':
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == 'kaiming_normal':
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError('initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`')


class Embedding2dLayer(nn.Module):
    """Embeds categorical and continuous features into a 2D tensor."""

    def __init__(self, continuous_dim: 'int', categorical_cardinality: 'List[int]', embedding_dim: 'int', shared_embedding_strategy: 'Optional[str]'=None, frac_shared_embed: 'float'=0.25, embedding_bias: 'bool'=False, batch_norm_continuous_input: 'bool'=False, virtual_batch_size: 'Optional[int]'=None, embedding_dropout: 'float'=0.0, initialization: 'Optional[str]'=None):
        """
        Args:
            continuous_dim: number of continuous features
            categorical_cardinality: list of cardinalities of categorical features
            embedding_dim: embedding dimension
            shared_embedding_strategy: strategy to use for shared embeddings
            frac_shared_embed: fraction of embeddings to share
            embedding_bias: whether to use bias in embedding layers
            batch_norm_continuous_input: whether to use batch norm on continuous features
            embedding_dropout: dropout to apply to embeddings
            initialization: initialization strategy to use for embedding layers"""
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_cardinality = categorical_cardinality
        self.embedding_dim = embedding_dim
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.shared_embedding_strategy = shared_embedding_strategy
        self.frac_shared_embed = frac_shared_embed
        self.embedding_bias = embedding_bias
        self.initialization = initialization
        d_sqrt_inv = 1 / math.sqrt(embedding_dim)
        if initialization is not None:
            assert initialization in ['kaiming_uniform', 'kaiming_normal'], 'initialization should be either of `kaiming` or `uniform`'
            self._do_kaiming_initialization = True
            self._initialize_kaiming = partial(_initialize_kaiming, initialization=initialization, d_sqrt_inv=d_sqrt_inv)
        else:
            self._do_kaiming_initialization = False
        if self.shared_embedding_strategy is not None:
            self.cat_embedding_layers = nn.ModuleList([SharedEmbeddings(c, self.embedding_dim, add_shared_embed=self.shared_embedding_strategy == 'add', frac_shared_embed=self.frac_shared_embed) for c in categorical_cardinality])
            if self._do_kaiming_initialization:
                for embedding_layer in self.cat_embedding_layers:
                    self._initialize_kaiming(embedding_layer.embed.weight)
                    self._initialize_kaiming(embedding_layer.shared_embed)
        else:
            self.cat_embedding_layers = nn.ModuleList([nn.Embedding(c, self.embedding_dim) for c in categorical_cardinality])
            if self._do_kaiming_initialization:
                for embedding_layer in self.cat_embedding_layers:
                    self._initialize_kaiming(embedding_layer.weight)
        if embedding_bias:
            self.cat_embedding_bias = nn.Parameter(torch.Tensor(len(self.categorical_cardinality), self.embedding_dim))
            if self._do_kaiming_initialization:
                self._initialize_kaiming(self.cat_embedding_bias)
        self.cont_embedding_layer = nn.Embedding(self.continuous_dim, self.embedding_dim)
        if self._do_kaiming_initialization:
            self._initialize_kaiming(self.cont_embedding_layer.weight)
        if embedding_bias:
            self.cont_embedding_bias = nn.Parameter(torch.Tensor(self.continuous_dim, self.embedding_dim))
            if self._do_kaiming_initialization:
                self._initialize_kaiming(self.cont_embedding_bias)
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None

    def forward(self, x: 'Dict[str, Any]') ->torch.Tensor:
        assert 'continuous' in x or 'categorical' in x, 'x must contain either continuous and categorical features'
        continuous_data, categorical_data = x.get('continuous', torch.empty(0, 0)), x.get('categorical', torch.empty(0, 0))
        assert categorical_data.shape[1] == len(self.cat_embedding_layers), 'categorical_data must have same number of columns as categorical embedding layers'
        assert continuous_data.shape[1] == self.continuous_dim, 'continuous_data must have same number of columns as continuous dim'
        embed = None
        if continuous_data.shape[1] > 0:
            cont_idx = torch.arange(self.continuous_dim, device=continuous_data.device).expand(continuous_data.size(0), -1)
            if self.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)
            embed = torch.mul(continuous_data.unsqueeze(2), self.cont_embedding_layer(cont_idx))
            if self.embedding_bias:
                embed += self.cont_embedding_bias
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat([embedding_layer(categorical_data[:, i]).unsqueeze(1) for i, embedding_layer in enumerate(self.cat_embedding_layers)], dim=1)
            if self.embedding_bias:
                categorical_embed += self.cat_embedding_bias
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get('PT_LOGLEVEL', 'INFO'))
    formatter = logging.Formatter('%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s')
    if not logger.hasHandlers():
        ch = RichHandler(show_level=False, show_time=False, show_path=False, rich_tracebacks=True)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    return logger


def _initialize_layers(activation, initialization, layers):
    if type(layers) is nn.Sequential:
        for layer in layers:
            if hasattr(layer, 'weight'):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == 'ReLU':
            nonlinearity = 'relu'
        elif activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        elif initialization == 'kaiming':
            logger.warning('Kaiming initialization is only recommended for ReLU and LeakyReLU.')
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        if initialization == 'kaiming':
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == 'xavier':
            nn.init.xavier_normal_(layers.weight, gain=nn.init.calculate_gain(nonlinearity) if activation in ['ReLU', 'LeakyReLU'] else 1)
        elif initialization == 'random':
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers


class AutoIntBackbone(nn.Module):

    def __init__(self, config: 'DictConfig'):
        """Automatic Feature Interaction Network.

        Args:
            config (DictConfig): config of the model

        """
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        _curr_units = self.hparams.embedding_dim
        if self.hparams.deep_layers:
            layers = []
            for units in self.hparams.layers.split('-'):
                layers.extend(_linear_dropout_bn(self.hparams.activation, self.hparams.initialization, self.hparams.use_batch_norm, _curr_units, int(units), self.hparams.dropout))
                _curr_units = int(units)
            self.linear_layers = nn.Sequential(*layers)
        self.attn_proj = nn.Linear(_curr_units, self.hparams.attn_embed_dim)
        _initialize_layers(self.hparams.activation, self.hparams.initialization, self.attn_proj)
        self.self_attns = nn.ModuleList([nn.MultiheadAttention(self.hparams.attn_embed_dim, self.hparams.num_heads, dropout=self.hparams.attn_dropouts) for _ in range(self.hparams.num_attn_blocks)])
        if self.hparams.has_residuals:
            self.V_res_embedding = torch.nn.Linear(_curr_units, self.hparams.attn_embed_dim * self.hparams.num_attn_blocks if self.hparams.attention_pooling else self.hparams.attn_embed_dim)
        self.output_dim = (self.hparams.continuous_dim + self.hparams.categorical_dim) * self.hparams.attn_embed_dim
        if self.hparams.attention_pooling:
            self.output_dim = self.output_dim * self.hparams.num_attn_blocks

    def _build_embedding_layer(self):
        return Embedding2dLayer(continuous_dim=self.hparams.continuous_dim, categorical_cardinality=self.hparams.categorical_cardinality, embedding_dim=self.hparams.embedding_dim, shared_embedding_strategy=self.hparams.share_embedding_strategy, frac_shared_embed=self.hparams.shared_embedding_fraction, embedding_bias=self.hparams.embedding_bias, batch_norm_continuous_input=self.hparams.batch_norm_continuous_input, embedding_dropout=self.hparams.embedding_dropout, initialization=self.hparams.embedding_initialization)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.hparams.deep_layers:
            x = self.linear_layers(x)
        cross_term = self.attn_proj(x).transpose(0, 1)
        if self.hparams.attention_pooling:
            attention_ops = []
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if self.hparams.attention_pooling:
                attention_ops.append(cross_term)
        if self.hparams.attention_pooling:
            cross_term = torch.cat(attention_ops, dim=-1)
        cross_term = cross_term.transpose(0, 1)
        if self.hparams.has_residuals:
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
        cross_term = nn.ReLU()(cross_term).reshape(-1, self.output_dim)
        return cross_term


class _CaptumModel(nn.Module):

    def __init__(self, model: 'BaseModel'):
        super().__init__()
        self.model = model

    def forward(self, x: 'Tensor'):
        x = self.model.compute_backbone(x)
        return self.model.compute_head(x)['logits']


class Embedding1dLayer(nn.Module):
    """Enables different values in a categorical features to have different embeddings."""

    def __init__(self, continuous_dim: 'int', categorical_embedding_dims: 'Tuple[int, int]', embedding_dropout: 'float'=0.0, batch_norm_continuous_input: 'bool'=False, virtual_batch_size: 'Optional[int]'=None):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.cat_embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)

    def forward(self, x: 'Dict[str, Any]') ->torch.Tensor:
        assert 'continuous' in x or 'categorical' in x, 'x must contain either continuous and categorical features'
        continuous_data, categorical_data = x.get('continuous', torch.empty(0, 0)), x.get('categorical', torch.empty(0, 0))
        assert categorical_data.shape[1] == len(self.cat_embedding_layers), 'categorical_data must have same number of columns as categorical embedding layers'
        assert continuous_data.shape[1] == self.continuous_dim, 'continuous_data must have same number of columns as continuous dim'
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat([embedding_layer(categorical_data[:, i]) for i, embedding_layer in enumerate(self.cat_embedding_layers)], dim=1)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


class CategoryEmbeddingBackbone(nn.Module):

    def __init__(self, config: 'DictConfig', **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        layers = []
        if hasattr(self.hparams, '_backbone_input_dim'):
            _curr_units = self.hparams._backbone_input_dim
        else:
            _curr_units = self.hparams.embedded_cat_dim + self.hparams.continuous_dim
        for units in self.hparams.layers.split('-'):
            layers.extend(_linear_dropout_bn(self.hparams.activation, self.hparams.initialization, self.hparams.use_batch_norm, _curr_units, int(units), self.hparams.dropout))
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        _initialize_layers(self.hparams.activation, self.hparams.initialization, self.linear_layers)
        self.output_dim = _curr_units

    def _build_embedding_layer(self):
        return Embedding1dLayer(continuous_dim=self.hparams.continuous_dim, categorical_embedding_dims=self.hparams.embedding_dims, embedding_dropout=self.hparams.embedding_dropout, batch_norm_continuous_input=self.hparams.batch_norm_continuous_input, virtual_batch_size=self.hparams.virtual_batch_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.linear_layers(x)
        return x


class Head(nn.Module):

    def __init__(self, layers, config_template, **kwargs):
        super().__init__()
        self.layers = layers
        self._config_template = config_template

    def forward(self, x):
        return self.layers(x)


LOG2PI = math.log(2 * math.pi)


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


def t_softmax(input: 'Tensor', t: 'Tensor'=None, dim: 'int'=-1) ->Tensor:
    if t is None:
        t = torch.tensor(0.5, device=input.device)
    assert (t >= 0.0).all()
    maxes = torch.max(input, dim=dim, keepdim=True).values
    input_minus_maxes = input - maxes
    w = torch.relu(input_minus_maxes + t) + 1e-08
    return torch.softmax(input_minus_maxes + torch.log(w), dim=dim)


class TSoftmax(torch.nn.Module):

    def __init__(self, dim: 'int'=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: 'Tensor', t: 'Tensor') ->Tensor:
        return t_softmax(input, t, self.dim)


class RSoftmax(torch.nn.Module):

    def __init__(self, dim: 'int'=-1, eps: 'float'=1e-08):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.tsoftmax = TSoftmax(dim=dim)

    @classmethod
    def calculate_t(cls, input: 'Tensor', r: 'Tensor', dim: 'int'=-1, eps: 'float'=1e-08):
        assert ((0.0 <= r) & (r <= 1.0)).all()
        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes
        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]
        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
            t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
            t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
            t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: 'Tensor', r: 'Tensor'):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return self.tsoftmax(input, t)


class Sparsemax(nn.Module):

    def __init__(self, dim=-1, k=None):
        """sparsemax: normalizing sparse transform (a la softmax).

        Solves the projection:

            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

        Parameters
        ----------
        dim : int
            The dimension along which to apply sparsemax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super().__init__()

    def forward(self, X):
        return sparsemax(X, dim=self.dim, k=self.k)


class Entmax15(nn.Module):

    def __init__(self, dim=-1, k=None):
        """1.5-entmax: normalizing sparse transform (a la softmax).

        Solves the optimization problem:

            max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

        where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

        Parameters
        ----------
        dim : int
            The dimension along which to apply 1.5-entmax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super().__init__()

    def forward(self, X):
        return entmax15(X, dim=self.dim, k=self.k)


class PreEncoded1dLayer(nn.Module):
    """Takes in pre-encoded categorical variables and just concatenates with continuous variables No learnable
    component."""

    def __init__(self, continuous_dim: 'int', categorical_dim: 'Tuple[int, int]', embedding_dropout: 'float'=0.0, batch_norm_continuous_input: 'bool'=False, virtual_batch_size: 'Optional[int]'=None):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dim = categorical_dim
        self.batch_norm_continuous_input = batch_norm_continuous_input
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)

    def forward(self, x: 'Dict[str, Any]') ->torch.Tensor:
        assert 'continuous' in x or 'categorical' in x, 'x must contain either continuous and categorical features'
        continuous_data, categorical_data = x.get('continuous', torch.empty(0, 0)), x.get('categorical', torch.empty(0, 0))
        assert categorical_data.shape[1] == self.categorical_dim, 'categorical_data must have same number of columns as categorical embedding layers'
        assert continuous_data.shape[1] == self.continuous_dim, 'continuous_data must have same number of columns as continuous dim'
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
        if categorical_data.shape[1] > 0:
            if embed is None:
                embed = categorical_data
            else:
                embed = torch.cat([embed, categorical_data], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


class PositionWiseFeedForward(nn.Module):
    """
    title: Position-wise Feed-Forward Network (FFN)
    summary: Documented reusable implementation of the position wise feedforward network.

    # Position-wise Feed-Forward Network (FFN)
    This is a [PyTorch](https://pytorch.org)  implementation
    of position-wise feedforward network used in transformer.
    FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$.
    So it is sometime also called the expand-and-contract network.
    There is an activation at the hidden layer, which is
    usually set to ReLU (Rectified Linear Unit) activation, $$\\\\max(0, x)$$
    That is, the FFN function is,
    $$FFN(x, W_1, W_2, b_1, b_2) = \\\\max(0, x W_1 + b_1) W_2 + b_2$$
    where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.
    Sometimes the
    GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
    $$x \\\\Phi(x)$$ where $\\\\Phi(x) = P(X \\\\le x), X \\\\sim \\\\mathcal{N}(0,1)$
    ### Gated Linear Units
    This is a generic implementation that supports different variants including
    [Gated Linear Units](https://arxiv.org/abs/2002.05202) (GLU).
    """

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1, activation=nn.ReLU(), is_gated: 'bool'=False, bias1: 'bool'=True, bias2: 'bool'=True, bias_gate: 'bool'=True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: 'torch.Tensor'):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class GEGLU(nn.Module):
    """Gated Exponential Linear Unit (GEGLU)"""

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.GELU(), True, False, False, False)

    def forward(self, x: 'torch.Tensor'):
        return self.ffn(x)


class ReGLU(nn.Module):
    """ReGLU."""

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.ReLU(), True, False, False, False)

    def forward(self, x: 'torch.Tensor'):
        return self.ffn(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.SiLU(), True, False, False, False)

    def forward(self, x: 'torch.Tensor'):
        return self.ffn(x)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Lambda(nn.Module):
    """A wrapper for a lambda function as a pytorch module."""

    def __init__(self, func: 'Callable'):
        """Initialize lambda module
        Args:
            func: any function/callable
        """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None

    def initialize(self, *args, **kwargs):
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError('Please implement ')

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class Add(nn.Module):
    """A module that adds a constant/parameter value to the input."""

    def __init__(self, add_value: 'Union[float, torch.Tensor]'):
        """Initialize the module.

        Args:
            add_value: The value to add to the input

        """
        super().__init__()
        self.add_value = add_value

    def forward(self, x):
        return x + self.add_value


def check_numpy(x):
    """Makes sure x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def sparsemoid(input):
    return (0.5 * input + 0.5).clamp_(0, 1)


class AddNorm(nn.Module):
    """Applies LayerNorm, Dropout and adds to input.

    Standard AddNorm operations in Transformers

    """

    def __init__(self, input_dim: 'int', dropout: 'float'):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: 'torch.Tensor', Y: 'torch.Tensor') ->torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    """Multi Headed Attention Block in Transformers."""

    def __init__(self, input_dim: 'int', num_heads: 'int'=8, head_dim: 'int'=16, dropout: 'int'=0.1, keep_attn: 'bool'=True):
        super().__init__()
        assert input_dim % num_heads == 0, "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim ** -0.5
        self.keep_attn = keep_attn
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=h) for t in (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if self.keep_attn:
            self.attn_weights = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


GATED_UNITS = {'GEGLU': GEGLU, 'ReGLU': ReGLU, 'SwiGLU': SwiGLU}


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block."""

    def __init__(self, input_embed_dim: 'int', num_heads: 'int'=8, ff_hidden_multiplier: 'int'=4, ff_activation: 'str'='GEGLU', attn_dropout: 'float'=0.1, keep_attn: 'bool'=True, ff_dropout: 'float'=0.1, add_norm_dropout: 'float'=0.1, transformer_head_dim: 'Optional[int]'=None):
        """
        Args:
            input_embed_dim: The input embedding dimension
            num_heads: The number of attention heads
            ff_hidden_multiplier: The hidden dimension multiplier for the position-wise feed-forward layer
            ff_activation: The activation function for the position-wise feed-forward layer
            attn_dropout: The dropout probability for the attention layer
            keep_attn: Whether to keep the attention weights
            ff_dropout: The dropout probability for the position-wise feed-forward layer
            add_norm_dropout: The dropout probability for the residual connections
            transformer_head_dim: The dimension of the attention heads. If None, will default to input_embed_dim
        """
        super().__init__()
        self.mha = MultiHeadedAttention(input_embed_dim, num_heads, head_dim=input_embed_dim if transformer_head_dim is None else transformer_head_dim, dropout=attn_dropout, keep_attn=keep_attn)
        try:
            self.pos_wise_ff = GATED_UNITS[ff_activation](d_model=input_embed_dim, d_ff=input_embed_dim * ff_hidden_multiplier, dropout=ff_dropout)
        except (AttributeError, KeyError):
            self.pos_wise_ff = PositionWiseFeedForward(d_model=input_embed_dim, d_ff=input_embed_dim * ff_hidden_multiplier, dropout=ff_dropout, activation=getattr(nn, ff_activation)())
        self.attn_add_norm = AddNorm(input_embed_dim, add_norm_dropout)
        self.ff_add_norm = AddNorm(input_embed_dim, add_norm_dropout)

    def forward(self, x):
        y = self.mha(x)
        x = self.attn_add_norm(x, y)
        y = self.pos_wise_ff(y)
        return self.ff_add_norm(x, y)


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: 'int', initialization: 'str') ->None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


class LearnableLocality(nn.Module):

    def __init__(self, input_dim, k):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.rand(k, input_dim)))
        self.smax = partial(entmax15, dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)
        return masked_x


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class AbstractLayer(nn.Module):

    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super().__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)
        x = self.fc(x.view(b, -1, 1))
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)
        x = sum([F.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])
        return x.squeeze(-1)


class BasicBlock(nn.Module):

    def __init__(self, input_dim, abstlay_dim_1, abstlay_dim_2, k, virtual_batch_size, fix_input_dim, drop_rate, block_activation):
        super().__init__()
        self.conv1 = AbstractLayer(input_dim, abstlay_dim_1, k, virtual_batch_size)
        self.conv2 = AbstractLayer(abstlay_dim_1, abstlay_dim_2, k, virtual_batch_size)
        self.downsample = nn.Sequential(nn.Dropout(drop_rate), AbstractLayer(fix_input_dim, abstlay_dim_2, k, virtual_batch_size))
        self.block_activation = block_activation

    def forward(self, x, pre_out=None):
        if pre_out is None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return self.block_activation(out)


class DANetBackbone(nn.Module):

    def __init__(self, n_continuous_features: 'int', cat_embedding_dims: 'list', n_layers: 'int', abstlay_dim_1: 'int', abstlay_dim_2: 'int', k: 'int', dropout_rate: 'float', block_activation: 'nn.Module', virtual_batch_size: 'int', embedding_dropout: 'float', batch_norm_continuous_input: 'bool'):
        super().__init__()
        self.cat_embedding_dims = cat_embedding_dims
        self.n_continuous_features = n_continuous_features
        self.input_dim = n_continuous_features + sum([y for x, y in cat_embedding_dims])
        self.n_layers = n_layers
        self.abstlay_dim_1 = abstlay_dim_1
        self.abstlay_dim_2 = abstlay_dim_2
        self.k = k
        self.dropout_rate = dropout_rate
        self.block_activation = block_activation
        self.virtual_batch_size = virtual_batch_size
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.embedding_dropout = embedding_dropout
        self.output_dim = self.abstlay_dim_2
        self._build_network()

    def _build_network(self):
        params = {'fix_input_dim': self.input_dim, 'k': self.k, 'virtual_batch_size': self.virtual_batch_size, 'abstlay_dim_1': self.abstlay_dim_1, 'abstlay_dim_2': self.abstlay_dim_2, 'drop_rate': self.dropout_rate, 'block_activation': self.block_activation}
        self.init_layer = BasicBlock(self.input_dim, **params)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.layers.append(BasicBlock(self.abstlay_dim_2, **params))

    def _build_embedding_layer(self):
        return Embedding1dLayer(continuous_dim=self.n_continuous_features, categorical_embedding_dims=self.cat_embedding_dims, embedding_dropout=self.embedding_dropout, batch_norm_continuous_input=self.batch_norm_continuous_input, virtual_batch_size=self.virtual_batch_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.init_layer(x)
        for layer in self.layers:
            out = layer(x, pre_out=out)
        return out


class FTTransformerBackbone(nn.Module):

    def __init__(self, config: 'DictConfig'):
        super().__init__()
        assert config.share_embedding_strategy in ['add', 'fraction'], f'`share_embedding_strategy` should be one of `add` or `fraction`, not {self.hparams.share_embedding_strategy}'
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.add_cls = AppendCLSToken(d_token=self.hparams.input_embed_dim, initialization=self.hparams.embedding_initialization)
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f'mha_block_{i}'] = TransformerEncoderBlock(input_embed_dim=self.hparams.input_embed_dim, num_heads=self.hparams.num_heads, ff_hidden_multiplier=self.hparams.ff_hidden_multiplier, ff_activation=self.hparams.transformer_activation, attn_dropout=self.hparams.attn_dropout, ff_dropout=self.hparams.ff_dropout, add_norm_dropout=self.hparams.add_norm_dropout, keep_attn=self.hparams.attn_feature_importance)
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        if self.hparams.attn_feature_importance:
            self.attention_weights_ = [None] * self.hparams.num_attn_blocks
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(self.hparams.continuous_dim, self.hparams.virtual_batch_size)
        self.output_dim = self.hparams.input_embed_dim

    def _build_embedding_layer(self):
        return Embedding2dLayer(continuous_dim=self.hparams.continuous_dim, categorical_cardinality=self.hparams.categorical_cardinality, embedding_dim=self.hparams.input_embed_dim, shared_embedding_strategy=self.hparams.share_embedding_strategy, frac_shared_embed=self.hparams.shared_embedding_fraction, embedding_bias=self.hparams.embedding_bias, batch_norm_continuous_input=self.hparams.batch_norm_continuous_input, embedding_dropout=self.hparams.embedding_dropout, initialization=self.hparams.embedding_initialization, virtual_batch_size=self.hparams.virtual_batch_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.add_cls(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if self.hparams.attn_feature_importance:
                self.attention_weights_[i] = block.mha.attn_weights
        if self.hparams.attn_feature_importance:
            self._calculate_feature_importance()
        return x[:, -1]

    def _calculate_feature_importance(self):
        n, h, f, _ = self.attention_weights_[0].shape
        device = self.attention_weights_[0].device
        L = len(self.attention_weights_)
        self.local_feature_importance = torch.zeros((n, f), device=device)
        for attn_weights in self.attention_weights_:
            self.local_feature_importance += attn_weights[:, :, :, -1].sum(dim=1)
        self.local_feature_importance = 1 / (h * L) * self.local_feature_importance[:, :-1]
        self.feature_importance_ = self.local_feature_importance.mean(dim=0).detach().cpu().numpy()


class GANDALFBackbone(nn.Module):

    def __init__(self, cat_embedding_dims: 'list', n_continuous_features: 'int', gflu_stages: 'int', gflu_dropout: 'float'=0.0, gflu_feature_init_sparsity: 'float'=0.3, learnable_sparsity: 'bool'=True, batch_norm_continuous_input: 'bool'=True, virtual_batch_size: 'int'=None, embedding_dropout: 'float'=0.0):
        super().__init__()
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = self.n_continuous_features + self._embedded_cat_features
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.virtual_batch_size = virtual_batch_size
        self._build_network()

    def _build_network(self):
        self.gflus = GatedFeatureLearningUnit(n_features_in=self.n_features, n_stages=self.gflu_stages, feature_mask_function=t_softmax, dropout=self.gflu_dropout, feature_sparsity=self.gflu_feature_init_sparsity, learnable_sparsity=self.learnable_sparsity)

    def _build_embedding_layer(self):
        return Embedding1dLayer(continuous_dim=self.n_continuous_features, categorical_embedding_dims=self.cat_embedding_dims, embedding_dropout=self.embedding_dropout, batch_norm_continuous_input=self.batch_norm_continuous_input, virtual_batch_size=self.virtual_batch_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.gflus(x)

    @property
    def feature_importance_(self):
        return self.gflus.feature_mask_function(self.gflus.feature_masks).sum(dim=0).detach().cpu().numpy()


class Entmoid15(Function):
    """A highly optimized equivalent of labda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmoid15 = Entmoid15.apply


class CustomHead(nn.Module):
    """Custom Head for GATE.

    Args:
        input_dim (int): Input dimension of the head
        hparams (DictConfig): Config of the model

    """

    def __init__(self, input_dim: 'int', hparams: 'DictConfig'):
        super().__init__()
        self.hparams = hparams
        self.input_dim = input_dim
        if self.hparams.share_head_weights:
            self.head = self._get_head_from_config()
        else:
            self.head = nn.ModuleList([self._get_head_from_config() for _ in range(self.hparams.num_trees)])
        self.eta = nn.Parameter(torch.rand(self.hparams.num_trees, requires_grad=True))
        if self.hparams.task == 'regression':
            self.T0 = nn.Parameter(torch.rand(self.hparams.output_dim), requires_grad=True)

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.hparams.head)
        return _head_callable(in_units=self.input_dim, output_dim=self.hparams.output_dim, config=_head_callable._config_template(**self.hparams.head_config))

    def forward(self, backbone_features: 'torch.Tensor') ->torch.Tensor:
        if not self.hparams.share_head_weights:
            y_hat = torch.cat([h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)], dim=1)
        else:
            y_hat = self.head(backbone_features.transpose(2, 1))
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        y_hat = y_hat.sum(dim=1)
        if self.hparams.task == 'regression':
            y_hat = y_hat + self.T0
        return y_hat


class NODEBackbone(nn.Module):

    def __init__(self, config: 'DictConfig', **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.hparams.node_input_dim = self.hparams.continuous_dim + self.hparams.embedded_cat_dim
        self.dense_block = DenseODSTBlock(input_dim=self.hparams.node_input_dim, num_trees=self.hparams.num_trees, num_layers=self.hparams.num_layers, tree_output_dim=self.hparams.output_dim + self.hparams.additional_tree_output_dim, max_features=self.hparams.max_features, input_dropout=self.hparams.input_dropout, depth=self.hparams.depth, choice_function=getattr(activations, self.hparams.choice_function), bin_function=getattr(activations, self.hparams.bin_function), initialize_response_=getattr(nn.init, self.hparams.initialize_response + '_'), initialize_selection_logits_=getattr(nn.init, self.hparams.initialize_selection_logits + '_'), threshold_init_beta=self.hparams.threshold_init_beta, threshold_init_cutoff=self.hparams.threshold_init_cutoff)
        self.output_dim = self.hparams.output_dim + self.hparams.additional_tree_output_dim

    def _build_embedding_layer(self):
        embedding = Embedding1dLayer(continuous_dim=self.hparams.continuous_dim, categorical_embedding_dims=self.hparams.embedding_dims, embedding_dropout=self.hparams.embedding_dropout, batch_norm_continuous_input=self.hparams.batch_norm_continuous_input, virtual_batch_size=self.hparams.virtual_batch_size)
        return embedding

    def forward(self, x: 'torch.Tensor'):
        x = self.dense_block(x)
        return x


class TabTransformerBackbone(nn.Module):

    def __init__(self, config: 'DictConfig'):
        super().__init__()
        assert config.share_embedding_strategy in ['add', 'fraction'], f'`share_embedding_strategy` should be one of `add` or `fraction`, not {self.hparams.share_embedding_strategy}'
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f'mha_block_{i}'] = TransformerEncoderBlock(input_embed_dim=self.hparams.input_embed_dim, num_heads=self.hparams.num_heads, ff_hidden_multiplier=self.hparams.ff_hidden_multiplier, ff_activation=self.hparams.transformer_activation, attn_dropout=self.hparams.attn_dropout, ff_dropout=self.hparams.ff_dropout, add_norm_dropout=self.hparams.add_norm_dropout, keep_attn=False)
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        self.attention_weights = [None] * self.hparams.num_attn_blocks
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(self.hparams.continuous_dim, self.hparams.virtual_batch_size)
        self.output_dim = self.hparams.input_embed_dim * self.hparams.categorical_dim + self.hparams.continuous_dim

    def _build_embedding_layer(self):
        return Embedding2dLayer(continuous_dim=0, categorical_cardinality=self.hparams.categorical_cardinality, embedding_dim=self.hparams.input_embed_dim, shared_embedding_strategy=self.hparams.share_embedding_strategy, frac_shared_embed=self.hparams.shared_embedding_fraction, embedding_bias=self.hparams.embedding_bias, embedding_dropout=self.hparams.embedding_dropout, initialization=self.hparams.embedding_initialization, virtual_batch_size=self.hparams.virtual_batch_size)

    def forward(self, x_cat: 'torch.Tensor', x_cont: 'torch.Tensor') ->torch.Tensor:
        x = None
        if self.hparams.categorical_dim > 0:
            for i, block in enumerate(self.transformer_blocks):
                x_cat = block(x_cat)
            x = rearrange(x_cat, 'b n h -> b (n h)')
        if self.hparams.continuous_dim > 0:
            if self.hparams.batch_norm_continuous_input:
                x_cont = self.normalizing_batch_norm(x_cont)
            else:
                x_cont = x_cont
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        return x


class TabNetBackbone(nn.Module):

    def __init__(self, config: 'DictConfig', **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        if self.hparams.grouped_features:
            features = self.hparams.categorical_cols + self.hparams.continuous_cols
            grp_list = [[features.index(col) for col in grp if col in features] for grp in self.hparams.grouped_features]
        else:
            grp_list = [[i] for i in range(self.hparams.continuous_dim + self.hparams.categorical_dim)]
        group_matrix = create_group_matrix(grp_list, self.hparams.continuous_dim + self.hparams.categorical_dim)
        self.tabnet = TabNet(input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim, output_dim=self.hparams.output_dim, n_d=self.hparams.n_d, n_a=self.hparams.n_a, n_steps=self.hparams.n_steps, gamma=self.hparams.gamma, cat_idxs=list(range(self.hparams.categorical_dim)), cat_dims=[cardinality for cardinality, _ in self.hparams.embedding_dims], cat_emb_dim=[embed_dim for _, embed_dim in self.hparams.embedding_dims], n_independent=self.hparams.n_independent, n_shared=self.hparams.n_shared, epsilon=1e-15, virtual_batch_size=self.hparams.virtual_batch_size, momentum=0.02, mask_type=self.hparams.mask_type, group_attention_matrix=group_matrix)

    def unpack_input(self, x: 'Dict'):
        x = x['categorical'], x['continuous']
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: 'Dict'):
        x = self.unpack_input(x)
        self.tabnet.embedder.embedding_group_matrix = self.tabnet.embedder.embedding_group_matrix
        self.tabnet.tabnet.encoder.group_attention_matrix = self.tabnet.tabnet.encoder.group_attention_matrix
        x, _ = self.tabnet(x)
        return x


class MultiTaskHead(nn.Module):
    """Simple Linear transformation to take last hidden representation to reconstruct inputs.

    Output is dictionary of variable type to tensor mapping.

    """

    def __init__(self, in_features, n_binary=0, n_categorical=0, n_numerical=0, cardinality=[]):
        super().__init__()
        assert n_categorical == len(cardinality), 'require cardinalities for each categorical variable'
        assert n_binary + n_categorical + n_numerical, 'need some targets'
        self.n_binary = n_binary
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical
        self.binary_linear = nn.Linear(in_features, n_binary) if n_binary else None
        self.categorical_linears = nn.ModuleList([nn.Linear(in_features, card) for card in cardinality])
        self.numerical_linear = nn.Linear(in_features, n_numerical) if n_numerical else None

    def forward(self, features):
        outputs = {}
        if self.binary_linear:
            outputs['binary'] = self.binary_linear(features)
        if self.categorical_linears:
            outputs['categorical'] = [linear(features) for linear in self.categorical_linears]
        if self.numerical_linear:
            outputs['continuous'] = self.numerical_linear(features)
        return outputs


class OneHot(nn.Module):

    def __init__(self, cardinality):
        super().__init__()
        self.cardinality = cardinality

    def forward(self, x):
        return F.one_hot(x, self.cardinality)


class MixedEmbedding1dLayer(nn.Module):
    """Enables different values in a categorical features to have different embeddings."""

    def __init__(self, continuous_dim: 'int', categorical_embedding_dims: 'Tuple[int, int]', max_onehot_cardinality: 'int'=4, embedding_dropout: 'float'=0.0, batch_norm_continuous_input: 'bool'=False, virtual_batch_size: 'int'=None):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.categorical_dim = len(categorical_embedding_dims)
        self.batch_norm_continuous_input = batch_norm_continuous_input
        binary_feat_idx = []
        onehot_feat_idx = []
        embedding_feat_idx = []
        embd_layers = {}
        one_hot_layers = {}
        for i, (cardinality, embed_dim) in enumerate(categorical_embedding_dims):
            if cardinality == 2:
                binary_feat_idx.append(i)
            elif cardinality <= max_onehot_cardinality:
                onehot_feat_idx.append(i)
                one_hot_layers[str(i)] = OneHot(cardinality)
            else:
                embedding_feat_idx.append(i)
                embd_layers[str(i)] = nn.Embedding(cardinality, embed_dim)
        if self.categorical_dim > 0:
            self.embedding_layer = nn.ModuleDict(embd_layers)
            self.one_hot_layers = nn.ModuleDict(one_hot_layers)
        self._onehot_feat_idx = onehot_feat_idx
        self._binary_feat_idx = binary_feat_idx
        self._embedding_feat_idx = embedding_feat_idx
        if embedding_dropout > 0 and len(embedding_feat_idx) > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)

    @property
    def embedded_cat_dim(self):
        return sum([embd_dim for i, (_, embd_dim) in enumerate(self.categorical_embedding_dims) if i in self._embedding_feat_idx])

    def forward(self, x: 'Dict[str, Any]') ->torch.Tensor:
        assert 'continuous' in x or 'categorical' in x, 'x must contain either continuous and categorical features'
        continuous_data, categorical_data = x.get('continuous', torch.empty(0, 0)), x.get('categorical', torch.empty(0, 0))
        assert categorical_data.shape[1] == len(self._onehot_feat_idx + self._binary_feat_idx + self._embedding_feat_idx), 'categorical_data must have same number of columns as categorical embedding layers'
        assert continuous_data.shape[1] == self.continuous_dim, 'continuous_data must have same number of columns as continuous dim'
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)
        if categorical_data.shape[1] > 0:
            x_cat = []
            x_cat_orig = []
            x_binary = []
            x_embed = []
            for i in range(self.categorical_dim):
                if i in self._binary_feat_idx:
                    x_binary.append(categorical_data[:, i:i + 1])
                elif i in self._onehot_feat_idx:
                    x_cat.append(self.one_hot_layers[str(i)](categorical_data[:, i]))
                    x_cat_orig.append(categorical_data[:, i:i + 1])
                else:
                    x_embed.append(self.embedding_layer[str(i)](categorical_data[:, i]))
            x_cat = torch.cat(x_cat, 1) if len(x_cat) > 0 else None
            x_cat_orig = torch.cat(x_cat_orig, 1) if len(x_cat_orig) > 0 else None
            x_binary = torch.cat(x_binary, 1) if len(x_binary) > 0 else None
            x_embed = torch.cat(x_embed, 1) if len(x_embed) > 0 else None
            all_none = x_cat is None and x_binary is None and x_embed is None
            assert not all_none, "All inputs can't be none!"
            if self.embd_dropout is not None:
                x_embed = self.embd_dropout(x_embed)
        else:
            x_cat = None
            x_cat_orig = None
            x_binary = None
            x_embed = None
        return OrderedDict(binary=x_binary, categorical=x_cat, _categorical_orig=x_cat_orig, continuous=continuous_data, embedding=x_embed)


class SwapNoiseCorrupter(nn.Module):
    """Apply swap noise on the input data.

    Each data point has specified chance be replaced by a random value from the same column.

    """

    def __init__(self, probas):
        super().__init__()
        self.probas = torch.from_numpy(np.array(probas))

    def forward(self, x):
        should_swap = torch.bernoulli(self.probas * torch.ones(x.shape))
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x, mask


class DenoisingAutoEncoderFeaturizer(nn.Module):
    output_tuple = namedtuple('output_tuple', ['features', 'mask'])
    output_tuple.__qualname__ = 'DenoisingAutoEncoderFeaturizer.output_tuple'

    def __init__(self, encoder, config: 'DictConfig', **kwargs):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.pick_keys = ['binary', 'categorical', 'continuous', 'embedding']
        self._build_network()

    def _get_noise_probability(self, name):
        return self.config.noise_probabilities.get(name, self.config.default_noise_probability)

    def _build_embedding_layer(self):
        return MixedEmbedding1dLayer(continuous_dim=self.config.continuous_dim, categorical_embedding_dims=self.config.embedding_dims, max_onehot_cardinality=self.config.max_onehot_cardinality, embedding_dropout=self.config.embedding_dropout, batch_norm_continuous_input=self.config.batch_norm_continuous_input, virtual_batch_size=self.config.virtual_batch_size)

    def _build_network(self):
        swap_probabilities = []
        for i, (name, (cardinality, embed_dim)) in enumerate(zip(self.config.categorical_cols, self.config.embedding_dims)):
            if cardinality == 2:
                swap_probabilities += [self._get_noise_probability(name)]
            elif cardinality <= self.config.max_onehot_cardinality:
                swap_probabilities += [self._get_noise_probability(name)] * cardinality
            else:
                swap_probabilities += [self._get_noise_probability(name)] * embed_dim
        for name in self.config.continuous_cols:
            swap_probabilities.append(self._get_noise_probability(name))
        self._swap_probabilities = swap_probabilities
        self.swap_noise = SwapNoiseCorrupter(swap_probabilities)

    def _concatenate_features(self, x: 'Dict'):
        x = torch.cat([x[key] for key in self.pick_keys if x[key] is not None], 1)
        return x

    def forward(self, x: 'Dict', perturb: 'bool'=True, return_input: 'bool'=False):
        x = self._concatenate_features(x)
        mask = None
        if perturb:
            with torch.no_grad():
                x, mask = self.swap_noise(x)
        z = self.encoder(x)
        if return_input:
            return self.output_tuple(z, mask), x
        else:
            return self.output_tuple(z, mask)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Add,
     lambda: ([], {'add_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AddNorm,
     lambda: ([], {'input_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GBN,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Head,
     lambda: ([], {'layers': torch.nn.ReLU(), 'config_template': SimpleNamespace()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Lambda,
     lambda: ([], {'func': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (PositionWiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ReGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

