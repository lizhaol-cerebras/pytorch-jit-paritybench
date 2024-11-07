
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


from collections import namedtuple


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import time


from functools import partial


import numpy as np


from torch.nn.functional import softplus


import logging


import pandas as pd


from torch.distributions import constraints


from torch.distributions import biject_to


import functools


from torchvision import transforms


import matplotlib.pyplot as plt


from torch.optim import Adam


import torch.autograd as autograd


import torch.optim as optim


from torch.distributions import transform_to


import copy


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import functional


from torchvision.utils import make_grid


from torch.distributions import transforms


from torch.autograd import grad


from torch import nn


import uuid


from matplotlib.gridspec import GridSpec


from torch.distributions.utils import broadcast_all


import collections


import numbers


import queue


from matplotlib.patches import Patch


from torch.nn.functional import softmax


import re


from collections import OrderedDict


import torch.multiprocessing as mp


from typing import Callable


from torch.distributions.bernoulli import Bernoulli


from torch.distributions.beta import Beta


from inspect import isclass


from functools import reduce


import itertools


from abc import ABCMeta


from abc import abstractmethod


from torch.nn import functional


from torchvision.utils import save_image


from torch.autograd import Variable


from torch.distributions.utils import lazy_property


from torch.utils.data import TensorDataset


from collections import defaultdict


import warnings


from abc import ABC


from torch.nn.functional import pad


from torch.nn.utils.rnn import pad_sequence


from scipy import sparse


import torchvision.datasets as datasets


from functools import singledispatch


from numbers import Number


from torch.nn import Parameter


import torch.distributions as torchdist


import random


from torch.distributions import Categorical


from torch.distributions import OneHotCategorical


from torch.distributions.transforms import AffineTransform


from torch.distributions.transforms import SigmoidTransform


from typing import Union


from torch import Tensor


from torch.distributions import MultivariateNormal


from torch import Size


from torch.autograd import Function


from torch.autograd.function import once_differentiable


import torch.nn


from torch.distributions.constraints import *


from torch.distributions.constraints import __all__ as torch_constraints


import inspect


from torch.distributions.transforms import AbsTransform


from torch.distributions.transforms import PowerTransform


from torch.distributions import Independent


from torch.distributions import Normal


from torch.distributions import kl_divergence


from torch.distributions import register_kl


from torch.nn.functional import logsigmoid


from torch.distributions.utils import clamp_probs


from typing import NamedTuple


from typing import Optional


from math import pi


from torch.distributions import VonMises


from torch import broadcast_shapes


from torch.distributions import Uniform


from torch.distributions.kl import kl_divergence


from torch.distributions.kl import register_kl


from torch.distributions.transforms import *


from torch.distributions.transforms import __all__ as torch_transforms


from torch.distributions.utils import _sum_rightmost


from torch.distributions.transforms import TanhTransform


from torch.distributions.transforms import Transform


from torch.distributions.transforms import CorrCholeskyTransform


from torch.distributions import Transform


from torch.distributions import Distribution


from torch.special import expit


from torch.special import logit


import torch.distributions as torch_dist


from torch import logsumexp


from torch.distributions.utils import logits_to_probs


from torch.distributions.utils import probs_to_logits


from typing import Tuple


from types import SimpleNamespace


from typing import Dict


from typing import Set


from typing import List


from typing import Collection


from itertools import product


from torch.distributions.transforms import ComposeTransform


from collections import Counter


from typing import Sequence


from torch.nn import functional as F


from types import TracebackType


from typing import TYPE_CHECKING


from typing import Any


from typing import Iterator


from typing import Type


from typing import TypeVar


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_map


from torch.utils._pytree import tree_unflatten


from numpy.polynomial.hermite import hermgauss


from torch.fft import irfft


from torch.fft import rfft


from typing import Hashable


from torch.optim.optimizer import Optimizer


from typing import ValuesView


from torch.optim import Optimizer


from typing import Iterable


from torch.nn.utils import clip_grad_norm_


from torch.nn.utils import clip_grad_value_


from typing import ItemsView


from typing import KeysView


from torch.serialization import MAP_LOCATION


from typing import FrozenSet


from typing import Literal


from typing import overload


from typing import Generic


from itertools import zip_longest


import torch.cuda


from numpy.testing import assert_allclose


from queue import LifoQueue


import scipy.stats as sp


from torch.distributions import AffineTransform


from torch.distributions import Beta


from torch.distributions import TransformedDistribution


from torch import tensor


from torch.distributions import Gamma


from torch.distributions import StudentT


from torch.autograd.functional import jacobian


from scipy.special import binom


from torch.distributions import HalfNormal


from scipy.integrate.quadpack import IntegrationWarning


from scipy.stats import ks_2samp


from scipy.stats import kstest


from scipy.stats import levy_stable


from torch import optim


import torch.distributions as dist


import numpy


import scipy.special as sc


import torch.optim


from torch import nn as nn


from scipy.special import iv


import scipy.fftpack as fftpack


from copy import copy


from queue import Queue


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class Encoder(nn.Module):

    def __init__(self, data_length, alphabet_length, z_dim):
        super().__init__()
        self.input_size = data_length * alphabet_length
        self.f1_mn = nn.Linear(self.input_size, z_dim)
        self.f1_sd = nn.Linear(self.input_size, z_dim)

    def forward(self, data):
        data = data.reshape(-1, self.input_size)
        z_loc = self.f1_mn(data)
        z_scale = softplus(self.f1_sd(data))
        return z_loc, z_scale


GuideState = namedtuple('GuideState', ['h', 'c', 'bl_h', 'bl_c', 'z_pres', 'z_where', 'z_what'])


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.

    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop('strict', False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError('shape mismatch: objects cannot be broadcast to a single shape: {}'.format(' vs '.join(map(str, shapes))))
    return tuple(reversed(reversed_shape))


class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        if len(input_args) == 1:
            input_args = input_args[0]
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax

    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


class MLP(nn.Module):

    def __init__(self, mlp_sizes, activation=nn.ReLU, output_activation=None, post_layer_fct=lambda layer_ix, total_layers, layer: None, post_act_fct=lambda layer_ix, total_layers, layer: None, allow_broadcast=False, use_cuda=False):
        super().__init__()
        assert len(mlp_sizes) >= 2, 'Must have input and output layer sizes defined'
        input_size, hidden_sizes, output_size = mlp_sizes[0], mlp_sizes[1:-1], mlp_sizes[-1]
        assert isinstance(input_size, (int, list, tuple)), 'input_size must be int, list, tuple'
        last_layer_size = input_size if type(input_size) == int else sum(input_size)
        all_modules = [ConcatModule(allow_broadcast)]
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, 'Hidden layer sizes must be ints'
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)
            all_modules.append(cur_linear_layer)
            post_linear = post_layer_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])
            if post_linear is not None:
                all_modules.append(post_linear)
            all_modules.append(activation())
            post_activation = post_act_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])
            if post_activation is not None:
                all_modules.append(post_activation)
            last_layer_size = layer_size
        assert isinstance(output_size, (int, list, tuple)), 'output_size must be int, list, tuple'
        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(call_nn_op(output_activation) if isclass(output_activation) else output_activation)
        else:
            out_layers = []
            for out_ix, out_size in enumerate(output_size):
                split_layer = []
                split_layer.append(nn.Linear(last_layer_size, out_size))
                act_out_fct = output_activation if not isinstance(output_activation, (list, tuple)) else output_activation[out_ix]
                if act_out_fct:
                    split_layer.append(call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct)
                out_layers.append(nn.Sequential(*split_layer))
            all_modules.append(ListOutModule(out_layers))
        self.sequential_mlp = nn.Sequential(*all_modules)

    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)


ModelState = namedtuple('ModelState', ['x', 'z_pres', 'z_where'])


class Predict(nn.Module):

    def __init__(self, input_size, h_sizes, z_pres_size, z_where_size, non_linear_layer):
        super().__init__()
        self.z_pres_size = z_pres_size
        self.z_where_size = z_where_size
        output_size = z_pres_size + 2 * z_where_size
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        z_pres_p = torch.sigmoid(out[:, 0:self.z_pres_size])
        z_where_loc = out[:, self.z_pres_size:self.z_pres_size + self.z_where_size]
        z_where_scale = softplus(out[:, self.z_pres_size + self.z_where_size:])
        return z_pres_p, z_where_loc, z_where_scale


def default_z_pres_prior_p(t):
    return 0.5


expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])


def expand_z_where(z_where):
    n = z_where.size(0)
    out = torch.cat((z_where.new_zeros(n, 1), z_where), 1)
    ix = expansion_indices
    if z_where.is_cuda:
        ix = ix
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out


def z_where_inv(z_where):
    n = z_where.size(0)
    out = torch.cat((z_where.new_ones(n, 1), -z_where[:, 1:]), 1)
    out = out / z_where[:, 0:1]
    return out


def image_to_window(z_where, window_size, image_size, images):
    n = images.size(0)
    assert images.size(1) == images.size(2) == image_size, 'Size mismatch.'
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = F.affine_grid(theta_inv, torch.Size((n, 1, window_size, window_size)))
    out = F.grid_sample(images.view(n, 1, image_size, image_size), grid)
    return out.view(n, -1)


def window_to_image(z_where, window_size, image_size, windows):
    n = windows.size(0)
    assert windows.size(1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(n, image_size, image_size)


class AIR(nn.Module):

    def __init__(self, num_steps, x_size, window_size, z_what_size, rnn_hidden_size, encoder_net=[], decoder_net=[], predict_net=[], embed_net=None, bl_predict_net=[], non_linearity='ReLU', decoder_output_bias=None, decoder_output_use_sigmoid=False, use_masking=True, use_baselines=True, baseline_scalar=None, scale_prior_mean=3.0, scale_prior_sd=0.1, pos_prior_mean=0.0, pos_prior_sd=1.0, likelihood_sd=0.3, use_cuda=False):
        super().__init__()
        self.num_steps = num_steps
        self.x_size = x_size
        self.window_size = window_size
        self.z_what_size = z_what_size
        self.rnn_hidden_size = rnn_hidden_size
        self.use_masking = use_masking
        self.use_baselines = use_baselines
        self.baseline_scalar = baseline_scalar
        self.likelihood_sd = likelihood_sd
        self.use_cuda = use_cuda
        prototype = torch.tensor(0.0) if use_cuda else torch.tensor(0.0)
        self.options = dict(dtype=prototype.dtype, device=prototype.device)
        self.z_pres_size = 1
        self.z_where_size = 3
        self.z_where_loc_prior = nn.Parameter(torch.FloatTensor([scale_prior_mean, pos_prior_mean, pos_prior_mean]), requires_grad=False)
        self.z_where_scale_prior = nn.Parameter(torch.FloatTensor([scale_prior_sd, pos_prior_sd, pos_prior_sd]), requires_grad=False)
        rnn_input_size = x_size ** 2 if embed_net is None else embed_net[-1]
        rnn_input_size += self.z_where_size + z_what_size + self.z_pres_size
        nl = getattr(nn, non_linearity)
        self.rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.encode = Encoder(window_size ** 2, encoder_net, z_what_size, nl)
        self.decode = Decoder(window_size ** 2, decoder_net, z_what_size, decoder_output_bias, decoder_output_use_sigmoid, nl)
        self.predict = Predict(rnn_hidden_size, predict_net, self.z_pres_size, self.z_where_size, nl)
        self.embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)
        self.bl_rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.bl_predict = MLP(rnn_hidden_size, bl_predict_net + [1], nl)
        self.bl_embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)
        self.h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.bl_h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.bl_c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.z_where_init = nn.Parameter(torch.zeros(1, self.z_where_size))
        self.z_what_init = nn.Parameter(torch.zeros(1, self.z_what_size))
        if use_cuda:
            self

    def prior(self, n, **kwargs):
        state = ModelState(x=torch.zeros(n, self.x_size, self.x_size, **self.options), z_pres=torch.ones(n, self.z_pres_size, **self.options), z_where=None)
        z_pres = []
        z_where = []
        for t in range(self.num_steps):
            state = self.prior_step(t, n, state, **kwargs)
            z_where.append(state.z_where)
            z_pres.append(state.z_pres)
        return (z_where, z_pres), state.x

    def prior_step(self, t, n, prev, z_pres_prior_p=default_z_pres_prior_p):
        z_pres = pyro.sample('z_pres_{}'.format(t), dist.Bernoulli(z_pres_prior_p(t) * prev.z_pres).to_event(1))
        sample_mask = z_pres if self.use_masking else torch.tensor(1.0)
        z_where = pyro.sample('z_where_{}'.format(t), dist.Normal(self.z_where_loc_prior.expand(n, self.z_where_size), self.z_where_scale_prior.expand(n, self.z_where_size)).mask(sample_mask).to_event(1))
        z_what = pyro.sample('z_what_{}'.format(t), dist.Normal(torch.zeros(n, self.z_what_size, **self.options), torch.ones(n, self.z_what_size, **self.options)).mask(sample_mask).to_event(1))
        y_att = self.decode(z_what)
        y = window_to_image(z_where, self.window_size, self.x_size, y_att)
        x = prev.x + y * z_pres.view(-1, 1, 1)
        return ModelState(x=x, z_pres=z_pres, z_where=z_where)

    def model(self, data, batch_size, **kwargs):
        pyro.module('decode', self.decode)
        with pyro.plate('data', data.size(0), device=data.device) as ix:
            batch = data[ix]
            n = batch.size(0)
            (z_where, z_pres), x = self.prior(n, **kwargs)
            pyro.sample('obs', dist.Normal(x.view(n, -1), self.likelihood_sd * torch.ones(n, self.x_size ** 2, **self.options)).to_event(1), obs=batch.view(n, -1))

    def guide(self, data, batch_size, **kwargs):
        pyro.module('rnn', self.rnn),
        pyro.module('predict', self.predict),
        pyro.module('encode', self.encode),
        pyro.module('embed', self.embed),
        pyro.module('bl_rnn', self.bl_rnn),
        pyro.module('bl_predict', self.bl_predict),
        pyro.module('bl_embed', self.bl_embed)
        pyro.param('h_init', self.h_init)
        pyro.param('c_init', self.c_init)
        pyro.param('z_where_init', self.z_where_init)
        pyro.param('z_what_init', self.z_what_init)
        pyro.param('bl_h_init', self.bl_h_init)
        pyro.param('bl_c_init', self.bl_c_init)
        with pyro.plate('data', data.size(0), subsample_size=batch_size, device=data.device) as ix:
            batch = data[ix]
            n = batch.size(0)
            flattened_batch = batch.view(n, -1)
            inputs = {'raw': batch, 'embed': self.embed(flattened_batch), 'bl_embed': self.bl_embed(flattened_batch)}
            state = GuideState(h=self.h_init.expand(n, -1), c=self.c_init.expand(n, -1), bl_h=self.bl_h_init.expand(n, -1), bl_c=self.bl_c_init.expand(n, -1), z_pres=torch.ones(n, self.z_pres_size, **self.options), z_where=self.z_where_init.expand(n, -1), z_what=self.z_what_init.expand(n, -1))
            z_pres = []
            z_where = []
            for t in range(self.num_steps):
                state = self.guide_step(t, n, state, inputs)
                z_where.append(state.z_where)
                z_pres.append(state.z_pres)
            return z_where, z_pres

    def guide_step(self, t, n, prev, inputs):
        rnn_input = torch.cat((inputs['embed'], prev.z_where, prev.z_what, prev.z_pres), 1)
        h, c = self.rnn(rnn_input, (prev.h, prev.c))
        z_pres_p, z_where_loc, z_where_scale = self.predict(h)
        infer_dict, bl_h, bl_c = self.baseline_step(prev, inputs)
        z_pres = pyro.sample('z_pres_{}'.format(t), dist.Bernoulli(z_pres_p * prev.z_pres).to_event(1), infer=infer_dict)
        sample_mask = z_pres if self.use_masking else torch.tensor(1.0)
        z_where = pyro.sample('z_where_{}'.format(t), dist.Normal(z_where_loc + self.z_where_loc_prior, z_where_scale * self.z_where_scale_prior).mask(sample_mask).to_event(1))
        x_att = image_to_window(z_where, self.window_size, self.x_size, inputs['raw'])
        z_what_loc, z_what_scale = self.encode(x_att)
        z_what = pyro.sample('z_what_{}'.format(t), dist.Normal(z_what_loc, z_what_scale).mask(sample_mask).to_event(1))
        return GuideState(h=h, c=c, bl_h=bl_h, bl_c=bl_c, z_pres=z_pres, z_where=z_where, z_what=z_what)

    def baseline_step(self, prev, inputs):
        if not self.use_baselines:
            return dict(), None, None
        rnn_input = torch.cat((inputs['bl_embed'], prev.z_where.detach(), prev.z_what.detach(), prev.z_pres.detach()), 1)
        bl_h, bl_c = self.bl_rnn(rnn_input, (prev.bl_h, prev.bl_c))
        bl_value = self.bl_predict(bl_h)
        if self.use_masking:
            bl_value = bl_value * prev.z_pres
        if self.baseline_scalar is not None:
            bl_value = bl_value * self.baseline_scalar
        infer_dict = dict(baseline=dict(baseline_value=bl_value.squeeze(-1)))
        return infer_dict, bl_h, bl_c


class TonesGenerator(nn.Module):

    def __init__(self, args, data_dim):
        self.args = args
        self.data_dim = data_dim
        super().__init__()
        self.x_to_hidden = nn.Linear(args.hidden_dim, args.nn_dim)
        self.y_to_hidden = nn.Linear(args.nn_channels * data_dim, args.nn_dim)
        self.conv = nn.Conv1d(1, args.nn_channels, 3, padding=1)
        self.hidden_to_logits = nn.Linear(args.nn_dim, data_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args.hidden_dim,)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.reshape(-1, 1, self.data_dim))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineNet(nn.Module):

    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y


class MaskedBCELoss(nn.Module):

    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        loss = F.binary_cross_entropy(input[target != self.masked_with], target[target != self.masked_with], reduction='none')
        return loss.sum()


class CVAE(nn.Module):

    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        pyro.module('generation_net', self)
        batch_size = xs.shape[0]
        with pyro.plate('data'):
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(xs.shape)
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))
            loc = self.generation_net(zs)
            if ys is not None:
                mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample('y', dist.Bernoulli(mask_loc, validate_args=False).to_event(1), obs=mask_ys)
            else:
                pyro.deterministic('y', loc.detach())
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate('data'):
            if ys is None:
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                loc, scale = self.recognition_net(xs, ys)
            pyro.sample('z', dist.Normal(loc, scale).to_event(1))


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()


def clamp_preserve_gradients(x, min, max):
    return x + (x.clamp(min, max) - x).detach()


def copy_docs_from(source_class, full_text=False):
    """
    Decorator to copy class and method docs from source to destin class.
    """

    def decorator(destin_class):
        for name in dir(destin_class):
            if name.startswith('_'):
                continue
            destin_attr = getattr(destin_class, name)
            destin_attr = getattr(destin_attr, '__func__', destin_attr)
            source_attr = getattr(source_class, name, None)
            source_doc = getattr(source_attr, '__doc__', None)
            if source_doc and not getattr(destin_attr, '__doc__', None):
                if full_text or source_doc.startswith('See '):
                    destin_doc = source_doc
                else:
                    destin_doc = 'See :meth:`{}.{}.{}`'.format(source_class.__module__, source_class.__name__, name)
                if isinstance(destin_attr, property):
                    updated_property = property(destin_attr.fget, destin_attr.fset, destin_attr.fdel, destin_doc)
                    setattr(destin_class, name, updated_property)
                else:
                    destin_attr.__doc__ = destin_doc
        return destin_class
    return decorator


@copy_docs_from(TransformModule)
class AffineAutoregressive(TransformModule):
    """
    An implementation of the bijective transform of Inverse Autoregressive Flow
    (IAF), using by default Eq (10) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\mu_t + \\sigma_t\\odot\\mathbf{x}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    :math:`\\mu_t,\\sigma_t` are calculated from an autoregressive network on
    :math:`\\mathbf{x}`, and :math:`\\sigma_t>0`.

    If the stable keyword argument is set to True then the transformation used is,

        :math:`\\mathbf{y} = \\sigma_t\\odot\\mathbf{x} + (1-\\sigma_t)\\odot\\mu_t`

    where :math:`\\sigma_t` is restricted to :math:`(0,1)`. This variant of IAF is
    claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be
    represented, presumably since the input is restricted to rescaling by a number
    on :math:`(0,1)`.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitrary value is being scored, it will calculate it manually. Note
    that this is an operation that scales as O(D) where D is the input dimension,
    and so should be avoided for large dimensional uses. So in general, it is cheap
    to sample from IAF and score a value that was sampled by IAF, but expensive to
    score an arbitrary value.

    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns a real-valued mean and logit-scale as a tuple
    :type autoregressive_nn: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    References:

    [1] Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever,
    Max Welling. Improving Variational Inference with Inverse Autoregressive Flow.
    [arXiv:1606.04934]

    [2] Danilo Jimenez Rezende, Shakir Mohamed. Variational Inference with
    Normalizing Flows. [arXiv:1505.05770]

    [3] Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. MADE: Masked
    Autoencoder for Distribution Estimation. [arXiv:1502.03509]

    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1
    autoregressive = True

    def __init__(self, autoregressive_nn, log_scale_min_clip=-5.0, log_scale_max_clip=3.0, sigmoid_bias=2.0, stable=False):
        super().__init__(cache_size=1)
        self.arn = autoregressive_nn
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid_bias = sigmoid_bias
        self.stable = stable
        if stable:
            self._call = self._call_stable
            self._inverse = self._inverse_stable

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        mean, log_scale = self.arn(x)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        scale = torch.exp(log_scale)
        y = scale * x + mean
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim
        for idx in perm:
            mean, log_scale = self.arn(torch.stack(x, dim=-1))
            inverse_scale = torch.exp(-clamp_preserve_gradients(log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            x[idx] = (y[..., idx] - mean) * inverse_scale
        x = torch.stack(x, dim=-1)
        log_scale = clamp_preserve_gradients(log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip)
        self._cached_log_scale = log_scale
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        elif not self.stable:
            _, log_scale = self.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        else:
            _, logit_scale = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)

    def _call_stable(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        mean, logit_scale = self.arn(x)
        logit_scale = logit_scale + self.sigmoid_bias
        scale = self.sigmoid(logit_scale)
        log_scale = self.logsigmoid(logit_scale)
        self._cached_log_scale = log_scale
        y = scale * x + (1 - scale) * mean
        return y

    def _inverse_stable(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim
        for idx in perm:
            mean, logit_scale = self.arn(torch.stack(x, dim=-1))
            inverse_scale = 1 + torch.exp(-logit_scale[..., idx] - self.sigmoid_bias)
            x[idx] = inverse_scale * y[..., idx] + (1 - inverse_scale) * mean[..., idx]
        self._cached_log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        x = torch.stack(x, dim=-1)
        return x


class MaskedLinear(nn.Linear):
    """
    A linear mapping with a given mask on the weights (arbitrary bias)

    :param in_features: the number of input features
    :type in_features: int
    :param out_features: the number of output features
    :type out_features: int
    :param mask: the mask to apply to the in_features x out_features weight matrix
    :type mask: torch.Tensor
    :param bias: whether or not `MaskedLinear` should include a bias term. defaults to `True`
    :type bias: bool
    """

    def __init__(self, in_features: 'int', out_features: 'int', mask: 'torch.Tensor', bias: 'bool'=True) ->None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask.data)

    def forward(self, _input: 'torch.Tensor') ->torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.linear(_input, masked_weight, self.bias)


def sample_mask_indices(input_dim: 'int', hidden_dim: 'int', simple: 'bool'=True) ->torch.Tensor:
    """
    Samples the indices assigned to hidden units during the construction of MADE masks

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dim: the dimensionality of the hidden layer
    :type hidden_dim: int
    :param simple: True to space fractional indices by rounding to nearest int, false round randomly
    :type simple: bool
    """
    indices = torch.linspace(1, input_dim, steps=hidden_dim)
    if simple:
        return torch.round(indices)
    else:
        ints = indices.floor()
        ints += torch.bernoulli(indices - ints)
        return ints


def create_mask(input_dim: 'int', context_dim: 'int', hidden_dims: 'List[int]', permutation: 'torch.LongTensor', output_dim_multiplier: 'int') ->Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Creates MADE masks for a conditional distribution

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param context_dim: the dimensionality of the variable that is conditioned on (for conditional densities)
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden layers(s)
    :type hidden_dims: list[int]
    :param permutation: the order of the input variables
    :type permutation: torch.LongTensor
    :param output_dim_multiplier: tiles the output (e.g. for when a separate mean and scale parameter are desired)
    :type output_dim_multiplier: int
    """
    var_index = torch.empty(permutation.shape, dtype=torch.get_default_dtype())
    var_index[permutation] = torch.arange(input_dim, dtype=torch.get_default_dtype())
    input_indices = torch.cat((torch.zeros(context_dim), 1 + var_index))
    if context_dim > 0:
        hidden_indices = [(sample_mask_indices(input_dim, h) - 1) for h in hidden_dims]
    else:
        hidden_indices = [sample_mask_indices(input_dim - 1, h) for h in hidden_dims]
    output_indices = (var_index + 1).repeat(output_dim_multiplier)
    mask_skip = (output_indices.unsqueeze(-1) > input_indices.unsqueeze(0)).type_as(var_index)
    masks = [(hidden_indices[0].unsqueeze(-1) >= input_indices.unsqueeze(0)).type_as(var_index)]
    for i in range(1, len(hidden_dims)):
        masks.append((hidden_indices[i].unsqueeze(-1) >= hidden_indices[i - 1].unsqueeze(0)).type_as(var_index))
    masks.append((output_indices.unsqueeze(-1) > hidden_indices[-1].unsqueeze(0)).type_as(var_index))
    return masks, mask_skip


class ConditionalAutoRegressiveNN(nn.Module):
    """
    An implementation of a MADE-like auto-regressive neural network that can input an additional context variable.
    (See Reference [2] Section 3.3 for an explanation of how the conditional MADE architecture works.)

    Example usage:

    >>> x = torch.randn(100, 10)
    >>> y = torch.randn(100, 5)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1])
    >>> p = arn(x, context=y)  # 1 parameters of size (100, 10)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1, 1])
    >>> m, s = arn(x, context=y) # 2 parameters of size (100, 10)
    >>> arn = ConditionalAutoRegressiveNN(10, 5, [50], param_dims=[1, 5, 3])
    >>> a, b, c = arn(x, context=y) # 3 parameters of sizes, (100, 1, 10), (100, 5, 10), (100, 3, 10)

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    Reference:

    1. MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    2. Inference Networks for Sequential Monte Carlo in Graphical Models [arXiv:1602.06701]
    Brooks Paige, Frank Wood

    """

    def __init__(self, input_dim: 'int', context_dim: 'int', hidden_dims: 'List[int]', param_dims: 'List[int]'=[1, 1], permutation: 'Optional[torch.LongTensor]'=None, skip_connections: 'bool'=False, nonlinearity: 'torch.nn.Module'=nn.ReLU()) ->None:
        super().__init__()
        if input_dim == 1:
            warnings.warn('ConditionalAutoRegressiveNN input_dim = 1. Consider using an affine transformation instead.')
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)
        self.all_ones = (torch.tensor(param_dims) == 1).all().item()
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]
        for h in hidden_dims:
            if h < input_dim:
                raise ValueError('Hidden dimension must not be less than input dimension.')
        if permutation is None:
            P = torch.randperm(input_dim, device='cpu')
        else:
            P = permutation.type(dtype=torch.int64)
        self.permutation: 'torch.LongTensor'
        self.register_buffer('permutation', P)
        self.masks, self.mask_skip = create_mask(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims, permutation=self.permutation, output_dim_multiplier=self.output_multiplier)
        layers = [MaskedLinear(input_dim + context_dim, hidden_dims[0], self.masks[0])]
        for i in range(1, len(hidden_dims)):
            layers.append(MaskedLinear(hidden_dims[i - 1], hidden_dims[i], self.masks[i]))
        layers.append(MaskedLinear(hidden_dims[-1], input_dim * self.output_multiplier, self.masks[-1]))
        self.layers = nn.ModuleList(layers)
        self.skip_layer: 'Optional[MaskedLinear]'
        if skip_connections:
            self.skip_layer = MaskedLinear(input_dim + context_dim, input_dim * self.output_multiplier, self.mask_skip, bias=False)
        else:
            self.skip_layer = None
        self.f = nonlinearity

    def get_permutation(self) ->torch.LongTensor:
        """
        Get the permutation applied to the inputs (by default this is chosen at random)
        """
        return self.permutation

    def forward(self, x: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None) ->Union[Sequence[torch.Tensor], torch.Tensor]:
        if context is None:
            context = self.context
        context = context.expand(x.size()[:-1] + (context.size(-1),))
        x = torch.cat([context, x], dim=-1)
        return self._forward(x)

    def _forward(self, x: 'torch.Tensor') ->Union[Sequence[torch.Tensor], torch.Tensor]:
        h = x
        for layer in self.layers[:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)
        if self.skip_layer is not None:
            h = h + self.skip_layer(x)
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier, self.input_dim])
            if self.count_params == 1:
                return h
            elif self.all_ones:
                return torch.unbind(h, dim=-2)
            else:
                return tuple([h[..., s, :] for s in self.param_slices])


class AutoRegressiveNN(ConditionalAutoRegressiveNN):
    """
    An implementation of a MADE-like auto-regressive neural network.

    Example usage:

    >>> x = torch.randn(100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1])
    >>> p = arn(x)  # 1 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 1])
    >>> m, s = arn(x) # 2 parameters of size (100, 10)
    >>> arn = AutoRegressiveNN(10, [50], param_dims=[1, 5, 3])
    >>> a, b, c = arn(x) # 3 parameters of sizes, (100, 1, 10), (100, 5, 10), (100, 3, 10)

    :param input_dim: the dimensionality of the input variable
    :type input_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n, input_dim) for p_n in param_dims
        when p_n > 1 and dimension (input_dim) when p_n == 1. The default is [1, 1], i.e. output two parameters
        of dimension (input_dim), which is useful for inverse autoregressive flow.
    :type param_dims: list[int]
    :param permutation: an optional permutation that is applied to the inputs and controls the order of the
        autoregressive factorization. in particular for the identity permutation the autoregressive structure
        is such that the Jacobian is upper triangular. By default this is chosen at random.
    :type permutation: torch.LongTensor
    :param skip_connections: Whether to add skip connections from the input to the output.
    :type skip_connections: bool
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.module

    Reference:

    MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle

    """

    def __init__(self, input_dim: 'int', hidden_dims: 'List[int]', param_dims: 'List[int]'=[1, 1], permutation: 'Optional[torch.LongTensor]'=None, skip_connections: 'bool'=False, nonlinearity: 'torch.nn.Module'=nn.ReLU()) ->None:
        super(AutoRegressiveNN, self).__init__(input_dim, 0, hidden_dims, param_dims=param_dims, permutation=permutation, skip_connections=skip_connections, nonlinearity=nonlinearity)

    def forward(self, x: 'torch.Tensor') ->Union[Sequence[torch.Tensor], torch.Tensor]:
        return self._forward(x)


def affine_autoregressive(input_dim, hidden_dims=None, **kwargs):
    """
    A helper function to create an
    :class:`~pyro.distributions.transforms.AffineAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float
    :param sigmoid_bias: A term to add the logit of the input when using the stable
        tranform.
    :type sigmoid_bias: float
    :param stable: When true, uses the alternative "stable" version of the transform
        (see above).
    :type stable: bool

    """
    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims)
    return AffineAutoregressive(arn, **kwargs)


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100, transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.0, num_iafs=0, iaf_dim=50, use_cuda=False):
        super().__init__()
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu', batch_first=True, bidirectional=False, num_layers=num_layers, dropout=rnn_dropout_rate)
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self

    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        T_max = mini_batch.size(1)
        pyro.module('dmm', self)
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))
        with pyro.plate('z_minibatch', len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.trans(z_prev)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample('z_%d' % t, dist.Normal(z_loc, z_scale).mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                emission_probs_t = self.emitter(z_t)
                pyro.sample('obs_x_%d' % t, dist.Bernoulli(emission_probs_t).mask(mini_batch_mask[:, t - 1:t]).to_event(1), obs=mini_batch[:, t - 1, :])
                z_prev = z_t

    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        T_max = mini_batch.size(1)
        pyro.module('dmm', self)
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        with pyro.plate('z_minibatch', len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (len(mini_batch), self.z_q_0.size(0))
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        z_t = pyro.sample('z_%d' % t, z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        z_t = pyro.sample('z_%d' % t, z_dist.mask(mini_batch_mask[:, t - 1:t]).to_event(1))
                z_prev = z_t


def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


class Z2Decoder(nn.Module):

    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        self.fc = make_fc(dims)

    def forward(self, z1, y):
        z1_y = torch.cat([z1, y], dim=-1)
        _z1_y = z1_y.reshape(-1, z1_y.size(-1))
        hidden = self.fc(_z1_y)
        hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale)
        return loc, scale


class XDecoder(nn.Module):

    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        gate_logits, mu = split_in_half(self.fc(z2))
        mu = softmax(mu, dim=-1)
        return gate_logits, mu


class Z2LEncoder(nn.Module):

    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
        self.fc = make_fc(dims)

    def forward(self, x):
        x = torch.log1p(x)
        h1, h2 = split_in_half(self.fc(x))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1])
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
        return z2_loc, z2_scale, l_loc, l_scale


def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args


class Z1Encoder(nn.Module):

    def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
        super().__init__()
        dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
        self.fc = make_fc(dims)

    def forward(self, z2, y):
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = softplus(scale)
        return loc, scale


class Classifier(nn.Module):

    def __init__(self, z2_dim, hidden_dims, num_labels):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_labels]
        self.fc = make_fc(dims)

    def forward(self, x):
        logits = self.fc(x)
        return logits


class SCANVI(nn.Module):

    def __init__(self, num_genes, num_labels, l_loc, l_scale, latent_dim=10, alpha=0.01, scale_factor=1.0):
        assert isinstance(num_genes, int)
        self.num_genes = num_genes
        assert isinstance(num_labels, int) and num_labels > 1
        self.num_labels = num_labels
        assert isinstance(latent_dim, int) and latent_dim > 0
        self.latent_dim = latent_dim
        assert isinstance(l_loc, float)
        self.l_loc = l_loc
        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale
        assert isinstance(alpha, float) and alpha > 0
        self.alpha = alpha
        assert isinstance(scale_factor, float) and scale_factor > 0
        self.scale_factor = scale_factor
        super().__init__()
        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels, z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim, z2_dim=self.latent_dim, hidden_dims=[50])
        self.epsilon = 0.005

    def model(self, x, y=None):
        pyro.module('scanvi', self)
        theta = pyro.param('inverse_dispersion', 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)
        with pyro.plate('batch', len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample('z1', dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample('y', dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y)
            z2_loc, z2_scale = self.z2_decoder(z1, y)
            z2 = pyro.sample('z2', dist.Normal(z2_loc, z2_scale).to_event(1))
            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample('l', dist.LogNormal(self.l_loc, l_scale).to_event(1))
            gate_logits, mu = self.x_decoder(z2)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits)
            pyro.sample('x', x_dist.to_event(1), obs=x)

    def guide(self, x, y=None):
        pyro.module('scanvi', self)
        with pyro.plate('batch', len(x)), poutine.scale(scale=self.scale_factor):
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            pyro.sample('l', dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample('z2', dist.Normal(z2_loc, z2_scale).to_event(1))
            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                y = pyro.sample('y', y_dist)
            else:
                classification_loss = y_dist.log_prob(y)
                pyro.factor('classification_loss', -self.alpha * classification_loss, has_rsample=False)
            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample('z1', dist.Normal(z1_loc, z1_scale).to_event(1))


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)


class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset

    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,), config_enum=None, use_cuda=False, aux_loss_multiplier=None):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        self.encoder_y = MLP([self.input_size] + hidden_sizes + [self.output_size], activation=nn.Softplus, output_activation=nn.Softmax, allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        self.encoder_z = MLP([self.input_size + self.output_size] + hidden_sizes + [[z_dim, z_dim]], activation=nn.Softplus, output_activation=[None, Exp], allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        self.decoder = MLP([z_dim + self.output_size] + hidden_sizes + [self.input_size], activation=nn.Softplus, output_activation=nn.Sigmoid, allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        if self.use_cuda:
            self

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # an image
        loc is given by a neural network  `decoder`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        pyro.module('ss_vae', self)
        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate('data'):
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (1.0 * self.output_size)
            ys = pyro.sample('y', dist.OneHotCategorical(alpha_prior), obs=ys)
            loc = self.decoder([zs, ys])
            pyro.sample('x', dist.Bernoulli(loc, validate_args=False).to_event(1), obs=xs)
            return loc

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        with pyro.plate('data'):
            if ys is None:
                alpha = self.encoder_y(xs)
                ys = pyro.sample('y', dist.OneHotCategorical(alpha))
            loc, scale = self.encoder_z([xs, ys])
            pyro.sample('z', dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        alpha = self.encoder_y(xs)
        res, ind = torch.topk(alpha, 1)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        pyro.module('ss_vae', self)
        with pyro.plate('data'):
            if ys is not None:
                alpha = self.encoder_y(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample('y_aux', dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


TEST = 'test'


TRAIN = 'train'


class VAE(object, metaclass=ABCMeta):
    """
    Abstract class for the variational auto-encoder. The abstract method
    for training the network is implemented by subclasses.
    """

    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.vae_encoder = Encoder()
        self.vae_decoder = Decoder()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.mode = TRAIN

    def set_train(self, is_train=True):
        if is_train:
            self.mode = TRAIN
            self.vae_encoder.train()
            self.vae_decoder.train()
        else:
            self.mode = TEST
            self.vae_encoder.eval()
            self.vae_decoder.eval()

    @abstractmethod
    def compute_loss_and_gradient(self, x):
        """
        Given a batch of data `x`, run the optimizer (backpropagate the gradient),
        and return the computed loss.

        :param x: batch of data or a single datum (MNIST image).
        :return: loss computed on the data batch.
        """
        return

    def model_eval(self, x):
        """
        Given a batch of data `x`, run it through the trained VAE network to get
        the reconstructed image.

        :param x: batch of data or a single datum (MNIST image).
        :return: reconstructed image, and the latent z's mean and variance.
        """
        z_mean, z_var = self.vae_encoder(x)
        if self.mode == TRAIN:
            z = Normal(z_mean, z_var.sqrt()).rsample()
        else:
            z = z_mean
        return self.vae_decoder(z), z_mean, z_var

    def train(self, epoch):
        self.set_train(is_train=True)
        train_loss = 0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            loss = self.compute_loss_and_gradient(x)
            train_loss += loss
        None

    def test(self, epoch):
        self.set_train(is_train=False)
        test_loss = 0
        for i, (x, _) in enumerate(self.test_loader):
            with torch.no_grad():
                recon_x = self.model_eval(x)[0]
                test_loss += self.compute_loss_and_gradient(x)
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], recon_x.reshape(self.args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.detach().cpu(), os.path.join(OUTPUT_DIR, 'reconstruction_' + str(epoch) + '.png'), nrow=n)
        test_loss /= len(self.test_loader.dataset)
        None


class PyTorchVAEImpl(VAE):
    """
    Adapted from pytorch/examples.
    Source: https://github.com/pytorch/examples/tree/master/vae
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = self.initialize_optimizer(lr=0.001)

    def compute_loss_and_gradient(self, x):
        self.optimizer.zero_grad()
        recon_x, z_mean, z_var = self.model_eval(x)
        binary_cross_entropy = functional.binary_cross_entropy(recon_x, x.reshape(-1, 784))
        kl_div = -0.5 * torch.sum(1 + z_var.log() - z_mean.pow(2) - z_var)
        kl_div /= self.args.batch_size * 784
        loss = binary_cross_entropy + kl_div
        if self.mode == TRAIN:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def initialize_optimizer(self, lr=0.001):
        model_params = itertools.chain(self.vae_encoder.parameters(), self.vae_decoder.parameters())
        return torch.optim.Adam(model_params, lr)


PYRO_STACK = []


class Messenger:

    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


def _block_fn(expose: 'List[str]', expose_types: 'List[str]', hide: 'List[str]', hide_types: 'List[str]', hide_all: 'bool', msg: "'Message'") ->bool:
    if msg['type'] == 'sample' and msg['is_observed']:
        msg_type = 'observe'
    else:
        msg_type = msg['type']
    is_not_exposed = msg['name'] not in expose and msg_type not in expose_types
    if msg['name'] in hide or msg_type in hide_types or is_not_exposed and hide_all:
        return True
    else:
        return False


def set(**kwargs) ->None:
    """
    Sets one or more settings.

    :param \\*\\*kwargs: alias=value pairs.
    """
    for alias, value in kwargs.items():
        module, deepname, validator = _REGISTRY[alias]
        if validator is not None:
            validator(value)
        destin = import_module(module)
        names = deepname.split('.')
        for name in names[:-1]:
            destin = getattr(destin, name)
        setattr(destin, names[-1], value)


def _make_default_hide_fn(hide_all: 'bool', expose_all: 'bool', hide: 'Optional[List[str]]', expose: 'Optional[List[str]]', hide_types: 'Optional[List[str]]', expose_types: 'Optional[List[str]]') ->Callable[['Message'], bool]:
    assert hide_all is False and expose_all is False or hide_all != expose_all, 'cannot hide and expose a site'
    if hide is None:
        hide = []
    else:
        hide_all = False
    if expose is None:
        expose = []
    else:
        hide_all = True
    assert set(hide).isdisjoint(set(expose)), 'cannot hide and expose a site'
    if hide_types is None:
        hide_types = []
    else:
        hide_all = False
    if expose_types is None:
        expose_types = []
    else:
        hide_all = True
    assert set(hide_types).isdisjoint(set(expose_types)), 'cannot hide and expose a site type'
    return partial(_block_fn, expose, expose_types, hide, hide_types, hide_all)


def _negate_fn(fn: "Callable[['Message'], Optional[bool]]") ->Callable[['Message'], bool]:

    def negated_fn(msg: "'Message'") ->bool:
        return not fn(msg)
    return negated_fn


class BlockMessenger(Messenger):
    """
    This handler selectively hides Pyro primitive sites from the outside world.
    Default behavior: block everything.

    A site is hidden if at least one of the following holds:

        0. ``hide_fn(msg) is True`` or ``(not expose_fn(msg)) is True``
        1. ``msg["name"] in hide``
        2. ``msg["type"] in hide_types``
        3. ``msg["name"] not in expose and msg["type"] not in expose_types``
        4. ``hide``, ``hide_types``, and ``expose_types`` are all ``None``

    For example, suppose the stochastic function fn has two sample sites "a" and "b".
    Then any effect outside of ``BlockMessenger(fn, hide=["a"])``
    will not be applied to site "a" and will only see site "b":

        >>> def fn():
        ...     a = pyro.sample("a", dist.Normal(0., 1.))
        ...     return pyro.sample("b", dist.Normal(a, 1.))
        >>> fn_inner = pyro.poutine.trace(fn)
        >>> fn_outer = pyro.poutine.trace(pyro.poutine.block(fn_inner, hide=["a"]))
        >>> trace_inner = fn_inner.get_trace()
        >>> trace_outer  = fn_outer.get_trace()
        >>> "a" in trace_inner
        True
        >>> "a" in trace_outer
        False
        >>> "b" in trace_inner
        True
        >>> "b" in trace_outer
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param hide_fn: function that takes a site and returns True to hide the site
        or False/None to expose it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param expose_fn: function that takes a site and returns True to expose the site
        or False/None to hide it.  If specified, all other parameters are ignored.
        Only specify one of hide_fn or expose_fn, not both.
    :param bool hide_all: hide all sites
    :param bool expose_all: expose all sites normally
    :param list hide: list of site names to hide
    :param list expose: list of site names to be exposed while all others hidden
    :param list hide_types: list of site types to be hidden
    :param list expose_types: list of site types to be exposed while all others hidden
    :returns: stochastic function decorated with a :class:`~pyro.poutine.block_messenger.BlockMessenger`
    """

    def __init__(self, hide_fn: "Optional[Callable[['Message'], Optional[bool]]]"=None, expose_fn: "Optional[Callable[['Message'], Optional[bool]]]"=None, hide_all: 'bool'=True, expose_all: 'bool'=False, hide: 'Optional[List[str]]'=None, expose: 'Optional[List[str]]'=None, hide_types: 'Optional[List[str]]'=None, expose_types: 'Optional[List[str]]'=None) ->None:
        super().__init__()
        if not (hide_fn is None or expose_fn is None):
            raise ValueError('Only specify one of hide_fn or expose_fn')
        if hide_fn is not None:
            self.hide_fn = hide_fn
        elif expose_fn is not None:
            self.hide_fn = _negate_fn(expose_fn)
        else:
            self.hide_fn = _make_default_hide_fn(hide_all, expose_all, hide, expose, hide_types, expose_types)

    def _process_message(self, msg: "'Message'") ->None:
        msg['stop'] = bool(self.hide_fn(msg))


_T = TypeVar('_T')


def _make_handler(msngr_cls, module=None):

    def handler_decorator(func):

        @functools.wraps(func)
        def handler(fn=None, *args, **kwargs):
            if fn is not None and not (callable(fn) or isinstance(fn, collections.abc.Iterable)):
                raise ValueError(f'{fn} is not callable, did you mean to pass it as a keyword arg?')
            msngr = msngr_cls(*args, **kwargs)
            return functools.update_wrapper(msngr(fn), fn, updated=()) if fn is not None else msngr
        handler.__doc__ = 'Convenient wrapper of :class:`~pyro.poutine.{}.{}` \n\n'.format(func.__name__ + '_messenger', msngr_cls.__name__) + (msngr_cls.__doc__ if msngr_cls.__doc__ else '')
        if module is not None:
            handler.__module__ = module
        return handler
    return handler_decorator


class ReplayMessenger(Messenger):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    ``replay`` makes ``sample`` statements behave as if they had sampled the values
    at the corresponding sites in the trace:

        >>> old_trace = pyro.poutine.trace(model).get_trace(1.0)
        >>> replayed_model = pyro.poutine.replay(model, trace=old_trace)
        >>> bool(replayed_model(0.0) == old_trace.nodes["_RETURN"]["value"])
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param trace: a :class:`~pyro.poutine.Trace` data structure to replay against
    :param params: dict of names of param sites and constrained values
        in fn to replay against
    :returns: a stochastic function decorated with a :class:`~pyro.poutine.replay_messenger.ReplayMessenger`
    """

    def __init__(self, trace: "Optional['Trace']"=None, params: "Optional[Dict[str, 'torch.Tensor']]"=None) ->None:
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """
        super().__init__()
        if trace is None and params is None:
            raise ValueError('must provide trace or params to replay against')
        self.trace = trace
        self.params = params

    def _pyro_sample(self, msg: "'Message'") ->None:
        """
        :param msg: current message at a trace site.

        At a sample site that appears in self.trace,
        returns the value from self.trace instead of sampling
        from the stochastic function at the site.

        At a sample site that does not appear in self.trace,
        reverts to default Messenger._pyro_sample behavior with no additional side effects.
        """
        assert msg['name'] is not None
        name = msg['name']
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if msg['is_observed']:
                return None
            if guide_msg['type'] != 'sample' or guide_msg['is_observed']:
                raise RuntimeError('site {} must be sampled in trace'.format(name))
            msg['done'] = True
            msg['value'] = guide_msg['value']
            msg['infer'] = guide_msg['infer']

    def _pyro_param(self, msg: "'Message'") ->None:
        name = msg['name']
        if self.params is not None and name in self.params:
            assert hasattr(self.params[name], 'unconstrained'), 'param {} must be constrained value'.format(name)
            msg['done'] = True
            msg['value'] = self.params[name]


class ScoreParts(NamedTuple):
    """
    This data structure stores terms used in stochastic gradient estimators that
    combine the pathwise estimator and the score function estimator.
    """
    log_prob: 'torch.Tensor'
    score_function: 'torch.Tensor'
    entropy_term: 'torch.Tensor'

    def scale_and_mask(self, scale: 'Union[float, torch.Tensor]'=1.0, mask: 'Optional[torch.BoolTensor]'=None) ->'ScoreParts':
        """
        Scale and mask appropriate terms of a gradient estimator by a data multiplicity factor.
        Note that the `score_function` term should not be scaled or masked.

        :param scale: a positive scale
        :type scale: torch.Tensor or number
        :param mask: an optional masking tensor
        :type mask: torch.BoolTensor or None
        """
        log_prob = scale_and_mask(self.log_prob, scale, mask)
        score_function = self.score_function
        entropy_term = scale_and_mask(self.entropy_term, scale, mask)
        return ScoreParts(log_prob, score_function, entropy_term)


def _format_table(rows: 'List[List[Optional[str]]]') ->str:
    """
    Formats a right justified table using None as column separator.
    """
    column_widths = [0, 0, 0]
    for row in rows:
        widths = [0, 0, 0]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                widths[j] += 1
        for j in range(3):
            column_widths[j] = max(column_widths[j], widths[j])
    justified_rows: 'List[List[str]]' = []
    for row in rows:
        cols: 'List[List[str]]' = [[], [], []]
        j = 0
        for cell in row:
            if cell is None:
                j += 1
            else:
                cols[j].append(cell)
        cols = [([''] * (width - len(col)) + col if direction == 'r' else col + [''] * (width - len(col))) for width, col, direction in zip(column_widths, cols, 'rrl')]
        justified_rows.append(sum(cols, []))
    cell_widths = [0] * len(justified_rows[0])
    for justified_row in justified_rows:
        for j, cell in enumerate(justified_row):
            cell_widths[j] = max(cell_widths[j], len(cell))
    return '\n'.join(' '.join(cell.rjust(width) for cell, width in zip(justified_row, cell_widths)) for justified_row in justified_rows)


def allow_all_sites(name: 'str', site: "'Message'") ->bool:
    return True


_VALIDATION_ENABLED = __debug__


def is_validation_enabled():
    return _VALIDATION_ENABLED


def pack(value, dim_to_symbol):
    """
    Converts an unpacked tensor to a packed tensor.

    :param value: a number or tensor
    :param dim_to_symbol: a map from negative integers to characters
    """
    if isinstance(value, torch.Tensor):
        assert not hasattr(value, '_pyro_dims'), 'tried to pack an already-packed tensor'
        shape = value.shape
        shift = len(shape)
        try:
            with ignore_jit_warnings():
                dims = ''.join(dim_to_symbol[dim - shift] for dim, size in enumerate(shape) if size > 1)
        except KeyError as e:
            raise ValueError('\n  '.join(['Invalid tensor shape.', 'Allowed dims: {}'.format(', '.join(map(str, sorted(dim_to_symbol)))), 'Actual shape: {}'.format(tuple(value.shape)), "Try adding shape assertions for your model's sample values and distribution parameters."])) from e
        value = value.squeeze()
        value._pyro_dims = dims
        assert value.dim() == len(value._pyro_dims)
    return value


def is_identically_one(x):
    """
    Check if argument is exactly the number one. True for the number one;
    false for other numbers; false for :class:`~torch.Tensor`s.
    """
    if isinstance(x, numbers.Number):
        return x == 1
    if not torch._C._get_tracing_state():
        if isinstance(x, torch.Tensor) and x.dtype == torch.int64 and not x.shape:
            return x.item() == 1
    return False


def scale_and_mask(tensor, scale=1.0, mask=None):
    """
    Scale and mask a packed tensor, broadcasting and avoiding unnecessary ops.

    :param torch.Tensor tensor: a packed tensor
    :param scale: a positive scale
    :type scale: torch.Tensor or number
    :param mask: an optional packed tensor mask
    :type mask: torch.BoolTensor, bool, or None
    """
    if isinstance(scale, torch.Tensor) and scale.dim():
        raise NotImplementedError('non-scalar scale is not supported')
    if mask is None or mask is True:
        if is_identically_one(scale):
            return tensor
        result = tensor * scale
        result._pyro_dims = tensor._pyro_dims
        return result
    if mask is False:
        result = torch.zeros_like(tensor)
        result._pyro_dims = tensor._pyro_dims
        return result
    tensor, mask = broadcast_all(tensor, mask)
    result = torch.where(mask, tensor, tensor.new_zeros(()))
    result._pyro_dims = tensor._pyro_dims
    return result


def warn_if_inf(value: 'Union[torch.Tensor, numbers.Number]', msg: 'str'='', allow_posinf: 'bool'=False, allow_neginf: 'bool'=False, *, filename: Optional[str]=None, lineno: Optional[int]=None) ->Union[torch.Tensor, numbers.Number]:
    """
    A convenient function to warn if a Tensor or its grad contains any inf,
    also works with numbers.
    """
    if filename is None:
        try:
            frame = sys._getframe(1)
        except ValueError:
            filename = 'sys'
            lineno = 1
        else:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
    if isinstance(value, torch.Tensor) and value.requires_grad:
        value.register_hook(lambda x: warn_if_inf(x, 'backward ' + msg, allow_posinf, allow_neginf, filename=filename, lineno=lineno))
    if not allow_posinf and (value == math.inf if isinstance(value, numbers.Number) else (value == math.inf).any()):
        assert isinstance(lineno, int)
        warnings.warn_explicit('Encountered +inf{}'.format(': ' + msg if msg else '.'), UserWarning, filename, lineno)
    if not allow_neginf and (value == -math.inf if isinstance(value, numbers.Number) else (value == -math.inf).any()):
        assert isinstance(lineno, int)
        warnings.warn_explicit('Encountered -inf{}'.format(': ' + msg if msg else '.'), UserWarning, filename, lineno)
    return value


def torch_isnan(x: 'Union[torch.Tensor, numbers.Number]') ->Union[bool, torch.Tensor]:
    """
    A convenient function to check if a Tensor contains any nan; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x != x
    return torch.isnan(x).any()


def warn_if_nan(value: 'Union[torch.Tensor, numbers.Number]', msg: 'str'='', *, filename: Optional[str]=None, lineno: Optional[int]=None) ->Union[torch.Tensor, numbers.Number]:
    """
    A convenient function to warn if a Tensor or its grad contains any nan,
    also works with numbers.
    """
    if filename is None:
        try:
            frame = sys._getframe(1)
        except ValueError:
            filename = 'sys'
            lineno = 1
        else:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
    if isinstance(value, torch.Tensor) and value.requires_grad:
        value.register_hook(lambda x: warn_if_nan(x, 'backward ' + msg, filename=filename, lineno=lineno))
    if torch_isnan(value):
        assert isinstance(lineno, int)
        warnings.warn_explicit('Encountered NaN{}'.format(': ' + msg if msg else '.'), UserWarning, filename, lineno)
    return value


class Trace:
    """
    Graph data structure denoting the relationships amongst different pyro primitives
    in the execution trace.

    An execution trace of a Pyro program is a record of every call
    to ``pyro.sample()`` and ``pyro.param()`` in a single execution of that program.
    Traces are directed graphs whose nodes represent primitive calls or input/output,
    and whose edges represent conditional dependence relationships
    between those primitive calls. They are created and populated by ``poutine.trace``.

    Each node (or site) in a trace contains the name, input and output value of the site,
    as well as additional metadata added by inference algorithms or user annotation.
    In the case of ``pyro.sample``, the trace also includes the stochastic function
    at the site, and any observed data added by users.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    We can record its execution using ``pyro.poutine.trace``
    and use the resulting data structure to compute the log-joint probability
    of all of the sample sites in the execution or extract all parameters.

        >>> trace = pyro.poutine.trace(model).get_trace(0.0)
        >>> logp = trace.log_prob_sum()
        >>> params = [trace.nodes[name]["value"].unconstrained() for name in trace.param_nodes]

    We can also inspect or manipulate individual nodes in the trace.
    ``trace.nodes`` contains a ``collections.OrderedDict``
    of site names and metadata corresponding to ``x``, ``s``, ``z``, and the return value:

        >>> list(name for name in trace.nodes.keys())  # doctest: +SKIP
        ["_INPUT", "s", "z", "_RETURN"]

    Values of ``trace.nodes`` are dictionaries of node metadata:

        >>> trace.nodes["z"]  # doctest: +SKIP
        {'type': 'sample', 'name': 'z', 'is_observed': False,
         'fn': Normal(), 'value': tensor(0.6480), 'args': (), 'kwargs': {},
         'infer': {}, 'scale': 1.0, 'cond_indep_stack': (),
         'done': True, 'stop': False, 'continuation': None}

    ``'infer'`` is a dictionary of user- or algorithm-specified metadata.
    ``'args'`` and ``'kwargs'`` are the arguments passed via ``pyro.sample``
    to ``fn.__call__`` or ``fn.log_prob``.
    ``'scale'`` is used to scale the log-probability of the site when computing the log-joint.
    ``'cond_indep_stack'`` contains data structures corresponding to ``pyro.plate`` contexts
    appearing in the execution.
    ``'done'``, ``'stop'``, and ``'continuation'`` are only used by Pyro's internals.

    :param string graph_type: string specifying the kind of trace graph to construct
    """

    def __init__(self, graph_type: "Literal['flat', 'dense']"='flat') ->None:
        assert graph_type in ('flat', 'dense'), '{} not a valid graph type'.format(graph_type)
        self.graph_type = graph_type
        self.nodes: "OrderedDict[str, 'Message']" = OrderedDict()
        self._succ: 'OrderedDict[str, Set[str]]' = OrderedDict()
        self._pred: 'OrderedDict[str, Set[str]]' = OrderedDict()

    def __contains__(self, name: 'str') ->bool:
        return name in self.nodes

    def __iter__(self) ->Iterable[str]:
        return iter(self.nodes.keys())

    def __len__(self) ->int:
        return len(self.nodes)

    @property
    def edges(self) ->Iterable[Tuple[str, str]]:
        for site, adj_nodes in self._succ.items():
            for adj_node in adj_nodes:
                yield site, adj_node

    def add_node(self, site_name: 'str', **kwargs: Any) ->None:
        """
        :param string site_name: the name of the site to be added

        Adds a site to the trace.

        Raises an error when attempting to add a duplicate node
        instead of silently overwriting.
        """
        if site_name in self:
            site = self.nodes[site_name]
            if site['type'] != kwargs['type']:
                raise RuntimeError('{} is already in the trace as a {}'.format(site_name, site['type']))
            elif kwargs['type'] != 'param':
                raise RuntimeError("Multiple {} sites named '{}'".format(kwargs['type'], site_name))
        self.nodes[site_name] = kwargs
        self._pred[site_name] = set()
        self._succ[site_name] = set()

    def add_edge(self, site1: 'str', site2: 'str') ->None:
        for site in (site1, site2):
            if site not in self.nodes:
                self.add_node(site)
        self._succ[site1].add(site2)
        self._pred[site2].add(site1)

    def remove_node(self, site_name: 'str') ->None:
        self.nodes.pop(site_name)
        for p in self._pred[site_name]:
            self._succ[p].remove(site_name)
        for s in self._succ[site_name]:
            self._pred[s].remove(site_name)
        self._pred.pop(site_name)
        self._succ.pop(site_name)

    def predecessors(self, site_name: 'str') ->Set[str]:
        return self._pred[site_name]

    def successors(self, site_name: 'str') ->Set[str]:
        return self._succ[site_name]

    def copy(self) ->'Trace':
        """
        Makes a shallow copy of self with nodes and edges preserved.
        """
        new_tr = Trace(graph_type=self.graph_type)
        new_tr.nodes.update(self.nodes)
        new_tr._succ.update(self._succ)
        new_tr._pred.update(self._pred)
        return new_tr

    def _dfs(self, site: 'str', visited: 'Set[str]') ->Iterable[str]:
        if site in visited:
            return
        for s in self._succ[site]:
            for node in self._dfs(s, visited):
                yield node
        visited.add(site)
        yield site

    def topological_sort(self, reverse: 'bool'=False) ->List[str]:
        """
        Return a list of nodes (site names) in topologically sorted order.

        :param bool reverse: Return the list in reverse order.
        :return: list of topologically sorted nodes (site names).
        """
        visited: 'Set[str]' = set()
        top_sorted = []
        for s in self._succ:
            for node in self._dfs(s, visited):
                top_sorted.append(node)
        return top_sorted if reverse else list(reversed(top_sorted))

    def log_prob_sum(self, site_filter: "Callable[[str, 'Message'], bool]"=allow_all_sites) ->Union['torch.Tensor', float]:
        """
        Compute the site-wise log probabilities of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        The computation of ``log_prob_sum`` is memoized.

        :returns: total log probability.
        :rtype: torch.Tensor
        """
        result = 0.0
        for name, site in self.nodes.items():
            if site['type'] == 'sample' and site_filter(name, site):
                if TYPE_CHECKING:
                    assert isinstance(site['fn'], Distribution)
                if 'log_prob_sum' in site:
                    log_p = site['log_prob_sum']
                else:
                    try:
                        log_p = site['fn'].log_prob(site['value'], *site['args'], **site['kwargs'])
                    except ValueError as e:
                        _, exc_value, traceback = sys.exc_info()
                        shapes = self.format_shapes(last_site=site['name'])
                        raise ValueError("Error while computing log_prob_sum at site '{}':\n{}\n{}\n".format(name, exc_value, shapes)).with_traceback(traceback) from e
                    log_p = scale_and_mask(log_p, site['scale'], site['mask']).sum()
                    site['log_prob_sum'] = log_p
                    if is_validation_enabled():
                        warn_if_nan(log_p, "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(log_p, "log_prob_sum at site '{}'".format(name), allow_neginf=True)
                result = result + log_p
        return result

    def compute_log_prob(self, site_filter: "Callable[[str, 'Message'], bool]"=allow_all_sites) ->None:
        """
        Compute the site-wise log probabilities of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        Both computations are memoized.
        """
        for name, site in self.nodes.items():
            if site['type'] == 'sample' and site_filter(name, site):
                if TYPE_CHECKING:
                    assert isinstance(site['fn'], Distribution)
                if 'log_prob' not in site:
                    try:
                        log_p = site['fn'].log_prob(site['value'], *site['args'], **site['kwargs'])
                    except ValueError as e:
                        _, exc_value, traceback = sys.exc_info()
                        shapes = self.format_shapes(last_site=site['name'])
                        raise ValueError("Error while computing log_prob at site '{}':\n{}\n{}".format(name, exc_value, shapes)).with_traceback(traceback) from e
                    site['unscaled_log_prob'] = log_p
                    log_p = scale_and_mask(log_p, site['scale'], site['mask'])
                    site['log_prob'] = log_p
                    site['log_prob_sum'] = log_p.sum()
                    if is_validation_enabled():
                        warn_if_nan(site['log_prob_sum'], "log_prob_sum at site '{}'".format(name))
                        warn_if_inf(site['log_prob_sum'], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

    def compute_score_parts(self) ->None:
        """
        Compute the batched local score parts at each site of the trace.
        Each ``log_prob`` has shape equal to the corresponding ``batch_shape``.
        Each ``log_prob_sum`` is a scalar.
        All computations are memoized.
        """
        for name, site in self.nodes.items():
            if site['type'] == 'sample' and 'score_parts' not in site:
                if TYPE_CHECKING:
                    assert isinstance(site['fn'], Distribution)
                try:
                    value = site['fn'].score_parts(site['value'], *site['args'], **site['kwargs'])
                except ValueError as e:
                    _, exc_value, traceback = sys.exc_info()
                    shapes = self.format_shapes(last_site=site['name'])
                    raise ValueError("Error while computing score_parts at site '{}':\n{}\n{}".format(name, exc_value, shapes)).with_traceback(traceback) from e
                site['unscaled_log_prob'] = value.log_prob
                value = value.scale_and_mask(site['scale'], site['mask'])
                site['score_parts'] = value
                site['log_prob'] = value.log_prob
                site['log_prob_sum'] = value.log_prob.sum()
                if is_validation_enabled():
                    warn_if_nan(site['log_prob_sum'], "log_prob_sum at site '{}'".format(name))
                    warn_if_inf(site['log_prob_sum'], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

    def detach_(self) ->None:
        """
        Detach values (in-place) at each sample site of the trace.
        """
        for _, site in self.nodes.items():
            if site['type'] == 'sample':
                assert site['value'] is not None
                site['value'] = site['value'].detach()

    @property
    def observation_nodes(self) ->List[str]:
        """
        :return: a list of names of observe sites
        """
        return [name for name, node in self.nodes.items() if node['type'] == 'sample' and node['is_observed']]

    @property
    def param_nodes(self) ->List[str]:
        """
        :return: a list of names of param sites
        """
        return [name for name, node in self.nodes.items() if node['type'] == 'param']

    @property
    def stochastic_nodes(self) ->List[str]:
        """
        :return: a list of names of sample sites
        """
        return [name for name, node in self.nodes.items() if node['type'] == 'sample' and not node['is_observed']]

    @property
    def reparameterized_nodes(self) ->List[str]:
        """
        :return: a list of names of sample sites whose stochastic functions
            are reparameterizable primitive distributions
        """
        return [name for name, node in self.nodes.items() if node['type'] == 'sample' and not node['is_observed'] and getattr(node['fn'], 'has_rsample', False)]

    @property
    def nonreparam_stochastic_nodes(self) ->List[str]:
        """
        :return: a list of names of sample sites whose stochastic functions
            are not reparameterizable primitive distributions
        """
        return list(set(self.stochastic_nodes) - set(self.reparameterized_nodes))

    def iter_stochastic_nodes(self) ->Iterator[Tuple[str, 'Message']]:
        """
        :return: an iterator over stochastic nodes in the trace.
        """
        for name, node in self.nodes.items():
            if node['type'] == 'sample' and not node['is_observed']:
                yield name, node

    def symbolize_dims(self, plate_to_symbol: 'Optional[Dict[str, str]]'=None) ->None:
        """
        Assign unique symbols to all tensor dimensions.
        """
        plate_to_symbol = {} if plate_to_symbol is None else plate_to_symbol
        symbol_to_dim = {}
        for site in self.nodes.values():
            if site['type'] != 'sample':
                continue
            dim_to_symbol: 'Dict[int, str]' = {}
            for frame in site['cond_indep_stack']:
                if frame.vectorized:
                    assert frame.dim is not None
                    if frame.name in plate_to_symbol:
                        symbol = plate_to_symbol[frame.name]
                    else:
                        symbol = opt_einsum.get_symbol(2 * len(plate_to_symbol))
                        plate_to_symbol[frame.name] = symbol
                    symbol_to_dim[symbol] = frame.dim
                    dim_to_symbol[frame.dim] = symbol
            assert site['infer'] is not None
            for dim, id_ in site['infer'].get('_dim_to_id', {}).items():
                symbol = opt_einsum.get_symbol(1 + 2 * id_)
                symbol_to_dim[symbol] = dim
                dim_to_symbol[dim] = symbol
            enum_dim = site['infer'].get('_enumerate_dim')
            if enum_dim is not None:
                site['infer']['_enumerate_symbol'] = dim_to_symbol[enum_dim]
            site['infer']['_dim_to_symbol'] = dim_to_symbol
        self.plate_to_symbol = plate_to_symbol
        self.symbol_to_dim = symbol_to_dim

    def pack_tensors(self, plate_to_symbol: 'Optional[Dict[str, str]]'=None) ->None:
        """
        Computes packed representations of tensors in the trace.
        This should be called after :meth:`compute_log_prob` or :meth:`compute_score_parts`.
        """
        self.symbolize_dims(plate_to_symbol)
        for site in self.nodes.values():
            if site['type'] != 'sample':
                continue
            assert site['infer'] is not None
            dim_to_symbol = site['infer']['_dim_to_symbol']
            packed = site.setdefault('packed', {})
            try:
                packed['mask'] = pack(site['mask'], dim_to_symbol)
                if 'score_parts' in site:
                    log_prob, score_function, entropy_term = site['score_parts']
                    log_prob = pack(log_prob, dim_to_symbol)
                    score_function = pack(score_function, dim_to_symbol)
                    entropy_term = pack(entropy_term, dim_to_symbol)
                    packed['score_parts'] = ScoreParts(log_prob, score_function, entropy_term)
                    packed['log_prob'] = log_prob
                    packed['unscaled_log_prob'] = pack(site['unscaled_log_prob'], dim_to_symbol)
                elif 'log_prob' in site:
                    packed['log_prob'] = pack(site['log_prob'], dim_to_symbol)
                    packed['unscaled_log_prob'] = pack(site['unscaled_log_prob'], dim_to_symbol)
            except ValueError as e:
                _, exc_value, traceback = sys.exc_info()
                shapes = self.format_shapes(last_site=site['name'])
                raise ValueError("Error while packing tensors at site '{}':\n  {}\n{}".format(site['name'], exc_value, shapes)).with_traceback(traceback) from e

    def format_shapes(self, title: 'str'='Trace Shapes:', last_site: 'Optional[str]'=None) ->str:
        """
        Returns a string showing a table of the shapes of all sites in the
        trace.
        """
        if not self.nodes:
            return title
        rows: 'List[List[Optional[str]]]' = [[title]]
        rows.append(['Param Sites:'])
        for name, site in self.nodes.items():
            if site['type'] == 'param':
                if TYPE_CHECKING:
                    assert isinstance(site['value'], torch.Tensor)
                rows.append([name, None] + [str(size) for size in site['value'].shape])
            if name == last_site:
                break
        rows.append(['Sample Sites:'])
        for name, site in self.nodes.items():
            if site['type'] == 'sample':
                batch_shape = getattr(site['fn'], 'batch_shape', ())
                event_shape = getattr(site['fn'], 'event_shape', ())
                rows.append([name + ' dist', None] + [str(size) for size in batch_shape] + ['|', None] + [str(size) for size in event_shape])
                event_dim = len(event_shape)
                shape = getattr(site['value'], 'shape', ())
                batch_shape = shape[:len(shape) - event_dim]
                event_shape = shape[len(shape) - event_dim:]
                rows.append(['value', None] + [str(size) for size in batch_shape] + ['|', None] + [str(size) for size in event_shape])
                if 'log_prob' in site:
                    batch_shape = getattr(site['log_prob'], 'shape', ())
                    rows.append(['log_prob', None] + [str(size) for size in batch_shape] + ['|', None])
            if name == last_site:
                break
        return _format_table(rows)


def site_is_subsample(site: "'Message'") ->bool:
    """
    Determines whether a trace site originated from a subsample statement inside an `plate`.
    """
    return site['type'] == 'sample' and type(site['fn']).__name__ == '_Subsample'


def identify_dense_edges(trace: 'Trace') ->None:
    """
    Modifies a trace in-place by adding all edges based on the
    `cond_indep_stack` information stored at each site.
    """
    for name, node in trace.nodes.items():
        if site_is_subsample(node):
            continue
        if node['type'] == 'sample':
            for past_name, past_node in trace.nodes.items():
                if site_is_subsample(past_node):
                    continue
                if past_node['type'] == 'sample':
                    if past_name == name:
                        break
                    past_node_independent = False
                    for query, target in zip(node['cond_indep_stack'], past_node['cond_indep_stack']):
                        if query.name == target.name and query.counter != target.counter:
                            past_node_independent = True
                            break
                    if not past_node_independent:
                        trace.add_edge(past_name, name)


def elbo(model, guide, *args, **kwargs):
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    elbo = 0.0
    for site in model_trace.values():
        if site['type'] == 'sample':
            elbo = elbo + site['fn'].log_prob(site['value']).sum()
    for site in guide_trace.values():
        if site['type'] == 'sample':
            elbo = elbo - site['fn'].log_prob(site['value']).sum()
    return -elbo


def am_i_wrapped() ->bool:
    """
    Checks whether the current computation is wrapped in a poutine.
    :returns: bool
    """
    return len(_PYRO_STACK) > 0


def default_process_message(msg: 'Message') ->None:
    """
    Default method for processing messages in inference.

    :param msg: a message to be processed
    :returns: None
    """
    if msg['done'] or msg['is_observed'] or msg['value'] is not None:
        msg['done'] = True
        return
    msg['value'] = msg['fn'](*msg['args'], **msg['kwargs'])
    msg['done'] = True


def apply_stack(initial_msg: 'Message') ->None:
    """
    Execute the effect stack at a single site according to the following scheme:

        1. For each ``Messenger`` in the stack from bottom to top,
           execute ``Messenger._process_message`` with the message;
           if the message field "stop" is True, stop;
           otherwise, continue
        2. Apply default behavior (``default_process_message``) to finish remaining site execution
        3. For each ``Messenger`` in the stack from top to bottom,
           execute ``_postprocess_message`` to update the message and internal messenger state with the site results
        4. If the message field "continuation" is not ``None``, call it with the message

    :param dict initial_msg: the starting version of the trace site
    :returns: ``None``
    """
    stack = _PYRO_STACK
    msg = initial_msg
    pointer = 0
    for frame in reversed(stack):
        pointer = pointer + 1
        frame._process_message(msg)
        if msg['stop']:
            break
    default_process_message(msg)
    for frame in stack[-pointer:]:
        frame._postprocess_message(msg)
    cont = msg['continuation']
    if cont is not None:
        cont(msg)


def effectful(fn: 'Optional[Callable[_P, _T]]'=None, type: 'Optional[str]'=None) ->Callable:
    """
    :param fn: function or callable that performs an effectful computation
    :param str type: the type label of the operation, e.g. `"sample"`

    Wrapper for calling :func:`~pyro.poutine.runtime.apply_stack` to apply any active effects.
    """
    if fn is None:
        return functools.partial(effectful, type=type)
    if getattr(fn, '_is_effectful', None):
        return fn
    assert type is not None, f'must provide a type label for operation {fn}'
    assert type != 'message', "cannot use 'message' as keyword"

    @functools.wraps(fn)
    def _fn(*args: _P.args, name: Optional[str]=None, infer: Optional[InferDict]=None, obs: Optional[_T]=None, **kwargs: _P.kwargs) ->_T:
        is_observed = obs is not None
        if not am_i_wrapped():
            return fn(*args, **kwargs)
        else:
            msg = Message(type=type, name=name, fn=fn, is_observed=is_observed, args=args, kwargs=kwargs, value=obs, scale=1.0, mask=None, cond_indep_stack=(), done=False, stop=False, continuation=None, infer=infer if infer is not None else {})
            apply_stack(msg)
            if TYPE_CHECKING:
                assert msg['value'] is not None
            return msg['value']
    _fn._is_effectful = True
    return _fn


def param(name: 'str', init_tensor: 'Union[torch.Tensor, Callable[[], torch.Tensor], None]'=None, constraint: 'constraints.Constraint'=constraints.real, event_dim: 'Optional[int]'=None) ->torch.Tensor:
    """
    Saves the variable as a parameter in the param store.
    To interact with the param store or write to disk,
    see `Parameters <parameters.html>`_.

    :param str name: name of parameter
    :param init_tensor: initial tensor or lazy callable that returns a tensor.
        For large tensors, it may be cheaper to write e.g.
        ``lambda: torch.randn(100000)``, which will only be evaluated on the
        initial statement.
    :type init_tensor: torch.Tensor or callable
    :param constraint: torch constraint, defaults to ``constraints.real``.
    :type constraint: torch.distributions.constraints.Constraint
    :param int event_dim: (optional) number of rightmost dimensions unrelated
        to batching. Dimension to the left of this will be considered batch
        dimensions; if the param statement is inside a subsampled plate, then
        corresponding batch dimensions of the parameter will be correspondingly
        subsampled. If unspecified, all dimensions will be considered event
        dims and no subsampling will be performed.
    :returns: A constrained parameter. The underlying unconstrained parameter
        is accessible via ``pyro.param(...).unconstrained()``, where
        ``.unconstrained`` is a weakref attribute.
    :rtype: torch.Tensor
    """
    args = (name,) if init_tensor is None else (name, init_tensor)
    value = _param(*args, constraint=constraint, event_dim=event_dim, name=name)
    assert value is not None
    return value


def enable_validation(is_validate: 'bool'=True) ->None:
    """
    Enable or disable validation checks in Pyro. Validation checks provide
    useful warnings and errors, e.g. NaN checks, validating distribution
    arguments and support values, detecting incorrect use of ELBO and MCMC.
    Since some of these checks may be expensive, you may want to disable
    validation of mature models to speed up inference.

    The default behavior mimics Python's ``assert`` statement: validation is on
    by default, but is disabled if Python is run in optimized mode (via
    ``python -O``). Equivalently, the default behavior depends on Python's
    global ``__debug__`` value via ``pyro.enable_validation(__debug__)``.

    Validation is temporarily disabled during jit compilation, for all
    inference algorithms that support the PyTorch jit. We recommend developing
    models with non-jitted inference algorithms to ease debugging, then
    optionally moving to jitted inference once a model is correct.

    :param bool is_validate: (optional; defaults to True) whether to
        enable validation checks.
    """
    dist.enable_validation(is_validate)
    infer.enable_validation(is_validate)
    poutine.enable_validation(is_validate)


class JitTrace_ELBO:

    def __init__(self, **kwargs):
        self.ignore_jit_warnings = kwargs.pop('ignore_jit_warnings', False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, model, guide, *args):
        if self._param_trace is None:
            with block(), trace() as tr, block(hide_fn=lambda m: m['type'] != 'param'):
                elbo(model, guide, *args)
            self._param_trace = tr
        unconstrained_params = tuple(param(name).unconstrained() for name in self._param_trace)
        params_and_args = unconstrained_params + args
        if self._compiled is None:

            def compiled(*params_and_args):
                unconstrained_params = params_and_args[:len(self._param_trace)]
                args = params_and_args[len(self._param_trace):]
                for name, unconstrained_param in zip(self._param_trace, unconstrained_params):
                    constrained_param = param(name)
                    assert constrained_param.unconstrained() is unconstrained_param
                    self._param_trace[name]['value'] = constrained_param
                return replay(elbo, guide_trace=self._param_trace)(model, guide, *args)
            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(compiled, params_and_args, check_trace=False)
        return self._compiled(*params_and_args)


class ELBOModule(torch.nn.Module):

    def __init__(self, model: 'torch.nn.Module', guide: 'torch.nn.Module', elbo: "'ELBO'"):
        super().__init__()
        self.model = model
        self.guide = guide
        self.elbo = elbo

    def forward(self, *args, **kwargs):
        return self.elbo.differentiable_loss(self.model, self.guide, *args, **kwargs)

