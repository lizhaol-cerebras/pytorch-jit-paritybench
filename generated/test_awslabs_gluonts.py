
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


import numpy as np


from matplotlib import pyplot as plt


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import math


import pandas as pd


import random


import time


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


import copy


from copy import deepcopy


import torch as pt


from typing import Union


from itertools import product


from torch.utils.data import Dataset as TorchDataset


from torch import Tensor


from typing import Callable


from functools import partial


from itertools import chain


from torch import BoolTensor


from torch.utils.data import Dataset


from torch.optim import AdamW


from typing import Iterator


from collections import defaultdict


import warnings


from torch import LongTensor


from torch import nn


from torch.nn import Parameter


from torch.nn import functional as F


from typing import TYPE_CHECKING


from torch.nn import init


from typing import NamedTuple


import re


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from typing import Sequence


from typing import overload


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import StandardScaler


from numpy import ndarray


from typing import Any


from itertools import repeat


from torch.nn.utils import clip_grad_norm_


from torch import distributed as dist


from typing import Iterable


from collections import OrderedDict


from functools import reduce


from torch.nn.parallel import DistributedDataParallel


from abc import ABC


from abc import abstractmethod


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.distributions import Distribution


from torch.distributions import Normal as Gaussian


from torch.nn.functional import linear


from torch.nn.functional import conv1d


from collections import deque


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import IterableDataset


from typing import Literal


from sklearn.gaussian_process import GaussianProcessRegressor


from sklearn.gaussian_process.kernels import RBF


from torch.utils.data import ConcatDataset


from functools import cached_property


from sklearn.neighbors import NearestNeighbors


import numpy.ma as ma


from itertools import cycle


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Type


from typing import TypeVar


import itertools


import matplotlib.pyplot as plt


from typing import cast


from torch.distributions import TransformedDistribution


from torch.distributions import AffineTransform


from torch.distributions import constraints


from torch.distributions import NegativeBinomial


from torch.distributions import Poisson


from torch.distributions.utils import broadcast_all


from torch.distributions.utils import lazy_property


from matplotlib.pyplot import sca


from torch.distributions import Beta


from torch.distributions import Gamma


from torch.distributions import Normal


from torch.distributions import StudentT


from torch.utils import data


from torch.nn.modules import loss


import inspect


from abc import abstractclassmethod


from torch.distributions import Categorical


from torch.distributions import MixtureSameFamily


from torch.distributions import Independent


from torch.distributions import LowRankMultivariateNormal


from torch.distributions import MultivariateNormal


from inspect import isfunction


from torch import einsum


from math import pi


from torch import nn as nn


from torch.optim import Adam


from torch.optim.lr_scheduler import OneCycleLR


import torch.distributions


from scipy import stats


import torch.nn


from torch.distributions.normal import Normal


from numbers import Number


from torch.distributions.distribution import Distribution


import torch.optim


import matplotlib


import logging


import numpy.typing as npt


from torch.utils.data import TensorDataset


from torch import optim


from torch.distributions import Laplace


from torch.distributions import NegativeBinomial as TorchNegativeBinomial


from scipy.stats import nbinom


from scipy.stats import t as ScipyStudentT


from torch.distributions import StudentT as TorchStudentT


from torch.distributions import Uniform


from itertools import islice


from scipy.special import softmax


from torch.optim import SGD


class ARNetworkBase(nn.Module):

    def __init__(self, prediction_length: 'int', context_length: 'int') ->None:
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.criterion = nn.SmoothL1Loss(reduction='none')
        modules = []
        modules.append(nn.Linear(context_length, prediction_length))
        self.linear = nn.Sequential(*modules)


class ARTrainingNetwork(ARNetworkBase):

    def forward(self, past_target: 'torch.Tensor', future_target: 'torch.Tensor') ->torch.Tensor:
        nu = min(torch.mean(past_target).item(), torch.mean(future_target).item())
        past_target /= 1 + nu
        future_target /= 1 + nu
        prediction = self.linear(past_target)
        loss = self.criterion(prediction, future_target)
        return loss


class ARPredictionNetwork(ARNetworkBase):

    def __init__(self, num_parallel_samples: 'int'=100, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(self, past_target: 'torch.Tensor') ->torch.Tensor:
        pass


class LSTMNetworkBase(nn.Module):

    def __init__(self, prediction_length: 'int', context_length: 'int', input_size: 'int'=1, hidden_layer_size: 'int'=100, num_layers: 'int'=2) ->None:
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, prediction_length)


class LSTMTrainingNetwork(LSTMNetworkBase):

    def forward(self, past_target: 'torch.Tensor', future_target: 'torch.Tensor') ->torch.Tensor:
        nu = min(torch.mean(past_target).item(), torch.mean(future_target).item())
        past_target /= 1 + nu
        future_target /= 1 + nu
        inputs = past_target.view(past_target.shape[1], past_target.shape[0], 1)
        lstm_out, _ = self.lstm(inputs)
        prediction = self.linear(lstm_out)
        loss = self.criterion(prediction[-1], future_target)
        return loss


class LSTMPredictionNetwork(LSTMNetworkBase):

    def __init__(self, num_parallel_samples: 'int'=100, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(self, past_target: 'torch.Tensor') ->torch.Tensor:
        pass


class DomAdaptEstimator(nn.Module):

    def __init__(self, src_module: 'AttentionEstimator', tgt_module: 'AttentionEstimator', balance_loss: 'bool'=True, forecast_target: 'bool'=True) ->None:
        super(DomAdaptEstimator, self).__init__()
        self.src = src_module
        self.tgt = tgt_module
        self.balance_loss = balance_loss
        self.forecast_target = forecast_target

    def forward(self, src_data: 'Tensor', tgt_data: 'Tensor', src_feats: 'Optional[Tensor]'=None, tgt_feats: 'Optional[Tensor]'=None, src_nan_mask: 'Optional[BoolTensor]'=None, tgt_nan_mask: 'Optional[BoolTensor]'=None, src_length: 'Optional[LongTensor]'=None, tgt_length: 'Optional[LongTensor]'=None) ->Tensor:
        src_loss = self.src(src_data, src_feats, src_nan_mask, src_length)
        tgt_loss = self.tgt(tgt_data, tgt_feats, tgt_nan_mask, tgt_length)
        if not self.forecast_target:
            tgt_loss = tgt_loss - self.tgt.tradeoff * self.tgt.fc_loss
        if self.balance_loss:
            src_scale = self.src._normalizer._buffers['scale']
            tgt_scale = self.tgt._normalizer._buffers['scale']
            src_scale = src_scale.view(src_scale.size(0), -1)
            tgt_scale = tgt_scale.view(tgt_scale.size(0), -1)
            weight = pt.mean(tgt_scale / src_scale, dim=1)
            src_loss = src_loss * weight
        loss = src_loss + tgt_loss
        return loss


class AdversarialDomAdaptEstimator(DomAdaptEstimator):

    def __init__(self, src_module: 'AdversarialEstimator', tgt_module: 'AdversarialEstimator', balance_loss: 'bool'=True, forecast_target: 'bool'=True, disc_lambda: 'float'=1.0) ->None:
        super(AdversarialDomAdaptEstimator, self).__init__(src_module, tgt_module, balance_loss, forecast_target)
        self.disc_lambda = disc_lambda
        self._generative = True

    def generative(self):
        self._generative = True
        self.src.generative()
        self.tgt.generative()

    def discriminative(self):
        self._generative = False
        self.src.discriminative()
        self.tgt.discriminative()

    def forward(self, src_data: 'Tensor', tgt_data: 'Tensor', src_feats: 'Optional[Tensor]'=None, tgt_feats: 'Optional[Tensor]'=None, src_nan_mask: 'Optional[BoolTensor]'=None, tgt_nan_mask: 'Optional[BoolTensor]'=None, src_length: 'Optional[LongTensor]'=None, tgt_length: 'Optional[LongTensor]'=None) ->Tensor:
        gen_loss = super(AdversarialDomAdaptEstimator, self).forward(src_data, tgt_data, src_feats, tgt_feats, src_nan_mask, tgt_nan_mask, src_length, tgt_length)
        batch_size = gen_loss.size(0)
        src_prob_domain = self.src.prob_domain.view(batch_size, -1)
        tgt_prob_domain = self.tgt.prob_domain.view(batch_size, -1)
        adv_type = 'grad_rev'
        disc_loss = -src_prob_domain.mean(dim=1) - pt.log(1.0 - tgt_prob_domain.exp() + 1e-10).mean(dim=1)
        if self._generative:
            if adv_type == 'grad_rev':
                loss = gen_loss - self.disc_lambda * disc_loss
            elif adv_type == 'label_inv':
                invert_tgt_loss = -tgt_prob_domain.mean(dim=1)
                loss = gen_loss - self.disc_lambda * invert_tgt_loss
        else:
            loss = disc_loss
        return loss


class AttentionBlock(nn.Module):

    def __init__(self, encoder: 'EncoderModule', kernel: 'AttentionKernel', decoder: 'DecoderModule') ->None:
        super(AttentionBlock, self).__init__()
        self.encoder = encoder
        self.kernel = kernel
        self.decoder = decoder
        self.window_size = max(self.encoder.window_size)
        self.register_buffer('shape', None, persistent=False)
        self.register_buffer('query', None, persistent=False)
        self.register_buffer('key', None, persistent=False)
        self.register_buffer('inter_score', None, persistent=False)
        self.register_buffer('extra_score', None, persistent=False)
        self.register_buffer('inter_value', None, persistent=False)
        self.register_buffer('extra_value', None, persistent=False)

    def forward(self, data: 'Tensor', feats: 'Optional[Tensor]', mask: 'Optional[BoolTensor]') ->Tuple[Tensor, Tensor]:
        shape, inter_value, extra_value = self.encoder(data, feats)
        self.shape = shape.detach()
        self.inter_value = inter_value.detach()
        self.extra_value = extra_value.detach()
        interp, extrap = self.kernel(shape, inter_value, extra_value, mask)
        self.query = self.kernel._query
        self.key = self.kernel._key
        self.inter_score = self.kernel._inter_score
        self.extra_score = self.kernel._extra_score
        interp = self.decoder(interp)
        extrap = self.decoder(extrap)
        return interp, extrap


class DecoderModule(nn.Module):

    def __init__(self, d_data: 'int', d_hidden: 'int') ->None:
        super(DecoderModule, self).__init__()
        self.d_data = d_data
        self.d_hidden = d_hidden
        self.weight = nn.Parameter(Tensor(self.d_data, self.d_hidden))
        self.bias = nn.Parameter(Tensor(d_data))
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, data: 'Tensor'):
        return F.linear(data, self.weight, self.bias)


T = TypeVar('T')


U = TypeVar('U')


R = TypeVar('R')

