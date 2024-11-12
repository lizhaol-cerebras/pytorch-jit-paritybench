
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


import torch.nn as nn


from typing import Type


from torch.autograd import Function


import math


from torch.optim import Optimizer


from torch.optim.optimizer import required


from torch.nn.utils import clip_grad_norm_


import logging


from typing import List


import torch.distributions as tdist


import numpy as np


import torch.nn.functional as F


from typing import Optional


import warnings


import pandas as pd


import matplotlib.pyplot as plt


from torch import nn


from torch.autograd import Variable


from typing import Tuple


from torch.nn import functional as tf


import typing


from torch import optim


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader


from typing import Union


from typing import Callable


from typing import Dict


import sklearn.metrics


import random


from torch.nn import MultiheadAttention


from torch.optim import Adam


from torch.optim import SGD


from torch.nn import MSELoss


from torch.nn import SmoothL1Loss


from torch.nn import PoissonNLLLoss


from torch.nn import L1Loss


from torch.nn import CrossEntropyLoss


from torch.nn import BCELoss


from torch.nn import BCEWithLogitsLoss


from typing import Any


from torch import einsum


from torch.utils.data import Dataset


from copy import deepcopy


import torch.optim as optim


from abc import ABC


from abc import abstractmethod


from math import sqrt


from math import ceil


from math import pi


from torch.nn.modules.activation import MultiheadAttention


from torch.nn.modules import Transformer


from torch.nn.modules import TransformerEncoder


from torch.nn.modules import TransformerEncoderLayer


from torch.nn.modules import LayerNorm


import copy


from torch.nn.parameter import Parameter


from sklearn.preprocessing import StandardScaler


import numpy


def the_last1(tensor: 'torch.Tensor', out_len: 'int') ->torch.Tensor:
    """Creates a tensor based on the last element.

    :param tensor: A tensor of dimension (batch_size, seq_len, n_time_series)
    :param out_len: The length or the forecast_length
    :type out_len: int

    :return: Returns a tensor of (batch_size, out_len, 1)
    :rtype: torch.Tensor
    """
    return tensor[:, -1, :].unsqueeze(0).permute(1, 0, 2).repeat(1, out_len, 1)


class NaiveBase(torch.nn.Module):
    """A very simple baseline model that returns the fixed value based on the input sequence.

    No learning used at all.
    """

    def __init__(self, seq_length: 'int', n_time_series: 'int', output_seq_len=1, metric: 'str'='last'):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.output_layer = torch.nn.Linear(seq_length, output_seq_len)
        self.metric_dict = {'last': the_last1}
        self.output_seq_len = output_seq_len
        self.metric_function = self.metric_dict[metric]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.metric_function(x, self.output_seq_len)


class NLinear(nn.Module):
    """Normalization-Linear."""

    def __init__(self, forecast_history: 'int', forecast_length: 'int', enc_in=128, individual=False, n_targs=1):
        super(NLinear, self).__init__()
        self.seq_len = forecast_history
        self.pred_len2 = forecast_length
        self.channels = enc_in
        self.individual = individual
        self.n_targs = n_targs
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len2))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len2, x.size(2)], dtype=x.dtype)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        if self.n_targs == 1:
            return x[:, :, -1]
        return x


class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: 'torch.Tensor'):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """Decomposition-Linear."""

    def __init__(self, forecast_history: 'int', forecast_length: 'int', individual, enc_in: 'int', n_targs=1):
        """Code from.

        :param forecast_history: _description_
        :type forecast_history: int
        :param forecast_length: _description_
        :type forecast_length: int
        :param individual: _description_
        :type individual: _type_
        :param enc_in: _description_
        :type enc_in: int
        """
        super(DLinear, self).__init__()
        self.seq_len = forecast_history
        self.pred_len2 = forecast_length
        self.n_targs = n_targs
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len2))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len2))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len2)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len2)

    def forward(self, x: 'torch.Tensor'):
        """The.

        :param x: PyTorch tensor of size [Batch, Input length, Channel]
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len2], dtype=seasonal_init.dtype)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len2], dtype=trend_init.dtype)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        if self.n_targs == 1:
            return x[:, :, -1]
        else:
            return x


class VanillaGRU(torch.nn.Module):

    def __init__(self, n_time_series: 'int', hidden_dim: 'int', num_layers: 'int', n_target: 'int', dropout: 'float', forecast_length=1, use_hidden=False, probabilistic=False):
        """Simple GRU to preform deep time series forecasting.

        :param n_time_series: The number of time series present in the data
        :type n_time_series int:
        :param hidden_dim:
        :type hidden_dim:

        Note for probablistic n_targets must be set to two and actual multiple targs are not supported now.
        """
        super(VanillaGRU, self).__init__()
        self.layer_dim = num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.use_hidden = use_hidden
        self.forecast_length = forecast_length
        self.probablistic = probabilistic
        self.gru = torch.nn.GRU(n_time_series, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, n_target)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function for GRU.

        :param x: torch of shape
        :type model: torch.Tensor
        :return: Returns a tensor of shape (batch_size, forecast_length, n_target) or (batch_size, n_target)
        :rtype: torch.Tensor
        """
        if self.hidden is not None and self.use_hidden:
            h0 = self.hidden
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, self.hidden = self.gru(x, h0.detach())
        out = out[:, -self.forecast_length:, :]
        out = self.fc(out)
        if self.probablistic:
            mean = out[..., 0][..., None]
            std = torch.clamp(out[..., 1][..., None], min=0.01)
            y_pred = torch.distributions.Normal(mean, std)
            return y_pred
        if self.fc.out_features == 1:
            return out[:, :, 0]
        return out


class SimpleLinearModel(torch.nn.Module):
    """A very simple baseline linear model to resolve some of the difficulties with bugs in the various train/validation
    loops in code.

    Has only two layers.
    """

    def __init__(self, seq_length: 'int', n_time_series: 'int', output_seq_len=1, probabilistic: 'bool'=False):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.probabilistic = probabilistic
        if self.probabilistic:
            self.output_len = 2
        else:
            self.output_len = output_seq_len
        self.output_layer = torch.nn.Linear(seq_length, self.output_len)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        x: A tensor of dimension (B, L, M) where
        B is the batch size, L is the length of the sequence
        """
        x = self.initial_layer(x)
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        if self.probabilistic:
            mean = x[..., 0][..., None]
            std = torch.clamp(x[..., 1][..., None], min=0.01)
            return torch.distributions.Normal(mean, std)
        else:
            return x.view(-1, self.output_len)


class LSTMForecast(torch.nn.Module):
    """A very simple baseline LSTM model that returns an output sequence given a multi-dimensional input seq.

    Inspired by the StackOverflow link below.
    https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
    """

    def __init__(self, seq_length: 'int', n_time_series: 'int', output_seq_len=1, hidden_states: 'int'=20, num_layers=2, bias=True, batch_size=100, probabilistic=False):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.hidden_dim = hidden_states
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(n_time_series, hidden_states, num_layers, bias, batch_first=True)
        self.probabilistic = probabilistic
        if self.probabilistic:
            output_seq_len = 2
        self.final_layer = torch.nn.Linear(seq_length * hidden_states, output_seq_len)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_hidden(batch_size)

    def init_hidden(self, batch_size: 'int') ->None:
        """[summary]

        :param batch_size: [description]
        :type batch_size: int
        """
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim), torch.zeros(self.num_layers, batch_size, self.hidden_dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        out_x, self.hidden = self.lstm(x, self.hidden)
        x = self.final_layer(out_x.contiguous().view(batch_size, -1))
        if self.probabilistic:
            mean = x[..., 0][..., None]
            std = torch.clamp(x[..., 1][..., None], min=0.01)
            x = torch.distributions.Normal(mean, std)
        return x


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim
    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """
    if k is None or k >= X.shape[dim]:
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)
    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum
    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size
    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)
        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_
    return tau, support_size


class SparsemaxFunction(Function):

    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


def sparsemax(X, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).
    Solves the projection:
        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return SparsemaxFunction.apply(X, dim, k)


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
        super(Sparsemax, self).__init__()

    def forward(self, X):
        return sparsemax(X, dim=self.dim, k=self.k)


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """
    if k is None or k >= X.shape[dim]:
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)
    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)
    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)
        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_
    return tau_star, support_size


class Entmax15Function(Function):

    @classmethod
    def forward(cls, ctx, X: 'torch.Tensor', dim=0, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        X = X / 2
        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)
        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def entmax15(X, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return Entmax15Function.apply(X, dim, k)


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
        super(Entmax15, self).__init__()

    def forward(self, X: 'torch.Tensor'):
        return entmax15(X, dim=self.dim, k=self.k)


class MASELoss(torch.nn.Module):

    def __init__(self, baseline_method):
        """This implements the MASE loss function (e.g. MAE_MODEL/MAE_NAIEVE)"""
        super(MASELoss, self).__init__()
        self.method_dict = {'mean': lambda x, y: torch.mean(x, 1).unsqueeze(1).repeat(1, y[1], 1)}
        self.baseline_method = self.method_dict[baseline_method]

    def forward(self, target: 'torch.Tensor', output: 'torch.Tensor', train_data: 'torch.Tensor', m=1) ->torch.Tensor:
        if len(train_data.shape) < 3:
            train_data = train_data.unsqueeze(0)
        if m == 1 and len(target.shape) == 1:
            output = output.unsqueeze(0)
            output = output.unsqueeze(2)
            target = target.unsqueeze(0)
            target = target.unsqueeze(2)
        if len(target.shape) == 2:
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
        result_baseline = self.baseline_method(train_data, output.shape)
        MAE = torch.nn.L1Loss()
        mae2 = MAE(output, target)
        mase4 = MAE(result_baseline, target)
        if mase4 < 0.001:
            mase4 = 0.001
        return mae2 / mase4


class RMSELoss(torch.nn.Module):
    """Returns RMSE using:

    target -> True y
    output -> Prediction by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor'):
        if len(output) > 1:
            diff = torch.sub(target, output)
            std_dev = torch.std(diff)
            var_penalty = self.variance_penalty * std_dev
            None
            None
            None
            return torch.sqrt(self.mse(target, output)) + var_penalty
        else:
            return torch.sqrt(self.mse(target, output))


class MAPELoss(torch.nn.Module):
    """Returns MAPE using:

    target -> True y output -> Predtion by model
    """

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.variance_penalty = variance_penalty

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor'):
        if len(output) > 1:
            return torch.mean(torch.abs(torch.sub(target, output) / target)) + self.variance_penalty * torch.std(torch.sub(target, output))
        else:
            return torch.mean(torch.abs(torch.sub(target, output) / target))


class PenalizedMSELoss(torch.nn.Module):
    """Returns MSE using:

    target -> True y
    output -> Predtion by model
    source: https://discuss.pytorch.org/t/rmse-loss-function/16540/3
    """

    def __init__(self, variance_penalty=0.0):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.variance_penalty = variance_penalty

    def forward(self, output: 'torch.Tensor', target: 'torch.Tensor'):
        return self.mse(target, output) + self.variance_penalty * torch.std(torch.sub(target, output))


class GaussianLoss(torch.nn.Module):

    def __init__(self, mu=0, sigma=0):
        """Compute the negative log likelihood of Gaussian Distribution From https://arxiv.org/abs/1907.00235."""
        super(GaussianLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: 'torch.Tensor'):
        loss = -tdist.Normal(self.mu, self.sigma).log_prob(x)
        return torch.sum(loss) / (loss.size(0) * loss.size(1))


class QuantileLoss(torch.nn.Module):
    """From https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629."""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class NegativeLogLikelihood(torch.nn.Module):
    """target -> True y output -> predicted distribution."""

    def __init__(self):
        super().__init__()

    def forward(self, output: 'torch.distributions', target: 'torch.Tensor'):
        """calculates NegativeLogLikelihood."""
        return -output.log_prob(target).sum()


class PathDTWBatch(Function):

    @staticmethod
    def forward(ctx, D, gamma):
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma])
        grad_gpu = torch.zeros((batch_size, N, N))
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3))
        E_gpu = torch.zeros((batch_size, N + 2, N + 2))
        for k in range(0, batch_size):
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()
        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N))
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma)
            Hessian[k:k + 1, :, :] = torch.FloatTensor(hess_k)
        return Hessian, None


class SoftDTWBatch(Function):

    @staticmethod
    def forward(ctx, D, gamma=1.0):
        dev = D.device
        batch_size, N, N = D.shape
        gamma = torch.FloatTensor([gamma])
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        total_loss = 0
        R = torch.zeros((batch_size, N + 2, N + 2))
        for k in range(0, batch_size):
            Rk = torch.FloatTensor(compute_softdtw(D_[k, :, :], g_))
            R[k:k + 1, :, :] = Rk
            total_loss = total_loss + Rk[-2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.zeros((batch_size, N, N))
        for k in range(batch_size):
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k, :, :], R_[k, :, :], g_))
            E[k:k + 1, :, :] = Ek
        return grad_output * E, None


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


class DilateLoss(torch.nn.Module):

    def __init__(self, gamma=0.001, alpha=0.5):
        """Dilate loss function originally from https://github.com/manjot4/NIPS-Reproducibility-Challenge."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, targets: 'torch.Tensor', outputs: 'torch.Tensor'):
        """
        :targets: tensor of dimension (batch_size, out_seq_len, 1)
        :outputs: tensor of dimension (batch_size, out_seq_len, 1)
        :returns a tuple of dimension (torch.Tensor)
        """
        outputs = outputs.float()
        targets = targets.float()
        if len(targets.size()) < 2:
            None
            targets = targets.unsqueeze(0)
            outputs = outputs.unsqueeze(0)
        if len(targets.size()) < 3:
            outputs = outputs.unsqueeze(2)
            targets = targets.unsqueeze(2)
        batch_size, N_output = outputs.shape[0:2]
        loss_shape = 0
        softdtw_batch = SoftDTWBatch.apply
        D = torch.zeros((batch_size, N_output, N_output))
        for k in range(batch_size):
            Dk = pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = softdtw_batch(D, self.gamma)
        path_dtw = PathDTWBatch.apply
        path = path_dtw(D, self.gamma)
        Omega = pairwise_distances(torch.range(1, N_output).view(N_output, 1))
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss


def one_hot(labels: 'torch.Tensor', num_classes: 'int', device: 'Optional[torch.device]'=None, dtype: 'Optional[torch.dtype]'=None, eps: 'float'=1e-06) ->torch.Tensor:
    """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f'Input labels type is not a torch.Tensor. Got {type(labels)}')
    if not labels.dtype == torch.int64:
        raise ValueError(f'labels must be of the same dtype torch.int64. Got: {labels.dtype}')
    if num_classes < 1:
        raise ValueError('The number of classes must be bigger than one. Got: {}'.format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(input: 'torch.Tensor', target: 'torch.Tensor', alpha: 'float', gamma: 'float'=2.0, reduction: 'str'='none', eps: 'Optional[float]'=None) ->torch.Tensor:
    """Criterion that computes Focal loss. According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn('`focal_loss` has been reworked for improved numerical stability and the `eps` argument is no longer necessary', DeprecationWarning, stacklevel=2)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not len(input.shape) >= 2:
        raise ValueError(f'Invalid input shape, we expect BxCx*. Got: {input.shape}')
    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')
    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')
    if not input.device == target.device:
        raise ValueError(f'input and target must be in the same device. Got: {input.device} and {target.device}')
    input_soft: 'torch.Tensor' = F.softmax(input, dim=1)
    log_input_soft: 'torch.Tensor' = F.log_softmax(input, dim=1)
    target_one_hot: 'torch.Tensor' = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)
    weight = torch.pow(-input_soft + 1.0, gamma)
    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f'Invalid reduction mode: {reduction}')
    return loss


class FocalLoss(nn.Module):
    """Criterion that computes Focal loss. According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: 'float', gamma: 'float'=2.0, reduction: 'str'='none', eps: 'Optional[float]'=None) ->None:
        super().__init__()
        self.alpha: 'float' = alpha
        self.gamma: 'float' = gamma
        self.reduction: 'str' = reduction
        self.eps: 'Optional[float]' = eps

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        if len(target.shape) == 3:
            target = target[:, 0, :]
        if len(target.shape) == 2:
            target = target[:, 0]
        if len(input.shape) == 3:
            input = input[:, 0, :]
        target = target.type(torch.int64)
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


def binary_focal_loss_with_logits(input: 'torch.Tensor', target: 'torch.Tensor', alpha: 'float'=0.25, gamma: 'float'=2.0, reduction: 'str'='none', eps: 'Optional[float]'=None) ->torch.Tensor:
    """Function that computes Binary Focal loss.

    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
    Returns:
        the computed loss.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn('`binary_focal_loss_with_logits` has been reworked for improved numerical stability and the `eps` argument is no longer necessary', DeprecationWarning, stacklevel=2)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not len(input.shape) >= 2:
        raise ValueError(f'Invalid input shape, we expect BxCx*. Got: {input.shape}')
    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')
    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input) - (1 - alpha) * torch.pow(probs_pos, gamma) * (1.0 - target) * F.logsigmoid(-input)
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f'Invalid reduction mode: {reduction}')
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    """Criterion that computes Focal loss. According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha): Weighting factor for the rare class :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: 'float', gamma: 'float'=2.0, reduction: 'str'='none') ->None:
        super().__init__()
        self.alpha: 'float' = alpha
        self.gamma: 'float' = gamma
        self.reduction: 'str' = reduction

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction)


class Concatenation(torch.nn.Module):

    def __init__(self, cat_dim: 'int', repeat: 'bool'=True, use_layer: 'bool'=False, combined_d: 'int'=1, out_shape: 'int'=1):
        """A function to combine two tensors together via concantenation.

        :param cat_dim: The dimension that you want to concatenate along (e.g. 0, 1, 2)
        :type cat_dim: int
        :param repeat: boolean of whether to repeate meta_data along temporal_dim , defaults to True
        :type repeat: bool, optional
        :param use_layer: to use a layer to get the final out_shape , defaults to False
        :type use_layer: bool, optional
        :param combined_shape: The final combined shape, defaults to 1
        :type combined_shape: int, optional
        :param out_shape: The output shape you want, defaults to 1
        :type out_shape: int, optional
        """
        super().__init__()
        self.combined_shape = combined_d
        self.out_shape = out_shape
        self.cat_dim = cat_dim
        self.repeat = repeat
        self.use_layer = use_layer
        if self.use_layer:
            self.linear = torch.nn.Linear(combined_d, out_shape)

    def forward(self, temporal_data: 'torch.Tensor', meta_data: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            temporal_data: (batch_size, seq_len, d_model)
            meta_data (batch_size, d_embedding)
        """
        if self.repeat:
            meta_data = meta_data.repeat(1, temporal_data.shape[1], 1)
        else:
            pass
        x = torch.cat((temporal_data, meta_data), self.cat_dim)
        if self.use_layer:
            x = self.linear(x)
        return x


class MultiModalSelfAttention(torch.nn.Module):

    def __init__(self, d_model: 'int', n_heads: 'int', dropout: 'float'):
        """Uses self-attention to combine the meta-data and the temporal data.

        :param d_model: The dimension of the meta-data
        :type d_model: int
        :param n_heads: The number of heads to use in multi-head mechanism
        :type n_heads: int
        :param dropout: The dropout score as a flow
        :type dropout: float
        """
        super().__init__()
        self.main_layer = MultiheadAttention(d_model, n_heads, dropout)

    def forward(self, temporal_data: 'torch.Tensor', meta_data: 'torch.Tensor') ->torch.Tensor:
        meta_data = meta_data.permute(2, 0, 1)
        temporal_data = temporal_data.permute(1, 0, 2)
        x = self.main_layer(temporal_data, meta_data, meta_data)
        return x.permute(1, 0, 2)


class MergingModel(torch.nn.Module):

    def __init__(self, method: 'str', other_params: 'Dict'):
        """A model meant to help merge meta-data with the temporal data.

        :param method: The method you want to use (Bilinear, Bilinear2, MultiAttn, Concat)
        :type method: str
        :param other_params: A dictionary of the additional parameters necessary to init the inner part.
        :type other_params: Dict

        ..code-block:: python

            merging_mod = MergingModel("Bilinear", {"in_features1": 5, "in_features_2":1, "out_features":40 })
            print(merging_mod(torch.rand(4, 5, 128), torch.rand(128)).shape) # (4, 40, 128)
        ...
        """
        super().__init__()
        self.method_dict = {'Bilinear': torch.nn.Bilinear, 'Bilinear2': torch.nn.Bilinear, 'MultiAttn': MultiModalSelfAttention, 'Concat': Concatenation, 'Other': 'other'}
        self.method_layer = self.method_dict[method](**other_params)
        self.method = method

    def forward(self, temporal_data: 'torch.Tensor', meta_data: 'torch.Tensor'):
        """Performs the forward pass on both meta and temporal data. Returns merged tensor.

        :param temporal_data: The temporal data should be in shape (batch_size, n_time_series, n_feats)
        :type temporal_data: torch.Tensor
        :param meta_data: The meta-data passed to the model will have dimension (d_meta)
        :type meta_data: torch.Tensor
        :return: The combined tensor with both the meta-data and temporal data. Shape will vary.
        :rtype: torch.Tensor
        """
        batch_size = temporal_data.shape[0]
        meta_data = meta_data.repeat(batch_size, 1).unsqueeze(1)
        if self.method == 'Bilinear':
            meta_data = meta_data.permute(0, 2, 1)
            temporal_data = temporal_data.permute(0, 2, 1).contiguous()
            x = self.method_layer(temporal_data, meta_data)
            x = x.permute(0, 2, 1)
        elif self.method == 'Bilinear2':
            temporal_shape = temporal_data.shape[1]
            meta_data = meta_data.repeat(1, temporal_shape, 1)
            x = self.method_layer(temporal_data, meta_data)
        else:
            x = self.method_layer(temporal_data, meta_data)
        return x


class MetaMerger(nn.Module):

    def __init__(self, meta_params, meta_method, embed_shape, in_shape):
        super().__init__()
        self.method_layer = meta_method
        if meta_method == 'down_sample':
            self.initial_layer = torch.nn.Linear(embed_shape, in_shape)
        elif meta_method == 'up_sample':
            self.initial_layer = torch.nn.Linear(in_shape, embed_shape)
        self.model_merger = MergingModel(meta_params['method'], meta_params['params'])

    def forward(self, temporal_data, meta_data):
        if self.method_layer == 'down_sample':
            meta_data = self.initial_layer(meta_data)
        else:
            None
        return self.model_merger(temporal_data, meta_data)


class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None):
        """_summary_

        :param layers: _description_
        :type layers: _type_
        :param norm_layer: _description_, defaults to None
        :type norm_layer: _type_, optional
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None) ->torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


def swish(x):
    return x * torch.sigmoid(x)


activation_dict = {'ReLU': torch.nn.ReLU(), 'Softplus': torch.nn.Softplus(), 'Swish': swish, 'entmax': entmax15, 'sparsemax': sparsemax, 'Softmax': torch.nn.Softmax}


class DARNN(nn.Module):

    def __init__(self, n_time_series: 'int', hidden_size_encoder: 'int', forecast_history: 'int', decoder_hidden_size: 'int', out_feats=1, dropout=0.01, meta_data=False, gru_lstm=True, probabilistic=False, final_act=None):
        """For model benchmark information see link on side https://rb.gy/koozff.

        :param n_time_series: Number of time series present in input
        :type n_time_series: int
        :param hidden_size_encoder: dimension of the hidden state encoder
        :type hidden_size_encoder: int
        :param forecast_history: How many historic time steps to use for forecasting (add one to this number)
        :type forecast_history: int
        :param decoder_hidden_size: dimension of hidden size of the decoder
        :type decoder_hidden_size: int
        :param out_feats: The number of targets (or in classification classes), defaults to 1
        :type out_feats: int, optional
        :param dropout: defaults to .01
        :type dropout: float, optional
        :param meta_data: [description], defaults to False
        :type meta_data: bool, optional
        :param gru_lstm: Specify true if you want to use LSTM, defaults to True
        :type gru_lstm: bool, optional
        :param probabilistic: Specify true if you want to use a probablistic variation, defaults to False
        :type probabilistic: bool, optional
        """
        super().__init__()
        self.probabilistic = probabilistic
        self.encoder = Encoder(n_time_series - 1, hidden_size_encoder, forecast_history, gru_lstm, meta_data)
        self.dropout = nn.Dropout(dropout)
        self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, forecast_history, out_feats, gru_lstm, self.probabilistic)
        self.final_act = final_act
        if final_act:
            self.final_act = activation_dict[final_act]

    def forward(self, x: 'torch.Tensor', meta_data: 'torch.Tensor'=None) ->torch.Tensor:
        """Performs standard forward pass of the DARNN. Special handling of probablistic.

        :param x: The core temporal data represented as a tensor (batch_size, forecast_history, n_time_series)
        :type x: torch.Tensor
        :param meta_data: The meta-data represented as a tensor (), defaults to None
        :type meta_data: torch( ).Tensor, optional
        :return: The predictetd number should be in format
        :rtype: torch.Tensor
        """
        _, input_encoded = self.encoder(x[:, :, 1:], meta_data)
        dropped_input = self.dropout(input_encoded)
        y_pred = self.decoder(dropped_input, x[:, :, 0].unsqueeze(2))
        if self.probabilistic:
            mean = y_pred[..., 0][..., None]
            std = torch.clamp(y_pred[..., 1][..., None], min=0.01)
            y_pred = torch.distributions.Normal(mean, std)
        if self.final_act:
            return self.final_act(y_pred)
        return y_pred


class AE(nn.Module):

    def __init__(self, input_shape: 'int', out_features: 'int'):
        """A basic and simple to use AutoEncoder.

        :param input_shape: The number of features for input.
        :type input_shape: int
        :param out_features: The number of output features (that will be merged)
        :type out_features: int
        """
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=out_features)
        self.encoder_output_layer = nn.Linear(in_features=out_features, out_features=out_features)
        self.decoder_hidden_layer = nn.Linear(in_features=out_features, out_features=out_features)
        self.decoder_output_layer = nn.Linear(in_features=out_features, out_features=input_shape)

    def forward(self, features: 'torch.Tensor'):
        """Runs the full forward pass on the model. In practice this will only be done during training.

        :param features: [description]
        :type features: [type]
        :return: [description]
        :rtype: [type]

        .. code-block:: python
            auto_model = AE(10, 4)
            x = torch.rand(2, 10) # batch_size, n_features
            result = auto_model(x)
            print(result.shape) # (2, 10)
        """
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def generate_representation(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code


class Conv1D(nn.Module):

    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):

    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()
        if sparse:
            None
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)
        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros(win_len, dtype=torch.float)
        if win_len // sub_len * 2 * log_l > index:
            mask[:index + 1] = 1
        else:
            while index >= 0:
                if index - log_l + 1 < 0:
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:index + 1] = 1
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2 ** i
                    if index - new_index <= sub_len and new_index >= 0:
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: 'torch.Tensor', key, value: 'torch.Tensor', activation='Softmax'):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1000000000.0 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)
        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim), GEGLU() if use_glu else nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):

    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        """

        """
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x: 'torch.Tensor'):
        """
        Args:
            x: Input tensor of shape [B, T, C]
        """
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


class SelfAttention(nn.Module):

    def __init__(self, dim: 'int', heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, use_rotary: 'bool'=True):
        """The self-attention mechanism used in the CrossVIVIT model. It is currently not used in other models and could
        likely be consolidated with those self-attention mechanisms.

        :param dim: The input dimension of the sequence.
        :type dim: [type]
        :param heads: [description]
        :type heads: [type]
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: 'torch.Tensor', pos_emb: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Sequence of shape [B, N, D]
            pos_emb: Positional embedding of sequence's tokens of shape [B, N, D]
        """
        q = self.to_q(x)
        qkv = q, *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), qkv)
        if self.use_rotary:
            sin, cos = map(lambda t: repeat(t, 'b n d -> (b h) n d', h=self.heads), pos_emb)
            dim_rotary = sin.shape[-1]
            (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
            q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out), attn


class CrossAttention(nn.Module):

    def __init__(self, dim: 'int', heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, use_rotary: 'bool'=True):
        """
        This is the CrossAttention module primarily used in the CrossVIVIT paper. It is currently not used in other
        models but may in the future be incorporated into other multi-modal models.
        :param dim: The input dimension of the sequence.
        :type dim: int
        :param heads: The number of heads for the attention mechanism.
        :type heads: int
        :param dim_head: The dimension of the heads.
        :type dim_head: int
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, src: "Float[torch.Tensor, '']", src_pos_emb, tgt, tgt_pos_emb):
        """
        Performs the forward pass of the CrossAttention module.

        """
        q = self.to_q(tgt)
        qkv = q, *self.to_kv(src).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), qkv)
        if self.use_rotary:
            sin_src, cos_src = map(lambda t: repeat(t, 'b n d -> (b h) n d', h=self.heads), src_pos_emb)
            sin_tgt, cos_tgt = map(lambda t: repeat(t, 'b n d -> (b h) n d', h=self.heads), tgt_pos_emb)
            dim_rotary = sin_src.shape[-1]
            (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
            q = q * cos_tgt + rotate_every_two(q) * sin_tgt
            k = k * cos_src + rotate_every_two(k) * sin_src
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out), attn


class CrossPreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, ctx, src_pos_emb, ts, tgt_pos_emb):
        return self.fn(self.norm_src(ctx), src_pos_emb, self.norm_tgt(ts), tgt_pos_emb)


class CrossTransformer(nn.Module):

    def __init__(self, dim: 'int', depth: 'int', heads: 'int', dim_head: 'int', mlp_dim: 'int', image_size: 'Union[List[int], Tuple[int], int]', dropout: 'float'=0.0, use_rotary: 'bool'=True, use_glu: 'bool'=True):
        """Computes the Cross-Attention between the source and target sequences.

        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int

        """
        super().__init__()
        self.image_size = image_size
        self.cross_layers = nn.ModuleList([])
        for _ in range(depth):
            self.cross_layers.append(nn.ModuleList([CrossPreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_rotary=use_rotary)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu))]))

    def forward(self, src: 'torch.Tensor', tgt: 'torch.Tensor', src_pos_emb: 'torch.Tensor', tgt_pos_emb: 'torch.Tensor'):
        """
        :param src: Source sequence of shape [B, N, D]. In the case of CrossVIVIT. src is the encoded video_ctx. Where
        B is the batch_size*forecast_history,  N is the number_of_patches after random masking is applied and D is the
        dimension of the model. In other use cases this might differ.
        :type src: torch.Tensor
        :param tgt: Target sequence of shape [B, M, D]. In the case of CrossVIVIT. tgt is the encoded_timeseries. Where
        B is the batch_size*forecast_history, M is usually one and D is the dimension of the model. In other use cases
        this might differ.
        :type tgt: torch.Tensor
        :param src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
        :type src_pos_emb: torch.Tensor
        :param tgt_pos_emb: Positional embedding of target sequence's tokens of shape [B, M, D]
        :type tgt_pos_emb: torch.Tensor
        :return: Tuple of (tgt, attention_scores)
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]


        Performs the following computation in each layer:
        1. Self-Attention on the source sequence
            2. FFN on the source sequence
            3. Cross-Attention between target and source sequence
            4. FFN on the target sequence
        """
        attention_scores = {}
        for i in range(len(self.cross_layers)):
            cattn, cff = self.cross_layers[i]
            out, cattn_scores = cattn(src, src_pos_emb, tgt, tgt_pos_emb)
            attention_scores['cross_attention'] = cattn_scores
            tgt = out + tgt
            tgt = cff(tgt) + tgt
        return tgt, attention_scores


def get_emb(sin_inp: 'torch.Tensor') ->torch.Tensor:
    """Gets a base embedding for one dimension with sin and cos intertwined."""
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class FlowAttention(nn.Module):

    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        normalizer_row = 1.0 / torch.einsum('nhld,nhd->nhl', queries + 1e-06, keys.sum(dim=2) + 1e-06)
        normalizer_col = 1.0 / torch.einsum('nhsd,nhd->nhs', keys + 1e-06, queries.sum(dim=2) + 1e-06)
        normalizer_row_refine = torch.einsum('nhld,nhd->nhl', queries + 1e-06, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-06)
        normalizer_col_refine = torch.einsum('nhsd,nhd->nhs', keys + 1e-06, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-06)
        normalizer_row_refine = torch.sigmoid(normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (queries @ kv * normalizer_row[:, :, :, None] * normalizer_row_refine[:, :, :, None]).transpose(1, 2).contiguous()
        return x, None


class FlashAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -10000000000.0
        EPSILON = 1e-10
        O1 = torch.zeros_like(Q, requires_grad=True)
        l3 = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF
        O1 = O1
        l3 = l3
        m = m
        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE
        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))
        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)
        O_BLOCKS = list(torch.split(O1, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l3, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))
        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]
            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]
                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale
                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)
                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.0)
                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)
                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij
                O_BLOCKS[i] = li / li_new * torch.exp(mi - mi_new) * Oi + torch.exp(m_block_ij - mi_new) / li_new * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new
        O = torch.cat(O_BLOCKS, dim=2)
        l3 = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l3, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3), attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    """The Attention operation."""

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        return V.contiguous()


class ProbMask(object):

    def __init__(self, B: 'int', H, L, index, scores, device='cpu'):
        """Creates a probablistic mask.

        :param B: batch_size
        :type B: int
        :param H: Number of heads
        :type H: int
        :param L: Sequence length
        :type L: in
        :param index: [description]s
        :type index: [type]
        :param scores: [description]
        :type scores: [type]
        :param device: [description], defaults to "cpu"
        :type device: str, optional
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """The Multi-head Self-Attention (MSA) Layer."""

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out)


class ReformerLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes, causal=causal)

    def fit_length(self, queries):
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = self.bucket_size * 2 - N % (self.bucket_size * 2)
            return torch.cat([queries, torch.zeros([B, fill_len, C])], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class DSW_embedding(nn.Module):

    def __init__(self, seg_len, d_model):
        """_summary_

        :param seg_len: _description_
        :type seg_len: _type_
        :param d_model: _description_
        :type d_model: _type_
        """
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        return x_embed


class Crossformer(nn.Module):

    def __init__(self, n_time_series: 'int', forecast_history: 'int', forecast_length: 'int', seg_len: 'int', win_size=4, factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3, dropout=0.0, baseline=False, n_targs=None, device=torch.device('cuda:0')):
        """Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.
        https://github.com/Thinklab-SJTU/Crossformer

        :param n_time_series: The total number of time series
        :type n_time_series: int
        :param forecast_history: The length of the input sequence
        :type forecast_history: int
        :param forecast_length: The number of steps to forecast
        :type forecast_length: int
        :param seg_len: Parameter specific to Crossformer, forecast_history must be divisible by seg_len
        :type seg_len: int
        :param win_size: The window size for the segment merge mechanism, defaults to 4 (original paper used 2)
        :type win_size: int, optional
        :param factor: _description_, defaults to 10
        :type factor: int, optional
        :param d_model: _description_, defaults to 512
        :type d_model: int, optional
        :param d_ff: _description_, defaults to 1024
        :type d_ff: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism, defaults to 8
        :type n_heads: int, optional
        :param e_layers: The number of encoder layers, defaults to 3
        :type e_layers: int, optional
        :param dropout: The amount of dropout to use when training the model, defaults to 0.0
        :type dropout: float, optional
        :param baseline: A boolean of whether to use mean of the past time series , defaults to False
        :type baseline: bool, optional
        :param device: _description_, defaults to torch.device("cuda:0")
        :type device: str, optional
        """
        super(Crossformer, self).__init__()
        self.data_dim = n_time_series
        self.in_len = forecast_history
        self.out_len = forecast_length
        self.seg_len = seg_len
        self.merge_win = win_size
        self.n_targs = n_time_series if n_targs is None else n_targs
        self.baseline = baseline
        self.device = device
        self.pad_in_len = ceil(1.0 * forecast_history / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * forecast_length / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, n_time_series, self.pad_in_len // seg_len, d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1, dropout=dropout, in_seg_num=self.pad_in_len // seg_len, factor=factor)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, n_time_series, self.pad_out_len // seg_len, d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, out_seg_num=self.pad_out_len // seg_len, factor=factor)

    def forward(self, x_seq: 'torch.Tensor'):
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        result = base + predict_y[:, :self.out_len, :]
        res = result[:, :, :self.n_targs]
        return res


class SegMerging(nn.Module):
    """Segment Merging Layer.

    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)
        x = self.norm(x)
        x = self.linear_trans(x)
        return x


class TwoStageAttentionLayer(nn.Module):
    """The Two Stage Attention (TSA) Layer input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]"""

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: 'torch.Tensor'):
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out


class scale_block(nn.Module):
    """
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper .
    """

    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10):
        super(scale_block, self).__init__()
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None) ->torch.Tensor:
        """_summary_

        :param x: The input tensor
        :type x: _type_
        :param cross: _description_
        :type cross: _type_
        :param x_mask: _description_, defaults to None
        :type x_mask: _type_, optional
        :param cross_mask: _description_, defaults to None
        :type cross_mask: _type_, optional
        :return: _description_
        :rtype: torch.Tensor
        """
        x, attn = self.self_attention(x, x, x, attn_mask=x_mask)
        res = self.dropout(x)
        x = x + res
        x = self.norm1(x)
        x, attn = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(x)
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class PositionalEmbedding(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.d = d
        inv_freq = 1 / 10000 ** (torch.arange(0.0, d, 2.0) / d)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions: 'torch.LongTensor'):
        sinusoid_inp = torch.einsum('i,j->ij', positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class TokenEmbedding(nn.Module):

    def __init__(self, c_in: 'int', d_model: 'int'):
        """Create the token embedding.

        :param c_in: [description]
        :type c_in: [type]
        :param d_model: [description]
        :type d_model: [type]
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Create the token embedding.

        :param x: The tensor passed to create the token embedding
        :type x: torch.Tensor
        :return: [description]
        :rtype: torch.Tensor
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):

    def __init__(self, c_in: 'torch.Tensor', d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):

    def __init__(self, d_model: 'int', embed_type='fixed', lowest_level=4):
        """A class to create.

        :param d_model: The model embedding dimension
        :type d_model: int
        :param embed_tsype: [description], defaults to 'fixed'
        :type embed_type: str, optional
        :param lowest_level: [description], defaults to 4
        :type lowest_level: int, optional
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        lowest_level_map = {'month_embed': Embed(month_size, d_model), 'day_embed': Embed(day_size, d_model), 'weekday_embed': Embed(weekday_size, d_model), 'hour_embed': Embed(hour_size, d_model), 'minute_embed': Embed(minute_size, d_model)}
        for i in range(0, lowest_level):
            setattr(self, list(lowest_level_map.keys())[i], list(lowest_level_map.values())[i])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Creates the datetime embedding component.

        :param x: A PyTorch tensor of shape (batch_size, seq_len, n_feats).
        n_feats is formatted in the following manner.
        following way ()
        :type x: torch.Tensor
        :return: The datetime embedding of shape (batch_size, seq_len, 1)
        :rtype: torch.Tensor
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.0
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):

    def __init__(self, c_in: 'int', d_model, embed_type='fixed', data=4, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, lowest_level=data)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark) ->torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):

    def __init__(self, c_in, d_model: 'int', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark) ->torch.Tensor:
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_input: 'int', d_inner: 'int', n_heads: 'int'=4, dropout: 'float'=0.1, dropouta: 'float'=0.0):
        super().__init__()
        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.linear_kv = nn.Linear(d_input, d_inner * n_heads * 2, bias=False)
        self.linear_q = nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.linear_p = nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.scale = 1 / d_inner ** 0.5
        self.dropa = nn.Dropout(dropouta)
        self.lout = nn.Linear(self.d_inner * self.n_heads, self.d_input, bias=False)
        self.norm = nn.LayerNorm(self.d_input)
        self.dropo = nn.Dropout(dropout)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        return torch.cat([zero_pad, x], dim=1).view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:].view_as(x)

    def forward(self, input_: 'torch.FloatTensor', pos_embs: 'torch.FloatTensor', memory: 'torch.FloatTensor', u: 'torch.FloatTensor', v: 'torch.FloatTensor', mask: 'Optional[torch.FloatTensor]'=None):
        """
        pos_embs: we pass the positional embeddings in separately
            because we need to handle relative positions
        input shape: (seq, bs, self.d_input)
        pos_embs shape: (seq + prev_seq, bs, self.d_input)
        output shape: (seq, bs, self.d_input)
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner
        input_with_memory = torch.cat([memory, input_], dim=0)
        k_tfmd, v_tfmd = torch.chunk(self.linear_kv(input_with_memory), 2, dim=-1)
        q_tfmd = self.linear_q(input_)
        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]
        content_attn = torch.einsum('ibhd,jbhd->ijbh', (q_tfmd.view(cur_seq, bs, H, d) + u, k_tfmd.view(cur_seq + prev_seq, bs, H, d)))
        p_tfmd = self.linear_p(pos_embs)
        position_attn = torch.einsum('ibhd,jhd->ijbh', (q_tfmd.view(cur_seq, bs, H, d) + v, p_tfmd.view(cur_seq + prev_seq, H, d)))
        position_attn = self._rel_shift(position_attn)
        attn = content_attn + position_attn
        if mask is not None and mask.any().item():
            attn = attn.masked_fill(mask[..., None], -float('inf'))
        attn = torch.softmax(attn * self.scale, dim=1)
        attn = self.dropa(attn)
        attn_weighted_values = torch.einsum('ijbh,jbhd->ibhd', (attn, v_tfmd.view(cur_seq + prev_seq, bs, H, d))).contiguous().view(cur_seq, bs, H * d)
        output = input_ + self.dropo(self.lout(attn_weighted_values))
        output = self.norm(output)
        return output


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module Taken from DSANET repos."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(self, window, n_multiv, n_kernels, w_kernel, d_k, d_v, d_model, d_inner, n_layers, n_head, drop_prob=0.1):
        """
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """
        super(Single_Global_SelfAttn_Module, self).__init__()
        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob) for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(self, window, local, n_multiv, n_kernels, w_kernel, d_k, d_v, d_model, d_inner, n_layers, n_head, drop_prob=0.1):
        """
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """
        super(Single_Local_SelfAttn_Module, self).__init__()
        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob) for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(nn.Module):

    def __init__(self, forecast_history, n_time_series, dsa_local, dsanet_n_kernels, dsanet_w_kernals, dsanet_d_model, dsanet_d_inner, dsanet_n_layers=2, dropout=0.1, dsanet_n_head=8, dsa_targs=0):
        super(DSANet, self).__init__()
        self.window = forecast_history
        self.local = dsa_local
        self.n_multiv = n_time_series
        self.n_kernels = dsanet_n_kernels
        self.w_kernel = dsanet_w_kernals
        dsanet_d_k = int(dsanet_d_model / dsanet_n_head)
        dsanet_d_v = int(dsanet_d_model / dsanet_n_head)
        self.d_model = dsanet_d_model
        self.d_inner = dsanet_d_inner
        self.n_layers = dsanet_n_layers
        self.n_head = dsanet_n_head
        self.d_k = dsanet_d_k
        self.d_v = dsanet_d_v
        self.drop_prob = dropout
        self.n_targets = dsa_targs
        self.__build_model()

    def __build_model(self):
        """Layout model."""
        self.sgsf = Single_Global_SelfAttn_Module(window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels, w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.slsf = Single_Local_SelfAttn_Module(window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels, w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x):
        """No special modification required for lightning, define as you normally would."""
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        sf_output = torch.transpose(sf_output, 1, 2)
        ar_output = self.ar(x)
        output = sf_output + ar_output
        if self.n_targets > 0:
            return output[:, :, -self.n_targets]
        return output


class DummyTorchModel(nn.Module):

    def __init__(self, forecast_length: 'int') ->None:
        """A dummy model that will return a tensor of ones (batch_size, forecast_len).

        :param forecast_length: The length to forecast
        :type forecast_length: int
        """
        super(DummyTorchModel, self).__init__()
        self.out_len = forecast_length
        self.linear_test_layer = nn.Linear(3, 10)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """The forward pass for the dummy model.

        :param x: Here the data is irrelvant. Only batch_size is grabbed
        :type x: torch.Tensor
        :param mask: [description], defaults to None
        :type mask: torch.Tensor, optional
        :return: A tensor with fixed data of one
        :rtype: torch.Tensor
        """
        batch_sz = x.size(0)
        result = torch.ones(batch_sz, self.out_len, requires_grad=True, device=x.device)
        return result


class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: 'torch.Tensor'):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Informer(nn.Module):

    def __init__(self, n_time_series: 'int', dec_in: 'int', c_out: 'int', seq_len, label_len, out_len, factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, dropout=0.0, attn='prob', embed='fixed', temp_depth=4, activation='gelu', device=torch.device('cuda:0')):
        """This is based on the implementation of the Informer available from the original authors
        https://github.com/zhouhaoyi/Informer2020. We have done some minimal refactoring, but the core code remains the
        same. Additionally, we have added a few more options to the code.

        :param n_time_series: The number of time series present in the multivariate forecasting problem.
        :type n_time_series: int
        :param dec_in: The input size to the decoder (e.g. the number of time series passed to the decoder)
        :type dec_in: int
        :param c_out: The output dimension of the model (usually will be the number of variables you are forecasting).
        :type c_out:  int
        :param seq_len: The number of historical time steps to pass into the model.
        :type seq_len: int
        :param label_len: The length of the label sequence passed into the decoder (n_time_steps not used forecasted)
        :type label_len: int
        :param out_len: The predicted number of time steps.
        :type out_len: int
        :param factor: The multiplicative factor in the probablistic attention mechanism, defaults to 5
        :type factor: int, optional
        :param d_model: The embedding dimension of the model, defaults to 512
        :type d_model: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism , defaults to 8
        :type n_heads: int, optional
        :param e_layers: The number of layers in the encoder, defaults to 3
        :type e_layers: int, optional
        :param d_layers: The number of layers in the decoder, defaults to 2
        :type d_layers: int, optional
        :param d_ff: The dimension of the forward pass, defaults to 512
        :type d_ff: int, optional
        :param dropout: Whether to use dropout, defaults to 0.0
        :type dropout: float, optional
        :param attn: The type of the attention mechanism either 'prob' or 'full', defaults to 'prob'
        :type attn: str, optional
        :param embed: Whether to use class: `FixedEmbedding` or `torch.nn.Embbeding` , defaults to 'fixed'
        :type embed: str, optional
        :param temp_depth: The temporal depth (e.g year, month, day, weekday, etc), defaults to 4
        :type data: int, optional
        :param activation: The activation function, defaults to 'gelu'
        :type activation: str, optional
        :param device: The device the model uses, defaults to torch.device('cuda:0')
        :type device: str, optional
        """
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.c_out = c_out
        self.enc_embedding = DataEmbedding(n_time_series, d_model, embed, temp_depth, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, temp_depth, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder([EncoderLayer(AttentionLayer(Attn(False, factor, attention_dropout=dropout), d_model, n_heads), d_model, d_ff, dropout=dropout, activation=activation) for b in range(e_layers)], [ConvLayer(d_model) for b in range(e_layers - 1)], norm_layer=torch.nn.LayerNorm(d_model))
        self.decoder = Decoder([DecoderLayer(AttentionLayer(Attn(True, factor, attention_dropout=dropout), d_model, n_heads), AttentionLayer(FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads), d_model, d_ff, dropout=dropout, activation=activation) for c in range(d_layers)], norm_layer=torch.nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: 'torch.Tensor', x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """

        :param x_enc: The core tensor going into the model. Of dimension (batch_size, seq_len, n_time_series)
        :type x_enc: torch.Tensor
        :param x_mark_enc: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_enc: torch.Tensor
        :param x_dec: The datetime tensor information. Has dimension batch_size, seq_len, n_time_series
        :type x_dec: torch.Tensor
        :param x_mark_dec: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_dec: torch.Tensor
        :param enc_self_mask: The mask of the encoder model has size (), defaults to None
        :type enc_self_mask: [type], optional
        :param dec_self_mask: [description], defaults to None
        :type dec_self_mask: [type], optional
        :param dec_enc_mask: torch.Tensor, defaults to None
        :type dec_enc_mask: torch.Tensor, optional
        :return: Returns a PyTorch tensor of shape (batch_size, out_len, n_targets)
        :rtype: torch.Tensor
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]


class ITransformer(nn.Module):
    """Paper link: https://arxiv.org/abs/2310.06625."""

    def __init__(self, forecast_history, forecast_length, d_model, embed, dropout, n_heads=8, use_norm=True, e_layers=3, d_ff=512, freq='h', activation='gelu', factor=1, output_attention=True, targs=1):
        """The complete iTransformer model.

        :param forecast_history: The number of historical steps to use for forecasting
        :type forecast_history: int
        :param forecast_length: The length of the forecast the model outputs.
        :type forecast_length: int
        :param d_model: The embedding dimension of the model. For the paper the authors used 512.
        :type d_model: int
        :param embed: THe embedding type to use. For the paper the authors used 'fixed'.
        :type embed: str
        :param dropout: The dropout for the model.
        :type dropout: float
        :param n_heads: Number of heads for the attention, defaults to 8
        :type n_heads: int, optional
        :param use_norm: Whether to use normalization, defaults to True
        :type use_norm: bool, optional
        :param e_layers: The number of embedding layers, defaults to 3
        :type e_layers: int, optional
        :param d_ff: _description_, defaults to 512
        :type d_ff: int, optional
        :param freq: The frequency of the time series data, defaults to 'h' for hourly
        :type freq: str, optional
        :param activation: The activation, defaults to 'gelu'
        :type activation: str, optional
        :param factor: =n_, defaults to 1
        :type factor: int, optional
        :param output_attention: Whether to output the scores, defaults to True
        :type output_attention: bool, optional
        """
        class_strategy = 'projection'
        super(ITransformer, self).__init__()
        self.seq_len = forecast_history
        self.pred_len = forecast_length
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, embed, freq, dropout)
        self.class_strategy = class_strategy
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads), d_model, d_ff, dropout=dropout, activation=activation) for el in range(e_layers)], norm_layer=torch.nn.LayerNorm(d_model))
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.c_out = targs

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """_summary_

        :param x_enc: _description_
        :type x_enc: _type_
        :param x_mark_enc: _description_
        :type x_mark_enc: _type_
        :param x_dec: _description_
        :type x_dec: _type_
        :param x_mark_dec: _description_
        :type x_mark_dec: _type_
        :return: _description_
        :rtype: _type_
        """
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-05)
            x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out[0]).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """_summary_

        :param x_enc: _description_
        :type x_enc: _type_
        :param x_mark_enc: _description_
        :type x_mark_enc: _type_
        :param x_dec: _description_
        :type x_dec: _type_
        :param x_mark_dec: _description_
        :type x_mark_dec: _type_
        :param mask: _description_, defaults to None
        :type mask: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class SimplePositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Creates a basic positional encoding."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiAttnHeadSimple(torch.nn.Module):
    """A simple multi-head attention model inspired by Vaswani et al."""

    def __init__(self, number_time_series: 'int', seq_len=10, output_seq_len=None, d_model=128, num_heads=8, dropout=0.1, output_dim=1, final_layer=False):
        super().__init__()
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.final_layer = torch.nn.Linear(d_model, output_dim)
        self.length_data = seq_len
        self.forecast_length = output_seq_len
        self.sigmoid = None
        self.output_dim = output_dim
        if self.forecast_length:
            self.last_layer = torch.nn.Linear(seq_len, output_seq_len)
        if final_layer:
            self.sigmoid = activation_dict[final_layer]()

    def forward(self, x: 'torch.Tensor', mask=None) ->torch.Tensor:
        """
        :param: x torch.Tensor: of shape (B, L, M)
        Where B is the batch size, L is the sequence length and M is the number of time
        :return: a tensor of dimension (B, forecast_length)
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        if mask is None:
            x = self.multi_attn(x, x, x)[0]
        else:
            x = self.multi_attn(x, x, x, attn_mask=self.mask)[0]
        x = self.final_layer(x)
        if self.forecast_length:
            x = x.permute(1, 2, 0)
            x = self.last_layer(x)
            if self.sigmoid:
                x = self.sigmoid(x)
                return x.permute(0, 2, 1)
            return x.view(-1, self.forecast_length)
        return x.view(-1, self.length_data)


def generate_square_subsequent_mask(sz: 'int') ->torch.Tensor:
    """Generates a square mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class SimpleTransformer(torch.nn.Module):

    def __init__(self, number_time_series: 'int', seq_length: 'int'=48, output_seq_len: 'int'=None, d_model: 'int'=128, n_heads: 'int'=8, dropout=0.1, forward_dim=2048, sigmoid=False):
        """A full transformer model.

        :param number_time_series: The total number of time series present
            (e.g. n_feature_time_series + n_targets)
        :type number_time_series: int
        :param seq_length: The length of your input sequence, defaults to 48
        :type seq_length: int, optional
        :param output_seq_len: The length of your output sequence, defaults
            to None
        :type output_seq_len: int, optional
        :param d_model: The dimensions of your model, defaults to 128
        :type d_model: int, optional
        :param n_heads: The number of heads in each encoder/decoder block,
            defaults to 8
        :type n_heads: int, optional
        :param dropout: The fraction of dropout you wish to apply during
            training, defaults to 0.1 (currently not functional)
        :type dropout: float, optional
        :param forward_dim: Currently not functional, defaults to 2048
        :type forward_dim: int, optional
        :param sigmoid: Whether to apply a sigmoid activation to the final
            layer (useful for binary classification), defaults to False
        :type sigmoid: bool, optional
        """
        super().__init__()
        if output_seq_len is None:
            output_seq_len = seq_length
        self.out_seq_len = output_seq_len
        self.mask = generate_square_subsequent_mask(seq_length)
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(d_model, nhead=n_heads)
        self.final_layer = torch.nn.Linear(d_model, 1)
        self.sequence_size = seq_length
        self.tgt_mask = generate_square_subsequent_mask(output_seq_len)
        self.sigmoid = None
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', tgt_mask=None, src_mask=None):
        x = self.encode_sequence(x[:, :-1, :], src_mask)
        return self.decode_seq(x, t, tgt_mask)

    def basic_feature(self, x: 'torch.Tensor'):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def encode_sequence(self, x, src_mask=None):
        x = self.basic_feature(x)
        x = self.transformer.encoder(x, src_mask)
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) ->torch.Tensor:
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
        t = self.basic_feature(t)
        x = self.transformer.decoder(t, mem, tgt_mask=tgt_mask)
        x = self.final_layer(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x.view(-1, view_number)


class CustomTransformerDecoder(torch.nn.Module):

    def __init__(self, seq_length: 'int', output_seq_length: 'int', n_time_series: 'int', d_model=128, output_dim=1, n_layers_encoder=6, forward_dim=2048, dropout=0.1, use_mask=False, meta_data=None, final_act=None, squashed_embedding=False, n_heads=8):
        """Uses a number of encoder layers with simple linear decoder layer.

        :param seq_length: The number of historical time-steps fed into the model in each forward pass.
        :type seq_length: int
        :param output_seq_length: The number of forecasted time-steps outputted by the model.
        :type output_seq_length: int
        :param n_time_series: The total number of time series present (targets + features)
        :type n_time_series: int
        :param d_model: The embedding dim of the mode, defaults to 128
        :type d_model: int, optional
        :param output_dim: The output dimension (should correspond to n_targets), defaults to 1
        :type output_dim: int, optional
        :param n_layers_encoder: The number of encoder layers, defaults to 6
        :type n_layers_encoder: int, optional
        :param forward_dim: The forward embedding dim, defaults to 2048
        :type forward_dim: int, optional
        :param dropout: How much dropout to use, defaults to 0.1
        :type dropout: float, optional
        :param use_mask: Whether to use subsquent sequence mask during training, defaults to False
        :type use_mask: bool, optional
        :param meta_data: Whether to use static meta-data, defaults to None
        :type meta_data: str, optional
        :param final_act: Whether to use a final activation function, defaults to None
        :type final_act: str, optional
        :param squashed_embedding: Whether to create a one 1-D time embedding, defaults to False
        :type squashed_embedding: bool, optional
        :param n_heads: [description], defaults to 8
        :type n_heads: int, optional
        """
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8, forward_dim, dropout)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, n_layers_encoder, encoder_norm)
        self.output_dim_layer = torch.nn.Linear(d_model, output_dim)
        self.output_seq_length = output_seq_length
        self.out_length_lay = torch.nn.Linear(seq_length, output_seq_length)
        self.mask = generate_square_subsequent_mask(seq_length)
        self.out_dim = output_dim
        self.mask_it = use_mask
        self.final_act = None
        self.squashed = None
        if final_act:
            self.final_act = activation_dict[final_act]
        if meta_data:
            self.meta_merger = MergingModel(meta_data['method'], meta_data['params'])
        if squashed_embedding:
            self.squashed = torch.nn.Linear(seq_length, 1)
            self.unsquashed = torch.nn.Linear(1, seq_length)

    def make_embedding(self, x: 'torch.Tensor'):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            x = self.transformer_enc(x)
        if self.squashed:
            x = x.permute(1, 2, 0)
            x = self.squashed(x)
        return x

    def __squashed__embedding(self, x: 'torch.Tensor'):
        x = x.permute(1, 2, 0)
        x = self.squashed(x)
        x = self.unsquashed(x)
        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x: 'torch.Tensor', meta_data=None) ->torch.Tensor:
        """Performs forward pass on tensor of (batch_size, sequence_length, n_time_series) Return tensor of dim
        (batch_size, output_seq_length)"""
        x = self.dense_shape(x)
        if type(meta_data) == torch.Tensor:
            x = self.meta_merger(x, meta_data)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            x = self.transformer_enc(x)
        if self.squashed:
            x = self.__squashed__embedding(x)
        x = self.output_dim_layer(x)
        x = x.permute(1, 2, 0)
        x = self.out_length_lay(x)
        if self.final_act:
            x = self.final_act(x)
        if self.out_dim > 1:
            return x.permute(0, 2, 1)
        return x.view(-1, self.output_seq_length)


class LayerNorm(nn.Module):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""

    def __init__(self, n_embd, e=1e-05):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT_FNS = {'relu': nn.ReLU(), 'swish': swish, 'gelu': gelu}


class MLP(nn.Module):

    def __init__(self, n_state, n_embd, acf='relu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):

    def __init__(self, n_head, win_len, n_embd, scale, q_len, sub_len, additional_params: 'Dict'):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, sub_len, **additional_params)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        ln1 = self.ln_1(x + attn)
        mlp = self.mlp(ln1)
        hidden = self.ln_2(ln1 + mlp)
        return hidden


class TransformerModel(nn.Module):
    """Transformer model."""

    def __init__(self, n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history: 'int', dropout: 'float', scale_att, q_len, additional_params: 'Dict', seq_num=None):
        super(TransformerModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = n_time_series
        self.n_head = n_head
        self.seq_num = None
        if seq_num:
            self.seq_num = seq_num
            self.id_embed = nn.Embedding(seq_num, n_embd)
            nn.init.normal_(self.id_embed.weight, std=0.02)
        self.n_embd = n_embd
        self.win_len = forecast_history
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        self.po_embed = nn.Embedding(forecast_history, n_embd)
        self.drop_em = nn.Dropout(dropout)
        block = Block(n_head, forecast_history, n_embd + n_time_series, scale=scale_att, q_len=q_len, sub_len=sub_len, additional_params=additional_params)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, series_id: 'int', x: 'torch.Tensor'):
        """Runs  forward pass of the DecoderTransformer model.

        :param series_id:   ID of the time series
        :type series_id: int
        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: [type]
        """
        batch_size = x.size(0)
        length = x.size(1)
        embedding_sum = torch.zeros(batch_size, length, self.n_embd)
        if self.seq_num:
            embedding_sum = torch.zeros(batch_size, length)
            embedding_sum = embedding_sum.fill_(series_id).type(torch.LongTensor)
            embedding_sum = self.id_embed(embedding_sum)
        position = torch.tensor(torch.arange(length), dtype=torch.long)
        po_embedding = self.po_embed(position)
        embedding_sum[:] = po_embedding
        x = torch.cat((x, embedding_sum), dim=2)
        for block in self.blocks:
            x = block(x)
        return x


class DecoderTransformer(nn.Module):

    def __init__(self, n_time_series: 'int', n_head: 'int', num_layer: 'int', n_embd: 'int', forecast_history: 'int', dropout: 'float', q_len: 'int', additional_params: 'Dict', activation='Softmax', forecast_length: 'int'=None, scale_att: 'bool'=False, seq_num1=None, sub_len=1, mu=None):
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            seq_num: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model.
            additional_params: Additional parameters used to initalize the attention model. Can inc
        """
        super(DecoderTransformer, self).__init__()
        self.transformer = TransformerModel(n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history, dropout, scale_att, q_len, additional_params, seq_num=seq_num1)
        self.softplus = nn.Softplus()
        self.mu = torch.nn.Linear(n_time_series + n_embd, 1, bias=True)
        self.sigma = torch.nn.Linear(n_time_series + n_embd, 1, bias=True)
        self._initialize_weights()
        self.mu_mode = mu
        self.forecast_len_layer = None
        if forecast_length:
            self.forecast_len_layer = nn.Linear(forecast_history, forecast_length)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: 'torch.Tensor', series_id: 'int'=None):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:
            Case 1: tensor of dimension (batch_size, forecast_length)
            Case 2: GLoss sigma and mu: tuple of ((batch_size, forecast_history, 1), (batch_size, forecast_history, 1))
        """
        h = self.transformer(series_id, x)
        mu = self.mu(h)
        sigma = self.sigma(h)
        if self.mu_mode:
            sigma = self.softplus(sigma)
            return mu, sigma
        if self.forecast_len_layer:
            sigma = sigma.permute(0, 2, 1)
            sigma = self.forecast_len_layer(sigma).permute(0, 2, 1)
        return sigma.squeeze(2)


class PositionwiseFF(nn.Module):

    def __init__(self, d_input, d_inner, dropout):
        super().__init__()
        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = nn.Sequential(nn.Linear(d_input, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_input), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, input_: 'torch.FloatTensor') ->torch.FloatTensor:
        ff_out = self.ff(input_)
        output = self.layer_norm(input_ + ff_out)
        return output


class DecoderBlock(nn.Module):

    def __init__(self, n_heads, d_input, d_head_inner, d_ff_inner, dropout, dropouta=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_input, d_head_inner, n_heads=n_heads, dropout=dropout, dropouta=dropouta)
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)

    def forward(self, input_: 'torch.FloatTensor', pos_embs: 'torch.FloatTensor', u: 'torch.FloatTensor', v: 'torch.FloatTensor', mask=None, mems=None):
        return self.ff(self.mha(input_, pos_embs, mems, u, v, mask=mask))


class StandardWordEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, div_val=1, sample_softmax=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = embedding_dim ** 0.5

    def forward(self, input_: 'torch.LongTensor'):
        return self.embedding(input_) * self.scale


class TransformerXL(torch.nn.Module):

    def __init__(self, num_embeddings, n_layers, n_heads, d_model, d_head_inner, d_ff_inner, dropout=0.1, dropouta=0.0, seq_len: 'int'=0, mem_len: 'int'=0):
        super().__init__()
        self.n_layers, self.n_heads, self.d_model, self.d_head_inner, self.d_ff_inner = n_layers, n_heads, d_model, d_head_inner, d_ff_inner
        self.word_embs = StandardWordEmbedding(num_embeddings, d_model)
        self.pos_embs = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderBlock(n_heads, d_model, d_head_inner=d_head_inner, d_ff_inner=d_ff_inner, dropout=dropout, dropouta=dropouta) for _ in range(n_layers)])
        self.output_projection = nn.Linear(d_model, num_embeddings)
        self.output_projection.weight = self.word_embs.embedding.weight
        self.loss_fn = nn.CrossEntropyLoss()
        self.seq_len, self.mem_len = seq_len, mem_len
        self.u, self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)), nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner))

    def init_memory(self, device=torch.device('cpu')) ->torch.FloatTensor:
        return [torch.empty(0, dtype=torch.float) for _ in range(self.n_layers + 1)]

    def update_memory(self, previous_memory: 'List[torch.FloatTensor]', hidden_states: 'List[torch.FloatTensor]'):
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def reset_length(self, seq_len, ext_len, mem_len):
        self.seq_len = seq_len
        self.mem_len = mem_len

    def forward(self, idxs: 'torch.LongTensor', target: 'torch.LongTensor', memory: 'Optional[List[torch.FloatTensor]]'=None) ->Dict[str, torch.Tensor]:
        if memory is None:
            memory: 'List[torch.FloatTensor]' = self.init_memory(idxs.device)
        assert len(memory) == len(self.layers) + 1
        cur_seq, bs = idxs.size()
        prev_seq = memory[0].size(0)
        dec_attn_mask = torch.triu(torch.ones((cur_seq, cur_seq + prev_seq)), diagonal=1 + prev_seq).byte()[..., None]
        word_embs = self.drop(self.word_embs(idxs))
        pos_idxs = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float)
        pos_embs = self.drop(self.pos_embs(pos_idxs))
        hidden_states = [word_embs]
        layer_out = word_embs
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(layer_out, pos_embs, self.u, self.v, mask=dec_attn_mask, mems=mem)
            hidden_states.append(layer_out)
        logits = self.output_projection(self.drop(layer_out))
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        new_memory = self.update_memory(memory, hidden_states)
        return {'loss': loss, 'logits': logits, 'memory': new_memory}


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AE,
     lambda: ([], {'input_shape': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AR,
     lambda: ([], {'window': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Attention,
     lambda: ([], {'n_head': 4, 'n_embd': 4, 'win_len': 4, 'scale': 1.0, 'q_len': 4, 'sub_len': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (AttentionLayer,
     lambda: ([], {'d_model': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (BinaryFocalLossWithLogits,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'c_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CustomTransformerDecoder,
     lambda: ([], {'seq_length': 4, 'output_seq_length': 4, 'n_time_series': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DLinear,
     lambda: ([], {'forecast_history': 4, 'forecast_length': 4, 'individual': 4, 'enc_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DataEmbedding_inverted,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DummyTorchModel,
     lambda: ([], {'forecast_length': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Entmax15,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FlowAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FocalLoss,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (FullAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LSTMForecast,
     lambda: ([], {'seq_length': 4, 'n_time_series': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'n_embd': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MAPELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'n_state': 4, 'n_embd': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MovingAvg,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NLinear,
     lambda: ([], {'forecast_history': 4, 'forecast_length': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NaiveBase,
     lambda: ([], {'seq_length': 4, 'n_time_series': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PenalizedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFF,
     lambda: ([], {'d_input': 4, 'd_inner': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProbAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (SegMerging,
     lambda: ([], {'d_model': 4, 'win_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeriesDecomp,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4])], {})),
    (SimpleLinearModel,
     lambda: ([], {'seq_length': 4, 'n_time_series': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SimplePositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sparsemax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TokenEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (VanillaGRU,
     lambda: ([], {'n_time_series': 4, 'hidden_dim': 4, 'num_layers': 1, 'n_target': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

