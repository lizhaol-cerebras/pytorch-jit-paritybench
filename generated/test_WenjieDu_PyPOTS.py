
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


from abc import ABC


from abc import abstractmethod


from typing import Optional


from typing import Union


from typing import Iterable


import torch


from torch.utils.tensorboard import SummaryWriter


import numpy as np


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


import inspect


from sklearn.cluster import KMeans


from scipy.stats import multivariate_normal


from sklearn.mixture import GaussianMixture


from typing import Tuple


from numpy import ndarray


from torch import Tensor


from torch.utils.data import Dataset


import warnings


from math import ceil


import copy


from sklearn.preprocessing import StandardScaler


import math


from typing import Callable


import torch.fft


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import pandas as pd


from torch.nn.modules import TransformerEncoderLayer


import torch.fft as fft


from scipy.fftpack import next_fast_len


from functools import partial


from typing import List


from scipy.special import eval_legendre


from torch import nn


from scipy import signal


from scipy import special as ss


from math import sqrt


from torch.nn import TransformerEncoderLayer


from torch.nn import TransformerEncoder


from typing import Any


from torch.nn import Linear


from torch.nn import init


from torch import einsum


from torch.cuda.amp import autocast


from functools import wraps


from functools import reduce


from torch.nn.utils import weight_norm


from torch.optim import Adadelta as torch_Adadelta


from torch.optim import Adagrad as torch_Adagrad


from torch.optim import Adam as torch_Adam


from torch.optim import AdamW as torch_AdamW


from torch.optim import Optimizer


from collections import Counter


from torch.optim import RMSprop as torch_RMSprop


from torch.optim import SGD as torch_SGD


import random


class FeatureRegression(nn.Module):

    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRU-D model.
    Please refer to the original paper :cite:`che2018GRUD` for more details.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def __init__(self, input_size: 'int', output_size: 'int', diag: 'bool'=False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self._reset_parameters()

    def _reset_parameters(self) ->None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: 'torch.Tensor') ->torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


def _check_inputs(predictions: 'Union[np.ndarray, torch.Tensor, list]', targets: 'Union[np.ndarray, torch.Tensor, list]', masks: 'Optional[Union[np.ndarray, torch.Tensor, list]]'=None, check_shape: 'bool'=True):
    assert isinstance(predictions, type(targets)), f'types of `predictions` and `targets` must match, but got`predictions`: {type(predictions)}, `target`: {type(targets)}'
    lib = np if isinstance(predictions, np.ndarray) else torch
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert prediction_shape == target_shape, f'shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}'
    assert not lib.isnan(predictions).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(targets).any(), "`targets` mustn't contain NaN values, but detected NaN in it"
    if masks is not None:
        assert isinstance(masks, type(targets)), f'types of `masks`, `predictions`, and `targets` must match, but got`masks`: {type(masks)}, `targets`: {type(targets)}'
        mask_shape = masks.shape
        assert mask_shape == target_shape, f'shape of `masks` must match `targets` shape, but got `mask`: {mask_shape} that is different from `targets`: {target_shape}'
        assert not lib.isnan(masks).any(), "`masks` mustn't contain NaN values, but detected NaN in it"
    return lib


def calc_mae(predictions: 'Union[np.ndarray, torch.Tensor]', targets: 'Union[np.ndarray, torch.Tensor]', masks: 'Optional[Union[np.ndarray, torch.Tensor]]'=None) ->Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = calc_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = calc_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    lib = _check_inputs(predictions, targets, masks)
    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.abs(predictions - targets))


class BackboneRITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rnn_cell :
        the LSTM cell to model temporal data

    temp_decay_h :
        the temporal decay module to decay RNN hidden state

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist_reg :
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg :
        the feature-regression module

    combining_weight :
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell


    """

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def forward(self, inputs: 'dict', direction: 'str') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from `inputs`.

        Returns
        -------
        imputed_data :
            Input data with missing parts imputed. Shape of [batch size, sequence length, feature number].

        estimations :
            Reconstructed data. Shape of [batch size, sequence length, feature number].

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        X = inputs[direction]['X']
        missing_mask = inputs[direction]['missing_mask']
        deltas = inputs[direction]['deltas']
        device = X.device
        hidden_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        cell_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        estimations = []
        reconstruction_loss = torch.tensor(0.0)
        for t in range(self.n_steps):
            x = X[:, t, :]
            m = missing_mask[:, t, :]
            d = deltas[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_states = hidden_states * gamma_h
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += calc_mae(x_h, x, m)
            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            reconstruction_loss += calc_mae(z_h, x, m)
            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))
            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += calc_mae(c_h, x, m)
            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))
            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))
        reconstruction_loss /= self.n_steps * 3
        reconstruction = torch.cat(estimations, dim=1)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        return imputed_data, reconstruction, hidden_states, reconstruction_loss


class BackboneBRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rits_f: RITS object
        the forward RITS model

    rits_b: RITS object
        the backward RITS model

    """

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.rits_f = BackboneRITS(n_steps, n_features, rnn_hidden_size)
        self.rits_b = BackboneRITS(n_steps, n_features, rnn_hidden_size)

    @staticmethod
    def _get_consistency_loss(pred_f: 'torch.Tensor', pred_b: 'torch.Tensor') ->torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 0.1
        return loss

    @staticmethod
    def _reverse(ret: 'Tuple') ->Tuple:
        """Reverse the array values on the time dimension in the given dictionary."""

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)
        collector = []
        for value in ret:
            collector.append(reverse_tensor(value))
        return tuple(collector)

    def forward(self, inputs: 'dict') ->Tuple[torch.Tensor, ...]:
        f_imputed_data, f_reconstruction, f_hidden_states, f_reconstruction_loss = self.rits_f(inputs, 'forward')
        b_imputed_data, b_reconstruction, b_hidden_states, b_reconstruction_loss = self._reverse(self.rits_b(inputs, 'backward'))
        imputed_data = (f_imputed_data + b_imputed_data) / 2
        consistency_loss = self._get_consistency_loss(f_imputed_data, b_imputed_data)
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss
        return imputed_data, f_reconstruction, b_reconstruction, f_hidden_states, b_hidden_states, consistency_loss, reconstruction_loss


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    """

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        imputed_data, f_reconstruction, b_reconstruction, f_hidden_states, b_hidden_states, consistency_loss, reconstruction_loss = self.model(inputs)
        results = {'imputed_data': imputed_data}
        if training:
            results['consistency_loss'] = consistency_loss
            results['reconstruction_loss'] = reconstruction_loss
            loss = consistency_loss + reconstruction_loss
            results['loss'] = loss
            results['reconstruction'] = (f_reconstruction + b_reconstruction) / 2
            results['f_reconstruction'] = f_reconstruction
            results['b_reconstruction'] = b_reconstruction
        return results


class Conv1dWithInit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1dWithInit, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class Decay(nn.Module):

    def __init__(self, input_size, output_size, diag=False):
        super(Decay, self).__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag == True:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Decay_obs(nn.Module):

    def __init__(self, input_size, output_size):
        super(Decay_obs, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, delta_diff):
        sign = torch.sign(delta_diff)
        weight_diff = self.linear(delta_diff)
        positive_part = F.relu(weight_diff)
        negative_part = F.relu(-weight_diff)
        weight_diff = positive_part + negative_part
        weight_diff = sign * weight_diff
        weight_diff = torch.tanh(weight_diff)
        weight = 0.5 * (1 - weight_diff)
        return weight


class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: 'int', n_positions: 'int'=1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (torch.arange(0, d_hid, 2).float() * -(torch.log(torch.tensor(10000)) / d_hid)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pos_table', pe)

    def forward(self, x: 'torch.Tensor', return_only_pos: 'bool'=False) ->torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, :x.size(1)].clone().detach()
        if return_only_pos:
            return pos_enc
        x_with_pos = x + pos_enc
        return x_with_pos


class TorchTransformerEncoder(nn.Module):

    def __init__(self, heads=8, layers=1, channels=64):
        super(TorchTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class BackboneCSAI(nn.Module):
    """
    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    medians_tensor :
        tensor of median values for features, used to adjust decayed observations

    temp_decay_h :
        the temporal decay module to decay the hidden state of the GRU

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist :
        the temporal-regression module that projects the GRU hidden state into the raw feature space

    feat_reg_v :
        the feature-regression module used for feature-based estimation

    weight_combine :
        the module that generates the weight to combine history regression and feature regression

    weighted_obs :
        the decay module that computes weighted decay based on observed data and deltas

    gru :
        the GRU cell that models temporal data for imputation

    pos_encoder :
        the positional encoding module that adds temporal information to the sequence data

    input_projection :
        the convolutional module used to project input features into a higher-dimensional space

    output_projection1 :
        the convolutional module used to project the output from the Transformer layer

    output_projection2 :
        the final convolutional module used to generate the hidden state from the time-layer's output

    time_layer :
        the Transformer encoder layer used to model complex temporal dependencies within the sequence

    device :
        the device (CPU/GPU) used for model computations

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    medians_df :
        dataframe of median values for each feature, optional

    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, step_channels, medians_df=None):
        super(BackboneCSAI, self).__init__()
        if medians_df is not None:
            self.medians_tensor = torch.tensor(list(medians_df.values())).float()
        else:
            self.medians_tensor = torch.zeros(n_features).float()
        self.n_steps = n_steps
        self.step_channels = step_channels
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag=False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag=True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.weighted_obs = Decay_obs(self.input_size, self.input_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.pos_encoder = PositionalEncoding(self.step_channels)
        self.input_projection = Conv1dWithInit(self.input_size, self.step_channels, 1)
        self.output_projection1 = Conv1dWithInit(self.step_channels, self.hidden_size, 1)
        self.output_projection2 = Conv1dWithInit(self.n_steps * 2, 1, 1)
        self.time_layer = TorchTransformerEncoder(channels=self.step_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, last_obs, h=None):
        [B, _, _] = x.shape
        medians = self.medians_tensor.unsqueeze(0).repeat(B, 1)
        decay_factor = self.weighted_obs(deltas - medians.unsqueeze(1))
        if h == None:
            data_last_obs = self.input_projection(last_obs.permute(0, 2, 1)).permute(0, 2, 1)
            data_decay_factor = self.input_projection(decay_factor.permute(0, 2, 1)).permute(0, 2, 1)
            data_last_obs = self.pos_encoder(data_last_obs.permute(1, 0, 2)).permute(1, 0, 2)
            data_decay_factor = self.pos_encoder(data_decay_factor.permute(1, 0, 2)).permute(1, 0, 2)
            data = torch.cat([data_last_obs, data_decay_factor], dim=1)
            data = self.time_layer(data)
            data = self.output_projection1(data.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.output_projection2(data).squeeze()
        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        reconstruction = []
        for t in range(self.n_steps):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            x_h = self.hist(h)
            x_r_t = m_t * x_t + (1 - m_t) * x_h
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h
            x_loss += calc_mae(x_comb_t, x_t, m_t)
            x_imp[:, t, :] = m_t * x_t + (1 - m_t) * x_comb_t
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)
            h = self.gru(input_t, h)
            Hiddens.append(h.unsqueeze(dim=1))
            reconstruction.append(x_comb_t.unsqueeze(dim=1))
        Hiddens = torch.cat(Hiddens, dim=1)
        return x_imp, reconstruction, h, x_loss


class BackboneBCSAI(nn.Module):

    def __init__(self, n_steps, n_features, rnn_hidden_size, step_channels, medians_df=None):
        super(BackboneBCSAI, self).__init__()
        self.model_f = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, medians_df)
        self.model_b = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, medians_df)

    def forward(self, xdata):
        x = xdata['forward']['X']
        m = xdata['forward']['missing_mask']
        d_f = xdata['forward']['deltas']
        last_obs_f = xdata['forward']['last_obs']
        x_b = xdata['backward']['X']
        m_b = xdata['backward']['missing_mask']
        d_b = xdata['backward']['deltas']
        last_obs_b = xdata['backward']['last_obs']
        f_imputed_data, f_reconstruction, f_hidden_states, f_reconstruction_loss = self.model_f(x, m, d_f, last_obs_f)
        b_imputed_data, b_reconstruction, b_hidden_states, b_reconstruction_loss = self.model_b(x_b, m_b, d_b, last_obs_b)
        x_imp = (f_imputed_data + b_imputed_data.flip(dims=[1])) / 2
        imputed_data = x * m + (1 - m) * x_imp
        consistency_loss = torch.abs(f_imputed_data - b_imputed_data.flip(dims=[1])).mean() * 0.1
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss
        return imputed_data, f_reconstruction, b_reconstruction, f_hidden_states, b_hidden_states, consistency_loss, reconstruction_loss


class _BCSAI(nn.Module):
    """
    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    intervals :
        time intervals between the observations, used for handling irregular time-series

    consistency_weight :
        weight assigned to the consistency loss during training

    imputation_weight :
        weight assigned to the reconstruction loss during training

    model :
        the underlying BackboneBCSAI model that handles forward and backward pass imputation

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    intervals :
        time intervals between observations

    consistency_weight :
        weight assigned to the consistency loss

    imputation_weight :
        weight assigned to the reconstruction loss

    Notes
    -----
    BCSAI is a bidirectional imputation model that uses forward and backward GRU cells to handle time-series data. It computes consistency and reconstruction losses to improve imputation accuracy. During training, the forward and backward reconstructions are combined, and losses are used to update the model. In evaluation mode, the model also outputs original data and indicating masks for further analysis.

    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, step_channels, consistency_weight, imputation_weight, intervals=None):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.step_channels = step_channels
        self.intervals = intervals
        self.consistency_weight = consistency_weight
        self.imputation_weight = imputation_weight
        self.model = BackboneBCSAI(n_steps, n_features, rnn_hidden_size, step_channels, intervals)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        imputed_data, f_reconstruction, b_reconstruction, f_hidden_states, b_hidden_states, consistency_loss, reconstruction_loss = self.model(inputs)
        results = {'imputed_data': imputed_data}
        if training:
            results['consistency_loss'] = consistency_loss
            results['reconstruction_loss'] = reconstruction_loss
            loss = self.consistency_weight * consistency_loss + self.imputation_weight * reconstruction_loss
            results['loss'] = loss
            results['f_reconstruction'] = f_reconstruction
            results['b_reconstruction'] = b_reconstruction
        if not training:
            results['X_ori'] = inputs['X_ori']
            results['indicating_mask'] = inputs['indicating_mask']
        return results


class BackboneGRUD(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_cell = nn.GRUCell(self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)

    def forward(self, X, missing_mask, deltas, empirical_mean, X_filledLOCF) ->Tuple[torch.Tensor, ...]:
        """Forward processing of GRU-D.

        Parameters
        ----------
        X:

        missing_mask:

        deltas:

        empirical_mean:

        X_filledLOCF:

        Returns
        -------
        classification_pred:

        logits:


        """
        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=X.device)
        representation_collector = []
        for t in range(self.n_steps):
            x = X[:, t, :]
            m = missing_mask[:, t, :]
            d = deltas[:, t, :]
            x_filledLOCF = X_filledLOCF[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h
            representation_collector.append(hidden_state)
            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(data_input, hidden_state)
        representation_collector = torch.stack(representation_collector, dim=1)
        return representation_collector, hidden_state


def calc_mse(predictions: 'Union[np.ndarray, torch.Tensor]', targets: 'Union[np.ndarray, torch.Tensor]', masks: 'Optional[Union[np.ndarray, torch.Tensor]]'=None) ->Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = calc_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = calc_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    lib = _check_inputs(predictions, targets, masks)
    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.square(predictions - targets))


class _GRUD(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.backbone = BackboneGRUD(n_steps, n_features, rnn_hidden_size)
        self.output_projection = nn.Linear(rnn_hidden_size, n_features)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        X = inputs['X']
        missing_mask = inputs['missing_mask']
        deltas = inputs['deltas']
        empirical_mean = inputs['empirical_mean']
        X_filledLOCF = inputs['X_filledLOCF']
        hidden_states, _ = self.backbone(X, missing_mask, deltas, empirical_mean, X_filledLOCF)
        reconstruction = self.output_projection(hidden_states)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            results['loss'] = calc_mse(reconstruction, X, missing_mask)
        return results


class _Raindrop(nn.Module):

    def __init__(self, n_features, n_layers, d_model, n_heads, d_ffn, n_classes, dropout=0.3, max_len=215, d_static=9, aggregation='mean', sensor_wise_mask=False, static=False):
        super().__init__()
        d_pe = 16
        self.aggregation = aggregation
        self.sensor_wise_mask = sensor_wise_mask
        self.backbone = BackboneRaindrop(n_features, n_layers, d_model, n_heads, d_ffn, n_classes, dropout, max_len, d_static, d_pe, aggregation, sensor_wise_mask, static)
        if static:
            d_final = d_model + n_features
        else:
            d_final = d_model + d_pe
        self.mlp_static = nn.Sequential(nn.Linear(d_final, d_final), nn.ReLU(), nn.Linear(d_final, n_classes))

    def forward(self, inputs, training=True):
        X, missing_mask, static, timestamps, lengths = inputs['X'], inputs['missing_mask'], inputs['static'], inputs['timestamps'], inputs['lengths']
        device = X.device
        batch_size = X.shape[1]
        representation, mask = self.backbone(X, timestamps, lengths)
        lengths2 = lengths.unsqueeze(1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if self.sensor_wise_mask:
            output = torch.zeros([batch_size, self.n_features, self.d_ob + 16], device=device)
            extended_missing_mask = missing_mask.view(-1, batch_size, self.n_features)
            for se in range(self.n_features):
                representation = representation.view(-1, batch_size, self.n_features, self.d_ob + 16)
                out = representation[:, :, se, :]
                l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (l_ + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.n_features * (self.d_ob + 16)])
        elif self.aggregation == 'mean':
            output = torch.sum(representation * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            raise RuntimeError
        if static is not None:
            emb = self.static_emb(static)
            output = torch.cat([output, emb], dim=1)
        logits = self.mlp_static(output)
        classification_pred = torch.softmax(logits, dim=1)
        results = {'classification_pred': classification_pred}
        if training:
            classification_loss = F.nll_loss(torch.log(classification_pred), inputs['label'])
            results['loss'] = classification_loss
        return results


class _YourNewModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Module
        self.submodule = nn.Module
        self.backbone = nn.Module

    def forward(self, inputs: 'dict') ->dict:
        output = self.backbone()
        loss = None
        results = {'loss': loss}
        return results


class CrliDecoder(nn.Module):

    def __init__(self, n_steps: 'int', d_input: 'int', d_output: 'int', fcn_output_dims: 'list'=None):
        super().__init__()
        self.n_steps = n_steps
        self.d_output = d_output
        if fcn_output_dims is None:
            fcn_output_dims = [d_input]
        self.fcn_output_dims = fcn_output_dims
        self.fcn = nn.ModuleList()
        for output_dim in fcn_output_dims:
            self.fcn.append(nn.Linear(d_input, output_dim))
            d_input = output_dim
        self.rnn_cell = nn.GRUCell(fcn_output_dims[-1], fcn_output_dims[-1])
        self.output_layer = nn.Linear(fcn_output_dims[-1], d_output)

    def forward(self, generator_fb_hidden_states: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        device = generator_fb_hidden_states.device
        bz, _ = generator_fb_hidden_states.shape
        fcn_latent = generator_fb_hidden_states
        for layer in self.fcn:
            fcn_latent = layer(fcn_latent)
        hidden_state = fcn_latent
        hidden_state_collector = torch.empty((bz, self.n_steps, self.fcn_output_dims[-1]), device=device)
        for i in range(self.n_steps):
            hidden_state = self.rnn_cell(hidden_state, hidden_state)
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.output_layer(hidden_state_collector)
        return reconstruction, fcn_latent


RNN_CELL = {'LSTM': nn.LSTMCell, 'GRU': nn.GRUCell}


class CrliDiscriminator(nn.Module):

    def __init__(self, cell_type: 'str', d_input: 'int'):
        super().__init__()
        self.cell_type = cell_type
        self.rnn_cell_module_list = nn.ModuleList([RNN_CELL[cell_type](d_input, 32), RNN_CELL[cell_type](32, 16), RNN_CELL[cell_type](16, 8), RNN_CELL[cell_type](8, 16), RNN_CELL[cell_type](16, 32)])
        self.output_layer = nn.Linear(32, d_input)

    def forward(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor', imputation_latent: 'torch.Tensor') ->torch.Tensor:
        imputed_X = X * missing_mask + imputation_latent * (1 - missing_mask)
        bz, n_steps, _ = imputed_X.shape
        device = imputed_X.device
        hidden_states = [torch.zeros((bz, 32), device=device), torch.zeros((bz, 16), device=device), torch.zeros((bz, 8), device=device), torch.zeros((bz, 16), device=device), torch.zeros((bz, 32), device=device)]
        hidden_state_collector = torch.empty((bz, n_steps, 32), device=device)
        if self.cell_type == 'LSTM':
            cell_states = [torch.zeros((bz, 32), device=device), torch.zeros((bz, 16), device=device), torch.zeros((bz, 8), device=device), torch.zeros((bz, 16), device=device), torch.zeros((bz, 32), device=device)]
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state, cell_state = rnn_cell(x, (hidden_states[i], cell_states[i]))
                    else:
                        hidden_state, cell_state = rnn_cell(hidden_states[i - 1], (hidden_states[i], cell_states[i]))
                    cell_states[i] = cell_state
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state
        elif self.cell_type == 'GRU':
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state = rnn_cell(x, hidden_states[i])
                    else:
                        hidden_state = rnn_cell(hidden_states[i - 1], hidden_states[i])
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state
        output_collector = self.output_layer(hidden_state_collector)
        return output_collector


class MultiRNNCell(nn.Module):

    def __init__(self, cell_type: 'str', n_layer: 'int', d_input: 'int', d_hidden: 'int'):
        super().__init__()
        self.cell_type = cell_type
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.model = nn.ModuleList()
        if cell_type in ['LSTM', 'GRU']:
            for i in range(n_layer):
                if i == 0:
                    self.model.append(RNN_CELL[cell_type](d_input, d_hidden))
                else:
                    self.model.append(RNN_CELL[cell_type](d_hidden, d_hidden))
        self.output_layer = nn.Linear(d_hidden, d_input)

    def forward(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        bz, n_steps, _ = X.shape
        device = X.device
        hidden_state = torch.zeros((bz, self.d_hidden), device=device)
        hidden_state_collector = torch.empty((bz, n_steps, self.d_hidden), device=device)
        output_collector = torch.empty((bz, n_steps, self.d_input), device=device)
        if self.cell_type == 'LSTM':
            cell_states = [torch.zeros((bz, self.d_hidden), device=device) for _ in range(self.n_layer)]
            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state, cell_state = self.model[i](imputed_x, (hidden_state, cell_states[i]))
                    else:
                        hidden_state, cell_state = self.model[i](hidden_state, (hidden_state, cell_states[i]))
                hidden_state_collector[:, step, :] = hidden_state
        elif self.cell_type == 'GRU':
            for step in range(n_steps):
                x = X[:, step, :]
                estimation = self.output_layer(hidden_state)
                output_collector[:, step] = estimation
                imputed_x = missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state = self.model[i](imputed_x, hidden_state)
                    else:
                        hidden_state = self.model[i](hidden_state, hidden_state)
                hidden_state_collector[:, step, :] = hidden_state
        output_collector = output_collector[:, 1:]
        estimation = self.output_layer(hidden_state).unsqueeze(1)
        output_collector = torch.concat([output_collector, estimation], dim=1)
        return output_collector, hidden_state


def reverse_tensor(tensor_: 'torch.Tensor') ->torch.Tensor:
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
    return tensor_.index_select(1, indices)


class CrliGenerator(nn.Module):

    def __init__(self, n_layers: 'int', n_features: 'int', d_hidden: 'int', cell_type: 'str'):
        super().__init__()
        self.f_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden)
        self.b_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden)

    def forward(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        f_outputs, f_final_hidden_state = self.f_rnn(X, missing_mask)
        b_outputs, b_final_hidden_state = self.b_rnn(X, missing_mask)
        b_outputs = reverse_tensor(b_outputs)
        imputation_latent = (f_outputs + b_outputs) / 2
        fb_final_hidden_states = torch.concat([f_final_hidden_state, b_final_hidden_state], dim=-1)
        return imputation_latent, fb_final_hidden_states


class BackboneCRLI(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_generator_layers: 'int', rnn_hidden_size: 'int', decoder_fcn_output_dims: 'Optional[list]', rnn_cell_type: 'str'='GRU'):
        super().__init__()
        self.generator = CrliGenerator(n_generator_layers, n_features, rnn_hidden_size, rnn_cell_type)
        self.discriminator = CrliDiscriminator(rnn_cell_type, n_features)
        self.decoder = CrliDecoder(n_steps, rnn_hidden_size * 2, n_features, decoder_fcn_output_dims)

    def forward(self, X, missing_mask) ->Tuple[torch.Tensor, ...]:
        imputation_latent, generator_fb_hidden_states = self.generator(X, missing_mask)
        discrimination = self.discriminator(X, missing_mask, imputation_latent)
        reconstruction, fcn_latent = self.decoder(generator_fb_hidden_states)
        return imputation_latent, discrimination, reconstruction, fcn_latent


class _CRLI(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_clusters: 'int', n_generator_layers: 'int', rnn_hidden_size: 'int', decoder_fcn_output_dims: 'Optional[list]', lambda_kmeans: 'float', rnn_cell_type: 'str'='GRU'):
        super().__init__()
        self.backbone = BackboneCRLI(n_steps, n_features, n_generator_layers, rnn_hidden_size, decoder_fcn_output_dims, rnn_cell_type)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        self.term_F = None
        self.counter_for_updating_F = 0
        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans

    def forward(self, inputs: 'dict', training_object: 'str'='generator', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        imputation_latent, discrimination, reconstruction, fcn_latent = self.backbone(X, missing_mask)
        results = {'imputation_latent': imputation_latent, 'discrimination': discrimination, 'reconstruction': reconstruction, 'fcn_latent': fcn_latent}
        if not training:
            return results
        if training_object == 'discriminator':
            l_D = F.binary_cross_entropy_with_logits(discrimination, missing_mask)
            results['discrimination_loss'] = l_D
        else:
            l_G = F.binary_cross_entropy_with_logits(discrimination, 1 - missing_mask, weight=1 - missing_mask)
            l_pre = calc_mse(imputation_latent, X, missing_mask)
            l_rec = calc_mse(reconstruction, X, missing_mask)
            HTH = torch.matmul(fcn_latent, fcn_latent.permute(1, 0))
            if self.counter_for_updating_F == 0 or self.counter_for_updating_F % 10 == 0:
                U, s, V = torch.linalg.svd(fcn_latent)
                self.term_F = U[:, :self.n_clusters]
            FTHTHF = torch.matmul(torch.matmul(self.term_F.permute(1, 0), HTH), self.term_F)
            l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)
            loss_gene = l_G + l_pre + l_rec + l_kmeans * self.lambda_kmeans
            results['generation_loss'] = loss_gene
        return results


class GMMLayer(nn.Module):

    def __init__(self, d_hidden: 'int', n_clusters: 'int'):
        super().__init__()
        self.mu_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.var_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.phi_c_unscaled = Parameter(torch.Tensor(n_clusters))

    def set_values(self, mu: 'torch.Tensor', var: 'torch.Tensor', phi: 'torch.Tensor') ->None:
        assert mu.shape == self.mu_c_unscaled.shape
        assert var.shape == self.var_c_unscaled.shape
        assert phi.shape == self.phi_c_unscaled.shape
        self.mu_c_unscaled = torch.nn.Parameter(mu)
        self.var_c_unscaled = torch.nn.Parameter(var)
        self.phi_c_unscaled = torch.nn.Parameter(phi)

    def forward(self) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_c = self.mu_c_unscaled
        var_c = F.softplus(self.var_c_unscaled)
        phi_c = torch.softmax(self.phi_c_unscaled, dim=0)
        return mu_c, var_c, phi_c


class ImplicitImputation(nn.Module):

    def __init__(self, d_input: 'int'):
        super().__init__()
        self.projection_layer = nn.Linear(d_input, d_input, bias=False)

    def forward(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->torch.Tensor:
        imputation = self.projection_layer(X)
        imputed_X = X * missing_mask + imputation * (1 - X)
        return imputed_X


class PeepholeLSTMCell(nn.LSTMCell):
    """
    Notes
    -----
    This implementation is adapted from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'=True):
        super().__init__(input_size, hidden_size, bias)
        self.weight_ch = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ch = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(hidden_size))
        self.reset_parameters()

    def forward(self, X: 'torch.Tensor', hx: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            zeros = torch.zeros(X.size(0), self.hidden_size, dtype=X.dtype, device=X.device)
            hx = zeros, zeros
        h, c = hx
        wx = F.linear(X, self.weight_ih, self.bias_ih)
        wh = F.linear(h, self.weight_hh, self.bias_hh)
        wc = F.linear(c, self.weight_ch, self.bias_ch)
        wxhc = wx + wh + torch.cat((wc[:, :2 * self.hidden_size], Variable(self.wc_blank).expand_as(h), wc[:, 2 * self.hidden_size:]), dim=1)
        i = torch.sigmoid(wxhc[:, :self.hidden_size])
        f = torch.sigmoid(wxhc[:, self.hidden_size:2 * self.hidden_size])
        g = torch.tanh(wxhc[:, 2 * self.hidden_size:3 * self.hidden_size])
        o = torch.sigmoid(wxhc[:, 3 * self.hidden_size:])
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class BackboneVaDER(nn.Module):
    """

    Parameters
    ----------
    n_steps :
    d_input :
    n_clusters :
    d_rnn_hidden :
    d_mu_stddev :
    eps :
    alpha :
        Weight of the latent loss.
        The final loss = `alpha`*latent loss + reconstruction loss


    Attributes
    ----------

    """

    def __init__(self, n_steps: 'int', d_input: 'int', n_clusters: 'int', d_rnn_hidden: 'int', d_mu_stddev: 'int', eps: 'float'=1e-09, alpha: 'float'=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.d_input = d_input
        self.n_clusters = n_clusters
        self.d_rnn_hidden = d_rnn_hidden
        self.d_mu_stddev = d_mu_stddev
        self.eps = eps
        self.alpha = alpha
        self.implicit_imputation_layer = ImplicitImputation(d_input)
        self.encoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.decoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.ae_encode_layers = nn.Sequential(nn.Linear(d_rnn_hidden, d_rnn_hidden), nn.Softplus())
        self.ae_decode_layers = nn.Sequential(nn.Linear(d_mu_stddev, d_rnn_hidden), nn.Softplus())
        self.mu_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)
        self.stddev_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)
        self.rnn_transform_layer = nn.Linear(d_rnn_hidden, d_input)
        self.gmm_layer = GMMLayer(d_mu_stddev, n_clusters)

    @staticmethod
    def z_sampling(mu_tilde: 'torch.Tensor', stddev_tilde: 'torch.Tensor') ->torch.Tensor:
        noise = mu_tilde.data.new(mu_tilde.size()).normal_()
        z = torch.add(mu_tilde, torch.exp(0.5 * stddev_tilde) * noise)
        return z

    def encode(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = X.size(0)
        X_imputed = self.implicit_imputation_layer(X, missing_mask)
        hidden_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        cell_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        for i in range(self.n_steps):
            x = X_imputed[:, i, :]
            hidden_state, cell_state = self.encoder(x, (hidden_state, cell_state))
        cell_state_collector = self.ae_encode_layers(cell_state)
        mu_tilde = self.mu_layer(cell_state_collector)
        stddev_tilde = self.stddev_layer(cell_state_collector)
        z = self.z_sampling(mu_tilde, stddev_tilde)
        return z, mu_tilde, stddev_tilde

    def decode(self, z: 'torch.Tensor') ->torch.Tensor:
        hidden_state = z
        hidden_state = self.ae_decode_layers(hidden_state)
        cell_state = torch.zeros(hidden_state.size(), dtype=z.dtype, device=z.device)
        inputs = torch.zeros((z.size(0), self.n_steps, self.d_input), dtype=z.dtype, device=z.device)
        hidden_state_collector = torch.empty((z.size(0), self.n_steps, self.d_rnn_hidden), dtype=z.dtype, device=z.device)
        for i in range(self.n_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.decoder(x, (hidden_state, cell_state))
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.rnn_transform_layer(hidden_state_collector)
        return reconstruction

    def forward(self, X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu_tilde, stddev_tilde = self.encode(X, missing_mask)
        X_reconstructed = self.decode(z)
        mu_c, var_c, phi_c = self.gmm_layer()
        return X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde


class _VaDER(nn.Module):
    """

    Parameters
    ----------
    n_steps :
    d_input :
    n_clusters :
    d_rnn_hidden :
    d_mu_stddev :
    eps :
    alpha :
        Weight of the latent loss.
        The final loss = `alpha`*latent loss + reconstruction loss


    Attributes
    ----------

    """

    def __init__(self, n_steps: 'int', d_input: 'int', n_clusters: 'int', d_rnn_hidden: 'int', d_mu_stddev: 'int', eps: 'float'=1e-09, alpha: 'float'=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.d_input = d_input
        self.n_clusters = n_clusters
        self.d_rnn_hidden = d_rnn_hidden
        self.d_mu_stddev = d_mu_stddev
        self.eps = eps
        self.alpha = alpha
        self.backbone = BackboneVaDER(n_steps, d_input, n_clusters, d_rnn_hidden, d_mu_stddev, eps, alpha)

    def forward(self, inputs: 'dict', pretrain: 'bool'=False, training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        device = X.device
        X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde = self.backbone(X, missing_mask)
        results = {'mu_tilde': mu_tilde, 'stddev_tilde': stddev_tilde, 'mu': mu_c, 'var': var_c, 'phi': phi_c, 'z': z, 'imputation_latent': X_reconstructed}
        unscaled_reconstruction_loss = calc_mse(X_reconstructed, X, missing_mask)
        reconstruction_loss = unscaled_reconstruction_loss * self.n_steps * self.d_input / missing_mask.sum()
        if pretrain:
            results['loss'] = reconstruction_loss
            return results
        if training:
            var_tilde = torch.exp(stddev_tilde)
            stddev_c = torch.log(var_c + self.eps)
            log_2pi = torch.log(torch.tensor([2 * torch.pi], device=device))
            log_phi_c = torch.log(phi_c + self.eps)
            batch_size = z.shape[0]
            ii, jj = torch.meshgrid(torch.arange(self.n_clusters, dtype=torch.int64, device=device), torch.arange(batch_size, dtype=torch.int64, device=device), indexing='ij')
            ii = ii.flatten()
            jj = jj.flatten()
            lsc_b = stddev_c.index_select(dim=0, index=ii)
            mc_b = mu_c.index_select(dim=0, index=ii)
            sc_b = var_c.index_select(dim=0, index=ii)
            z_b = z.index_select(dim=0, index=jj)
            log_pdf_z = -0.5 * (lsc_b + log_2pi + torch.square(z_b - mc_b) / sc_b)
            log_pdf_z = log_pdf_z.reshape([batch_size, self.n_clusters, self.d_mu_stddev])
            log_p = log_phi_c + log_pdf_z.sum(dim=2)
            lse_p = log_p.logsumexp(dim=1, keepdim=True)
            log_gamma_c = log_p - lse_p
            gamma_c = torch.exp(log_gamma_c)
            term1 = torch.log(var_c + self.eps)
            st_b = var_tilde.index_select(dim=0, index=jj)
            sc_b = var_c.index_select(dim=0, index=ii)
            term2 = torch.reshape(st_b / (sc_b + self.eps), [batch_size, self.n_clusters, self.d_mu_stddev])
            mt_b = mu_tilde.index_select(dim=0, index=jj)
            mc_b = mu_c.index_select(dim=0, index=ii)
            term3 = torch.reshape(torch.square(mt_b - mc_b) / (sc_b + self.eps), [batch_size, self.n_clusters, self.d_mu_stddev])
            latent_loss1 = 0.5 * torch.sum(gamma_c * torch.sum(term1 + term2 + term3, dim=2), dim=1)
            latent_loss2 = -torch.sum(gamma_c * (log_phi_c - log_gamma_c), dim=1)
            latent_loss3 = -0.5 * torch.sum(1 + stddev_tilde, dim=1)
            latent_loss1 = latent_loss1.mean()
            latent_loss2 = latent_loss2.mean()
            latent_loss3 = latent_loss3.mean()
            latent_loss = latent_loss1 + latent_loss2 + latent_loss3
            results['loss'] = reconstruction_loss + self.alpha * latent_loss
        return results


class CsdiDiffusionEmbedding(nn.Module):

    def __init__(self, n_diffusion_steps, d_embedding=128, d_projection=None):
        super().__init__()
        if d_projection is None:
            d_projection = d_embedding
        self.register_buffer('embedding', self._build_embedding(n_diffusion_steps, d_embedding // 2), persistent=False)
        self.projection1 = nn.Linear(d_embedding, d_projection)
        self.projection2 = nn.Linear(d_projection, d_projection)

    @staticmethod
    def _build_embedding(n_steps, d_embedding=64):
        steps = torch.arange(n_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(d_embedding) / (d_embedding - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

    def forward(self, diffusion_step: 'int'):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation='gelu')
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class CsdiResidualBlock(nn.Module):

    def __init__(self, d_side, n_channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, n_channels)
        self.cond_projection = conv1d_with_init(d_side, 2 * n_channels, 1)
        self.mid_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.output_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CsdiDiffusionModel(nn.Module):

    def __init__(self, n_diffusion_steps, d_diffusion_embedding, d_input, d_side, n_channels, n_heads, n_layers):
        super().__init__()
        self.diffusion_embedding = CsdiDiffusionEmbedding(n_diffusion_steps=n_diffusion_steps, d_embedding=d_diffusion_embedding)
        self.input_projection = conv1d_with_init(d_input, n_channels, 1)
        self.output_projection1 = conv1d_with_init(n_channels, n_channels, 1)
        self.output_projection2 = conv1d_with_init(n_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList([CsdiResidualBlock(d_side=d_side, n_channels=n_channels, diffusion_embedding_dim=d_diffusion_embedding, nheads=n_heads) for _ in range(n_layers)])
        self.n_channels = n_channels

    def forward(self, x, cond_info, diffusion_step):
        n_samples, input_dim, n_features, n_steps = x.shape
        x = x.reshape(n_samples, input_dim, n_features * n_steps)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(n_samples, self.n_channels, n_features, n_steps)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(n_samples, self.n_channels, n_features * n_steps)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(n_samples, n_features, n_steps)
        return x


class BackboneCSDI(nn.Module):

    def __init__(self, n_layers, n_heads, n_channels, d_target, d_time_embedding, d_feature_embedding, d_diffusion_embedding, is_unconditional, n_diffusion_steps, schedule, beta_start, beta_end):
        super().__init__()
        self.d_target = d_target
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.is_unconditional = is_unconditional
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps
        d_side = d_time_embedding + d_feature_embedding
        if self.is_unconditional:
            d_input = 1
        else:
            d_side += 1
            d_input = 2
        self.diff_model = CsdiDiffusionModel(n_diffusion_steps, d_diffusion_embedding, d_input, d_side, n_channels, n_heads, n_layers)
        if schedule == 'quad':
            self.beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, self.n_diffusion_steps) ** 2
        elif schedule == 'linear':
            self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
        else:
            raise ValueError(f"The argument schedule should be 'quad' or 'linear', but got {schedule}")
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer('alpha_torch', torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1))

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    def calc_loss_valid(self, observed_data, cond_mask, indicating_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.n_diffusion_steps):
            loss = self.calc_loss(observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.n_diffusion_steps

    def calc_loss(self, observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        device = observed_data.device
        if is_train != 1:
            t = (torch.ones(B) * set_t).long()
        else:
            t = torch.randint(0, self.n_diffusion_steps, [B])
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = current_alpha ** 0.5 * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diff_model(total_input, side_info, t)
        target_mask = indicating_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def forward(self, observed_data, cond_mask, side_info, n_sampling_times):
        B, K, L = observed_data.shape
        device = observed_data.device
        imputed_samples = torch.zeros(B, n_sampling_times, K, L)
        for i in range(n_sampling_times):
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.n_diffusion_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = self.alpha_hat[t] ** 0.5 * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            current_sample = torch.randn_like(observed_data)
            for t in range(self.n_diffusion_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                predicted = self.diff_model(diff_input, side_info, torch.tensor([t]))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


class _CSDI(nn.Module):

    def __init__(self, n_features, n_layers, n_heads, n_channels, d_time_embedding, d_feature_embedding, d_diffusion_embedding, is_unconditional, n_diffusion_steps, schedule, beta_start, beta_end):
        super().__init__()
        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.is_unconditional = is_unconditional
        self.embed_layer = nn.Embedding(num_embeddings=n_features, embedding_dim=d_feature_embedding)
        self.backbone = BackboneCSDI(n_layers, n_heads, n_channels, n_features, d_time_embedding, d_feature_embedding, d_diffusion_embedding, is_unconditional, n_diffusion_steps, schedule, beta_start, beta_end)

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)
        time_embed = time_embed
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.n_features))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def forward(self, inputs, training=True, n_sampling_times=1):
        results = {}
        if training:
            observed_data, indicating_mask, cond_mask, observed_tp = inputs['X_ori'], inputs['indicating_mask'], inputs['cond_mask'], inputs['observed_tp']
            side_info = self.get_side_info(observed_tp, cond_mask)
            training_loss = self.backbone.calc_loss(observed_data, cond_mask, indicating_mask, side_info, training)
            results['loss'] = training_loss
        elif not training and n_sampling_times == 0:
            observed_data, indicating_mask, cond_mask, observed_tp = inputs['X_ori'], inputs['indicating_mask'], inputs['cond_mask'], inputs['observed_tp']
            side_info = self.get_side_info(observed_tp, cond_mask)
            validating_loss = self.backbone.calc_loss_valid(observed_data, cond_mask, indicating_mask, side_info, training)
            results['loss'] = validating_loss
        elif not training and n_sampling_times > 0:
            observed_data, cond_mask, observed_tp = inputs['X'], inputs['cond_mask'], inputs['observed_tp']
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.backbone(observed_data, cond_mask, side_info, n_sampling_times)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            imputed_data = repeated_obs + samples * (1 - repeated_mask)
            results['imputed_data'] = imputed_data.permute(0, 1, 3, 2)
        return results


class AttentionOperator(nn.Module):
    """
    The abstract class for all attention layers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class AutoCorrelation(AttentionOperator):
    """
    AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery
        (2) time delay aggregation

    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, factor=1, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        B, L, H, E = q.shape
        _, S, _, D = v.shape
        if L > S:
            zeros = torch.zeros_like(q[:, :L - S, :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :L, :, :]
            k = k[:, :L, :, :]
        q_fft = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        if self.training:
            V = self.time_delay_agg_training(v.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(v.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        attn = corr.permute(0, 3, 1, 2)
        output = V.contiguous()
        return output, attn


class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention module.

    Parameters
    ----------
    attn_opt:
        The attention operator, e.g. the self-attention proposed in Transformer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    """

    def __init__(self, attn_opt: 'AttentionOperator', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int'):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.attention_operator = attn_opt
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]', **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the multi-head attention module.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        v:
            The output of the multi-head attention layer.

        attn_weights:
            The attention map.

        """
        batch_size, q_len = q.size(0), q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)
        q = self.w_qs(q).view(batch_size, q_len, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, k_len, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, v_len, self.n_heads, self.d_v)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        v, attn_weights = self.attention_operator(q, k, v, attn_mask, **kwargs)
        v = v.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        v = self.fc(v)
        return v, attn_weights


class MovingAvgBlock(nn.Module):
    """
    The moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompositionBlock(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvgBlock(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class AutoformerEncoderLayer(nn.Module):
    """Autoformer encoder layer with the progressive decomposition architecture."""

    def __init__(self, attn_opt: 'AttentionOperator', d_model: 'int', n_heads: 'int', d_ffn: 'int', moving_avg: 'int'=25, dropout: 'float'=0.1, activation='relu'):
        super().__init__()
        d_ffn = d_ffn or 4 * d_model
        self.attention = MultiHeadAttention(attn_opt, d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ffn, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ffn, out_channels=d_model, kernel_size=1, bias=False)
        self.series_decomp1 = SeriesDecompositionBlock(moving_avg)
        self.series_decomp2 = SeriesDecompositionBlock(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.series_decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.series_decomp2(x + y)
        return res, attn


class InformerEncoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class SeasonalLayerNorm(nn.Module):
    """A special designed layer normalization for the seasonal part."""

    def __init__(self, n_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_channels)

    def forward(self, x):
        x_hat = self.layer_norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoformerEncoder(nn.Module):

    def __init__(self, n_layers, d_model, n_heads, d_ffn, factor, moving_avg_window_size, dropout, activation='relu'):
        super().__init__()
        self.encoder = InformerEncoder([AutoformerEncoderLayer(AutoCorrelation(factor, dropout), d_model, n_heads, d_ffn, moving_avg_window_size, dropout, activation) for _ in range(n_layers)], norm_layer=SeasonalLayerNorm(d_model))

    def forward(self, x, attn_mask=None):
        enc_out, attns = self.encoder(x, attn_mask)
        return enc_out, attns


class SaitsEmbedding(nn.Module):
    """The embedding method from the SAITS paper :cite:`du2023SAITS`.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(self, d_in: 'int', d_out: 'int', with_pos: 'bool', n_max_steps: 'int'=1000, dropout: 'float'=0):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout
        self.embedding_layer = nn.Linear(d_in, d_out)
        self.position_enc = PositionalEncoding(d_out, n_positions=n_max_steps) if with_pos else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X, missing_mask=None):
        if missing_mask is not None:
            X = torch.cat([X, missing_mask], dim=2)
        X_embedding = self.embedding_layer(X)
        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)
        return X_embedding


class SaitsLoss(nn.Module):

    def __init__(self, ORT_weight, MIT_weight, loss_calc_func: 'Callable'=calc_mae):
        super().__init__()
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.loss_calc_func = loss_calc_func

    def forward(self, reconstruction, X_ori, missing_mask, indicating_mask):
        ORT_loss = self.ORT_weight * self.loss_calc_func(reconstruction, X_ori, missing_mask)
        MIT_loss = self.MIT_weight * self.loss_calc_func(reconstruction, X_ori, indicating_mask)
        loss = ORT_loss + MIT_loss
        return loss, ORT_loss, MIT_loss


class _Autoformer(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, n_heads, d_ffn, factor, moving_avg_window_size, dropout, ORT_weight: 'float'=1, MIT_weight: 'float'=1, activation='relu'):
        super().__init__()
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False, dropout=dropout)
        self.encoder = AutoformerEncoder(n_layers, d_model, n_heads, d_ffn, factor, moving_avg_window_size, dropout, activation)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, attns = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class CrossformerEncoder(nn.Module):

    def __init__(self, attn_layers):
        super().__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x, src_mask=None):
        attn_weights_collector = []
        enc_output = x
        for block in self.encode_blocks:
            enc_output, attn_weights = block(enc_output, src_mask)
            attn_weights_collector.append(attn_weights)
        return enc_output, attn_weights_collector


class PatchEmbedding(nn.Module):

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)
        return x


class PredictionHead(nn.Module):

    def __init__(self, d_model, n_patches, n_steps_forecast, head_dropout=0, individual=False, n_features=0):
        super().__init__()
        head_dim = d_model * n_patches
        self.individual = individual
        self.n_features = n_features
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_features):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, n_steps_forecast))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, n_steps_forecast)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_features):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.linear(x)
        return x.transpose(2, 1)


class SegMerging(nn.Module):

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
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


class ScaledDotProductAttention(AttentionOperator):
    """Scaled dot-product attention.

    Parameters
    ----------
    temperature:
        The temperature for scaling.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, temperature: 'float', attn_dropout: 'float'=0.1):
        super().__init__()
        assert temperature > 0, 'temperature should be positive'
        assert attn_dropout >= 0, 'dropout rate should be non-negative'
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn:
            The scaled dot-product attention map.

        """
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1000000000.0)
        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, seg_num, factor, d_model, n_heads, d_k, d_v, d_ff=None, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.time_attention = MultiHeadAttention(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v)
        self.dim_sender = MultiHeadAttention(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v)
        self.dim_receiver = MultiHeadAttention(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch, ts_d, seg_num, d_model = x.shape
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out


class ScaleBlock(nn.Module):

    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num, factor):
        super().__init__()
        d_k = d_model // n_heads
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_k, d_k, d_ff, dropout))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x, None


class _Crossformer(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, n_heads, d_ffn, factor, seg_len, win_size, dropout, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.d_model = d_model
        pad_in_len = ceil(1.0 * n_steps / seg_len) * seg_len
        in_seg_num = pad_in_len // seg_len
        out_seg_num = ceil(in_seg_num / win_size ** (n_layers - 1))
        self.enc_value_embedding = PatchEmbedding(d_model, seg_len, seg_len, pad_in_len - n_steps, 0)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, d_model, in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = CrossformerEncoder([ScaleBlock(1 if layer == 0 else win_size, d_model, n_heads, d_ffn, 1, dropout, in_seg_num if layer == 0 else ceil(in_seg_num / win_size ** layer), factor) for layer in range(n_layers)])
        self.head = PredictionHead(d_model, out_seg_num, n_steps, dropout)
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        input_X = self.saits_embedding(X, missing_mask)
        x_enc = self.enc_value_embedding(input_X.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=self.d_model)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        reconstruction = self.output_projection(dec_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class BackboneDLinear(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', individual: 'bool'=False, d_model: 'Optional[int]'=None):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual
        if individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for i in range(n_features):
                self.linear_seasonal.append(nn.Linear(n_steps, n_steps))
                self.linear_trend.append(nn.Linear(n_steps, n_steps))
                self.linear_seasonal[i].weight = nn.Parameter(1 / n_steps * torch.ones([n_steps, n_steps]))
                self.linear_trend[i].weight = nn.Parameter(1 / n_steps * torch.ones([n_steps, n_steps]))
        else:
            if d_model is None:
                raise ValueError('The argument d_model is necessary for DLinear in the non-individual mode.')
            self.linear_seasonal = nn.Linear(n_steps, n_steps)
            self.linear_trend = nn.Linear(n_steps, n_steps)
            self.linear_seasonal.weight = nn.Parameter(1 / n_steps * torch.ones([n_steps, n_steps]))
            self.linear_trend.weight = nn.Parameter(1 / n_steps * torch.ones([n_steps, n_steps]))

    def forward(self, seasonal_init, trend_init):
        if self.individual:
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.n_steps], dtype=seasonal_init.dtype)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.n_steps], dtype=trend_init.dtype)
            for i in range(self.n_features):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        else:
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
        return seasonal_output, trend_output


class _DLinear(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', moving_avg_window_size: 'int', individual: 'bool'=False, d_model: 'Optional[int]'=None, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.individual = individual
        self.series_decomp = SeriesDecompositionBlock(moving_avg_window_size)
        self.backbone = BackboneDLinear(n_steps, n_features, individual, d_model)
        if not individual:
            self.seasonal_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.trend_saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
            self.linear_seasonal_output = nn.Linear(d_model, n_features)
            self.linear_trend_output = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        seasonal_init, trend_init = self.series_decomp(X)
        if not self.individual:
            seasonal_init = self.seasonal_saits_embedding(seasonal_init, missing_mask)
            trend_init = self.trend_saits_embedding(trend_init, missing_mask)
        seasonal_output, trend_output = self.backbone(seasonal_init, trend_init)
        if not self.individual:
            seasonal_output = self.linear_seasonal_output(seasonal_output)
            trend_output = self.linear_trend_output(trend_output)
        reconstruction = seasonal_output + trend_output
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class ETSformerDecoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.d_model = layers[0].d_model
        self.d_out = layers[0].d_out
        self.pred_len = layers[0].pred_len
        self.n_head = layers[0].n_heads
        self.layers = nn.ModuleList(layers)
        self.pred = nn.Linear(self.d_model, self.d_out)

    def forward(self, growths, seasons):
        growth_repr = []
        season_repr = []
        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)
        return self.pred(growth_repr), self.pred(season_repr)


class DampingLayer(nn.Module):

    def __init__(self, pred_len, n_heads, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_heads = n_heads
        self._damping_factor = nn.Parameter(torch.randn(1, n_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        b, t, d = x.shape
        powers = torch.arange(self.pred_len) + 1
        powers = powers.view(self.pred_len, 1)
        damping_factors = self.damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)
        x = x.view(b, t, self.n_heads, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)
        return x.view(b, t, d)

    @property
    def damping_factor(self):
        return torch.sigmoid(self._damping_factor)


class ETSformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_out, pred_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_out = d_out
        self.pred_len = pred_len
        self.growth_damping = DampingLayer(pred_len, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, growth, season):
        growth_horizon = self.growth_damping(growth[:, -1:])
        growth_horizon = self.dropout1(growth_horizon)
        seasonal_horizon = season[:, -self.pred_len:]
        return growth_horizon, seasonal_horizon


class ETSformerEncoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        for layer in self.layers:
            res, level, growth, season = layer(res, level, attn_mask=attn_mask)
            growths.append(growth)
            seasons.append(season)
        return level, growths, seasons


class Feedforward(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FourierLayer(nn.Module):

    def __init__(self, d_model, pred_len, k=None, low_freq=1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        device = x_freq.device
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = rearrange(f[index_tuple], 'b f d -> b f () d')
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float), 't -> () () t ()')
        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1)
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


def conv1d_fft(f, g, dim=-1):
    N = f.size(dim)
    M = g.size(dim)
    fast_len = next_fast_len(N + M - 1)
    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)
    F_fg = F_f * F_g.conj()
    out = fft.irfft(F_fg, fast_len, dim=dim)
    out = out.roll((-1,), dims=(dim,))
    idx = torch.as_tensor(range(fast_len - N, fast_len))
    out = out.index_select(dim, idx)
    return out


class ExponentialSmoothing(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1, aux=False):
        super().__init__()
        self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    def forward(self, values, aux_values=None):
        b, t, h, d = values.shape
        init_weight, weight = self.get_exponential_weight(t)
        output = conv1d_fft(self.dropout(values), weight, dim=1)
        output = init_weight * self.v0 + output
        if aux_values is not None:
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight)
            output = output + aux_output
        return output

    def get_exponential_weight(self, T):
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        weight = (1 - self.weight) * self.weight ** torch.flip(powers, dims=(0,))
        init_weight = self.weight ** (powers + 1)
        return rearrange(init_weight, 'h t -> 1 t h 1'), rearrange(weight, 'h t -> 1 t h 1')

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)


class GrowthLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_head=None, dropout=0.1):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.z0 = nn.Parameter(torch.randn(self.n_heads, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.n_heads)
        self.es = ExponentialSmoothing(self.d_head, self.n_heads, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.n_heads, self.d_model)
        assert self.d_head * self.n_heads == self.d_model, 'd_model must be divisible by n_heads'

    def forward(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        b, t, d = inputs.shape
        values = self.in_proj(inputs).view(b, t, self.n_heads, -1)
        values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)
        out = torch.cat([repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), out], dim=1)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return self.out_proj(out)


class LevelLayer(nn.Module):

    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out
        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level, growth, season):
        b, t, _ = level.shape
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season).view(b, t, self.c_out, 1)
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return out


class ETSformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_out, seq_len, pred_len, k, d_ffn=None, dropout=0.1, activation='sigmoid', layer_norm_eps=1e-05):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_out = d_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        d_ffn = d_ffn or 4 * d_model
        self.d_ffn = d_ffn
        self.growth_layer = GrowthLayer(d_model, n_heads, dropout=dropout)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k)
        self.level_layer = LevelLayer(d_model, d_out, dropout=dropout)
        self.ff = Feedforward(d_model, d_ffn, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res, level, attn_mask=None):
        season = self._season_block(res)
        res = res - season[:, :-self.pred_len]
        growth = self._growth_block(res)
        res = self.norm1(res - growth[:, 1:])
        res = self.norm2(res + self.ff(res))
        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])
        return res, level, growth, season

    def _growth_block(self, x):
        x = self.growth_layer(x)
        return self.dropout1(x)

    def _season_block(self, x):
        x = self.seasonal_layer(x)
        return self.dropout2(x)


class _ETSformer(nn.Module):

    def __init__(self, n_steps, n_features, n_e_layers, n_d_layers, d_model, n_heads, d_ffn, dropout, top_k, ORT_weight: 'float'=1, MIT_weight: 'float'=1, activation='sigmoid'):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.encoder = ETSformerEncoder([ETSformerEncoderLayer(d_model, n_heads, n_features, n_steps, n_steps, top_k, d_ffn=d_ffn, dropout=dropout, activation=activation) for _ in range(n_e_layers)])
        self.decoder = ETSformerDecoder([ETSformerDecoderLayer(d_model, n_heads, n_features, n_steps, dropout=dropout) for _ in range(n_d_layers)])
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        res = self.saits_embedding(X, missing_mask)
        level, growths, seasons = self.encoder(res, X, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        reconstruction = level[:, -1:] + growth + season
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(AttentionOperator):

    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super().__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain,
        it does FFT, linear transform, and Inverse FFT.
        """
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum('bhi,hio->bho', input, weights)

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, None]:
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x, None


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-08] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-08] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]
            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-08] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-08] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-08] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-08] = 0
        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]
            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)
            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-08] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-08] = 0
            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)
    return phi, psi1, psi2


def legendreDer(k, x):

    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def get_filter(base, k):

    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = np.pi / kUse / 2
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()
                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2
        PHI0[np.abs(PHI0) < 1e-08] = 0
        PHI1[np.abs(PHI1) < 1e-08] = 0
    H0[np.abs(H0) < 1e-08] = 0
    H1[np.abs(H1) < 1e-08] = 0
    G0[np.abs(G0) < 1e-08] = 0
    G1[np.abs(G1) < 1e-08] = 0
    return H0, H1, G0, G1, PHI0, PHI1


class sparseKernelFT1d(nn.Module):

    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.modes1 = alpha
        self.scale = 1 / (c * k * c * k)
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float))
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True
        self.k = k

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag), torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, x):
        B, N, c, k = x.shape
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        mode = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mode] = self.compl_mul1d('bix,iox->box', x_fft[:, :, :mode], torch.complex(self.weights1, self.weights2)[:, :, :mode])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):

    def __init__(self, k=3, alpha=64, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-08] = 0
        H1r[np.abs(H1r) < 1e-08] = 0
        G0r[np.abs(G0r) < 1e-08] = 0
        G1r[np.abs(G1r) < 1e-08] = 0
        self.max_item = 3
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class MultiWaveletTransform(AttentionOperator):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super().__init__()
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, None]:
        B, L, H, E = q.shape
        _, S, _, D = v.shape
        if L > S:
            zeros = torch.zeros_like(q[:, :L - S, :]).float()
            v = torch.cat([v, zeros], dim=1)
        else:
            v = v[:, :L, :, :]
        v = v.reshape(B, L, -1)
        V = self.Lk0(v).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return V.contiguous(), None


class FEDformerEncoder(nn.Module):

    def __init__(self, n_steps, n_layers, d_model, n_heads, d_ffn, moving_avg_window_size, dropout, version='Fourier', modes=32, mode_select='random', activation='relu'):
        super().__init__()
        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
        elif version == 'Fourier':
            encoder_self_att = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=n_steps, modes=modes, mode_select_method=mode_select)
        else:
            raise ValueError(f"Unsupported version: {version}. Please choose from ['Wavelets', 'Fourier'].")
        self.encoder = InformerEncoder([AutoformerEncoderLayer(encoder_self_att, d_model, n_heads, d_ffn, moving_avg_window_size, dropout, activation) for _ in range(n_layers)], norm_layer=SeasonalLayerNorm(d_model))

    def forward(self, X, attn_mask=None):
        enc_out, attns = self.encoder(X, attn_mask)
        return enc_out, attns


class _FEDformer(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, n_heads, d_ffn, moving_avg_window_size, dropout, version='Fourier', modes=32, mode_select='random', ORT_weight: 'float'=1, MIT_weight: 'float'=1, activation='relu'):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.encoder = FEDformerEncoder(n_steps, n_layers, d_model, n_heads, d_ffn, moving_avg_window_size, dropout, version, modes, mode_select, activation)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, attns = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class HiPPO_LegT(nn.Module):

    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        A, B = self.transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        B = B.squeeze(-1)
        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T))

    @staticmethod
    def transition(N):
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        return A, B

    def forward(self, inputs: 'torch.Tensor'):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        device = inputs.device
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N]))
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len, modes1, ratio=0.5, mode_type=0):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.ratio = ratio
        if mode_type == 0:
            self.modes2 = min(32, seq_len // 2)
            self.index = list(range(0, self.modes2))
        elif mode_type == 1:
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index0 = list(range(0, int(ratio * min(seq_len // 2, modes2))))
            self.index1 = list(range(len(self.index0), self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len // 2, self.modes2) - int(ratio * min(seq_len // 2, modes2))]
            self.index = self.index0 + self.index1
            self.index.sort()
        elif mode_type == 2:
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index = list(range(0, seq_len // 2))
            np.random.shuffle(self.index)
            self.index = self.index[:self.modes2]
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        if self.modes1 > 1000:
            for wi, i in enumerate(self.index):
                out_ft[:, :, :, i] = torch.einsum('bji,io->bjo', (x_ft[:, :, :, i], self.weights1[:, :, wi]))
        else:
            a = x_ft[:, :, :, :self.modes2]
            out_ft[:, :, :, :self.modes2] = torch.einsum('bjix,iox->bjox', a, self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class BackboneFiLM(nn.Module):

    def __init__(self, n_steps: 'int', in_channels: 'int', n_pred_steps: 'int', window_size: 'list', multiscale: 'list', modes1: 'int', ratio: 'float', mode_type: 'int'):
        super().__init__()
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.window_size = window_size
        self.multiscale = multiscale
        self.ratio = ratio
        self.affine_weight = nn.Parameter(torch.ones(1, 1, in_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1.0 / n_pred_steps / i) for n in window_size for i in multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n, seq_len=min(n_pred_steps, n_steps), modes1=modes1, ratio=ratio, mode_type=mode_type) for n in window_size for _ in range(len(multiscale))])
        self.mlp = nn.Linear(len(multiscale) * len(window_size), 1)

    def forward(self, X) ->torch.Tensor:
        x_enc = X * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.n_pred_steps
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.n_steps >= self.n_pred_steps:
                x_dec_c = out1.transpose(2, 3)[:, :, self.n_pred_steps - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.n_pred_steps:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        return x_dec


class _FiLM(nn.Module):

    def __init__(self, n_steps, n_features, window_size, multiscale, modes1, ratio, mode_type, d_model, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
        self.backbone = BackboneFiLM(n_steps, d_model, n_steps, window_size, multiscale, modes1, ratio, mode_type)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        X_embedding = self.saits_embedding(X, missing_mask)
        backbone_output = self.backbone(X_embedding)
        reconstruction = self.output_projection(backbone_output)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class BackboneFITS(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_pred_steps: 'int', cut_freq: 'int', individual: 'bool'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.individual = individual
        self.dominance_freq = cut_freq
        self.length_ratio = (n_steps + n_pred_steps) / n_steps
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.n_features):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))

    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=1)
        assert low_specx.size(1) >= self.dominance_freq, f'The sequence length after FFT {low_specx.size(1)} is less than the cut frequency {self.dominance_freq}. Please check the input sequence length, or decrease the cut frequency.'
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)], dtype=low_specx.dtype)
            for i in range(self.n_features):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        low_specxy = torch.zeros([low_specxy_.size(0), int((self.n_steps + self.n_pred_steps) / 2 + 1), low_specxy_.size(2)], dtype=low_specxy_.dtype)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio
        return low_xy


def nonstationary_denorm(X: 'torch.Tensor', means: 'torch.Tensor', stdev: 'torch.Tensor') ->torch.Tensor:
    """De-Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be de-normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization . Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    Returns
    -------
    X_denorm : torch.Tensor
        De-normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    """
    assert len(X) == len(means) == len(stdev), 'Input data and normalization parameters should have the same number of samples.'
    if len(means.shape) == 2:
        means = means.unsqueeze(1)
    if len(stdev.shape) == 2:
        stdev = stdev.unsqueeze(1)
    X = X * stdev
    X = X + means
    return X


def nonstationary_norm(X: 'torch.Tensor', missing_mask: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    missing_mask : torch.Tensor, optional
        Missing mask has the same shape as X. 1 indicates observed and 0 indicates missing.

    Returns
    -------
    X_enc : torch.Tensor
        Normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    """
    if torch.isnan(X).any():
        if missing_mask is None:
            missing_mask = torch.isnan(X)
        else:
            raise ValueError('missing_mask is given but X still contains nan values.')
    if missing_mask is None:
        means = X.mean(1, keepdim=True).detach()
        X_enc = X - means
        variance = torch.var(X_enc, dim=1, keepdim=True, unbiased=False) + 1e-09
        stdev = torch.sqrt(variance).detach()
    else:
        missing_sum = torch.sum(missing_mask == 1, dim=1, keepdim=True) + 1e-09
        means = torch.sum(X, dim=1, keepdim=True) / missing_sum
        X_enc = X - means
        X_enc = X_enc.masked_fill(missing_mask == 0, 0)
        variance = torch.sum(X_enc * X_enc, dim=1, keepdim=True) + 1e-09
        stdev = torch.sqrt(variance / missing_sum)
    X_enc /= stdev
    return X_enc, means, stdev


class _FITS(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', cut_freq: 'int', individual: 'bool', ORT_weight: 'float'=1, MIT_weight: 'float'=1, apply_nonstationary_norm: 'bool'=False):
        super().__init__()
        self.n_steps = n_steps
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.saits_embedding = SaitsEmbedding(n_features * 2, n_features, with_pos=False)
        self.backbone = BackboneFITS(n_steps, n_features, 0, cut_freq, individual)
        self.output_projection = nn.Linear(n_features, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, missing_mask)
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out = self.backbone(enc_out)
        if self.apply_nonstationary_norm:
            enc_out = nonstationary_denorm(enc_out, means, stdev)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class BackboneFreTS(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', embed_size: 'int', n_pred_steps: 'int', hidden_size: 'int', channel_independence: 'bool'=False):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.fc = nn.Sequential(nn.Linear(self.n_steps * self.embed_size, self.hidden_size), nn.LeakyReLU(), nn.Linear(self.hidden_size, self.n_pred_steps))

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.n_steps, dim=2, norm='ortho')
        return x

    def MLP_channel(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.n_features, dim=2, norm='ortho')
        x = x.permute(0, 2, 1, 3)
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)
        o1_real = F.relu(torch.einsum('bijd,dd->bijd', x.real, r) - torch.einsum('bijd,dd->bijd', x.imag, i) + rb)
        o1_imag = F.relu(torch.einsum('bijd,dd->bijd', x.imag, r) + torch.einsum('bijd,dd->bijd', x.real, i) + ib)
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        B, T, N = x.shape
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x


class _FreTS(nn.Module):

    def __init__(self, n_steps, n_features, embed_size: 'int'=128, hidden_size: 'int'=256, channel_independence: 'bool'=False, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, embed_size, with_pos=False)
        self.backbone = BackboneFreTS(n_steps, n_features, embed_size, n_steps, hidden_size, channel_independence)
        self.output_projection = nn.Linear(embed_size, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        backbone_output = self.backbone(enc_out)
        reconstruction = self.output_projection(backbone_output)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


def make_nn(input_size, output_size, hidden_sizes):
    """This function used to creates fully connected neural network.

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[i]))
        else:
            layers.append(nn.Linear(in_features=hidden_sizes[i - 1], out_features=hidden_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    return nn.Sequential(*layers)


class GpvaeDecoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=(256, 256)):
        """This module is a decoder with Gaussian output distribution.

        Parameters
        ----------
        output_size : int,
            the feature dimension of the output

        hidden_sizes: tuple
            the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers.
        """
        super().__init__()
        self.net = make_nn(input_size, output_size, hidden_sizes)

    def forward(self, x):
        mu = self.net(x)
        var = torch.ones_like(mu)
        return torch.distributions.Normal(mu, var)


class CustomConv1d(torch.nn.Conv1d):

    def __init(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        if len(x.shape) > 2:
            shape = list(np.arange(len(x.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            out = super().forward(x.permute(*new_shape))
            shape = list(np.arange(len(out.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            if self.kernel_size[0] % 2 == 0:
                out = F.pad(out, (0, -1), 'constant', 0)
            return out.permute(new_shape)
        return super().forward(x)


def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """This function used to construct neural network consisting of
       one 1d-convolutional layer that utilizes temporal dependencies,
       fully connected network

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers,

    kernel_size : int
        kernel size for convolutional layer

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    padding = kernel_size // 2
    cnn_layer = CustomConv1d(input_size, hidden_sizes[0], kernel_size=kernel_size, padding=padding)
    layers = [cnn_layer]
    for i, h in zip(hidden_sizes, hidden_sizes[1:]):
        layers.extend([nn.Linear(i, h), nn.ReLU()])
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [nn.Linear(hidden_sizes[-1], o) for o in output_size]
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return nn.Sequential(*layers)


class GpvaeEncoder(nn.Module):

    def __init__(self, input_size, z_size, hidden_sizes=(128, 128), window_size=24):
        """This module is an encoder with 1d-convolutional network and multivariate Normal posterior used by GP-VAE with
        proposed banded covariance matrix

        Parameters
        ----------
        input_size : int,
            the feature dimension of the input

        z_size : int,
            the feature dimension of the output latent embedding

        hidden_sizes : tuple,
            the tuple of the hidden layer sizes, and the tuple length sets the number of hidden layers

        window_size : int
            the kernel size for the Conv1D layer
        """
        super().__init__()
        self.z_size = int(z_size)
        self.input_size = input_size
        self.net, self.mu_layer, self.logvar_layer = make_cnn(input_size, (z_size, z_size * 2), hidden_sizes, window_size)

    def forward(self, x):
        mapped = self.net(x)
        batch_size = mapped.size(0)
        time_length = mapped.size(1)
        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        mapped_mean = torch.transpose(mu, num_dim - 1, num_dim - 2)
        mapped_covar = torch.transpose(logvar, num_dim - 1, num_dim - 2)
        mapped_covar = torch.sigmoid(mapped_covar)
        mapped_reshaped = mapped_covar.reshape(batch_size, self.z_size, 2 * time_length)
        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size * (2 * time_length - 1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), 2 * time_length - 1), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length - 1)]), batch_size * self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1, time_length)]), batch_size * self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)
        mapped_values = mapped_reshaped[:, :, :-1].reshape(-1)
        prec_sparse = torch.sparse_coo_tensor(torch.LongTensor(idxs_all).t(), mapped_values, dense_shape)
        prec_sparse = prec_sparse.coalesce()
        prec_tril = prec_sparse.to_dense()
        eye = torch.eye(prec_tril.shape[-1]).unsqueeze(0).repeat(prec_tril.shape[0], prec_tril.shape[1], 1, 1)
        prec_tril = prec_tril + eye
        cov_tril = torch.linalg.solve_triangular(prec_tril, eye, upper=True)
        cov_tril = torch.where(torch.isfinite(cov_tril), cov_tril, torch.zeros_like(cov_tril))
        num_dim = len(cov_tril.shape)
        cov_tril_lower = torch.transpose(cov_tril, num_dim - 1, num_dim - 2)
        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, scale_tril=cov_tril_lower)
        return z_dist


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = sigma / (distance_matrix_scaled + 1.0)
    alpha = 0.001
    eye = torch.eye(kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, 'length_scale has to be smaller than 0.5 for the kernel matrix to be diagonally dominant'
    sigmas = torch.ones(T, T) * length_scale
    sigmas_tridiag = torch.diagonal(sigmas, offset=0, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=1, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=-1, dim1=-2, dim2=-1)
    kernel_matrix = sigmas_tridiag + torch.eye(T) * (1.0 - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale).type(torch.float32)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def rbf_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


class BackboneGPVAE(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(self, input_dim, time_length, latent_dim, encoder_sizes=(64, 64), decoder_sizes=(64, 64), beta=1, M=1, K=1, kernel='cauchy', sigma=1.0, length_scale=7.0, kernel_scales=1, window_size=24):
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, window_size)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes)
        self.M = M
        self.K = K
        self.prior = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if not torch.is_tensor(z):
            z = torch.tensor(z).float()
        num_dim = len(z.shape)
        assert num_dim > 2
        return self.decoder(torch.transpose(z, num_dim - 1, num_dim - 2))

    @staticmethod
    def kl_divergence(a, b):
        return torch.distributions.kl.kl_divergence(a, b)

    def _init_prior(self, device='cpu'):
        kernel_matrices = []
        for i in range(self.kernel_scales):
            if self.kernel == 'rbf':
                kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2 ** i))
            elif self.kernel == 'diffusion':
                kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2 ** i))
            elif self.kernel == 'matern':
                kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2 ** i))
            elif self.kernel == 'cauchy':
                kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2 ** i))
        tiled_matrices = []
        total = 0
        for i in range(self.kernel_scales):
            if i == self.kernel_scales - 1:
                multiplier = self.latent_dim - total
            else:
                multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                total += multiplier
            tiled_matrices.append(torch.unsqueeze(kernel_matrices[i], 0).repeat(multiplier, 1, 1))
        kernel_matrix_tiled = torch.cat(tiled_matrices)
        assert len(kernel_matrix_tiled) == self.latent_dim
        prior = torch.distributions.MultivariateNormal(loc=torch.zeros(self.latent_dim, self.time_length, device=device), covariance_matrix=kernel_matrix_tiled)
        return prior

    def impute(self, X, missing_mask, n_sampling_times=1):
        n_samples, n_steps, n_features = X.shape
        X = X.repeat(n_sampling_times, 1, 1)
        missing_mask = missing_mask.repeat(n_sampling_times, 1, 1).type(torch.bool)
        decode_x_mean = self.decode(self.encode(X).mean).mean
        imputed_data = decode_x_mean * ~missing_mask + X * missing_mask
        imputed_data = imputed_data.reshape(n_sampling_times, n_samples, n_steps, n_features).permute(1, 0, 2, 3)
        return imputed_data

    def forward(self, X, missing_mask):
        X = X.repeat(self.K * self.M, 1, 1)
        missing_mask = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)
        if self.prior is None:
            self.prior = self._init_prior(device=X.device)
        qz_x = self.encode(X)
        z = qz_x.rsample()
        px_z = self.decode(z)
        nll = -px_z.log_prob(X)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if missing_mask is not None:
            nll = torch.where(missing_mask, nll, torch.zeros_like(nll))
        nll = nll.sum(dim=(1, 2))
        if self.K > 1:
            kl = qz_x.log_prob(z) - self.prior.log_prob(z)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)
            weights = -nll - kl
            weights = torch.reshape(weights, [self.M, self.K, -1])
            elbo = torch.logsumexp(weights, dim=1)
            elbo = elbo.mean()
        else:
            kl = self.kl_divergence(qz_x, self.prior)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)
            elbo = -nll - self.beta * kl
            elbo = elbo.mean()
        return -elbo


class _GPVAE(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(self, input_dim, time_length, latent_dim, encoder_sizes=(64, 64), decoder_sizes=(64, 64), beta=1, M=1, K=1, kernel='cauchy', sigma=1.0, length_scale=7.0, kernel_scales=1, window_size=24):
        super().__init__()
        self.backbone = BackboneGPVAE(input_dim, time_length, latent_dim, encoder_sizes, decoder_sizes, beta, M, K, kernel, sigma, length_scale, kernel_scales, window_size)

    def forward(self, inputs, training=True, n_sampling_times=1):
        X, missing_mask = inputs['X'], inputs['missing_mask']
        results = {}
        if training:
            elbo_loss = self.backbone(X, missing_mask)
            results['loss'] = elbo_loss
        else:
            imputed_data = self.backbone.impute(X, missing_mask, n_sampling_times)
            results['imputed_data'] = imputed_data
        return results


class EmbeddedAttention(nn.Module):
    """
    Spatial embedded attention layer.
    The node embedding serves as the query and key matrices for attentive aggregation on graphs.
    """

    def __init__(self, model_dim, node_embedding_dim):
        super().__init__()
        self.model_dim = model_dim
        self.FC_Q_K = nn.Linear(node_embedding_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb):
        batch_size = value.shape[0]
        query = self.FC_Q_K(emb)
        key = self.FC_Q_K(emb)
        value = self.FC_V(value)
        key = key.transpose(-1, -2)
        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)
        query = repeat(query, 'n s1 s2 -> b n s1 s2', b=batch_size)
        key = repeat(key, 'n s2 s1 -> b n s2 s1', b=batch_size)
        out = key @ value
        out = query @ out
        return out


class EmbeddedAttentionLayer(nn.Module):

    def __init__(self, model_dim, node_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()
        self.attn = EmbeddedAttention(model_dim, node_embedding_dim)
        self.feed_forward = nn.Sequential(nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True), nn.Linear(feed_forward_dim, model_dim))
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, emb)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out


class MLP(nn.Module):
    """
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    """

    def __init__(self, d_in, d_out, d_hidden=128, n_hidden_layers=2, dropout=0.05, activation='tanh'):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        layers = [nn.Linear(self.d_in, self.d_hidden), self.activation, nn.Dropout(self.dropout)]
        for i in range(self.n_hidden_layers - 2):
            layers += [nn.Linear(self.d_hidden, self.d_hidden), self.activation, nn.Dropout(dropout)]
        layers += [nn.Linear(d_hidden, d_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = query @ key / self.head_dim ** 0.5
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer.
    A low-rank factorization is achieved in the temporal attention matrix.
    """

    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(seq_len, dim_proj, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.seq_len = seq_len

    def forward(self, x):
        batch = x.shape[0]
        projector = repeat(self.projector, 'seq_len dim_proj d_model -> repeat seq_len dim_proj d_model', repeat=batch)
        message_out = self.out_attn(projector, x, x)
        message_in = self.in_attn(x, projector, message_out)
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)
        return message


class _ImputeFormer(nn.Module):
    """
    Spatiotemporal Imputation Transformer induced by low-rank factorization, KDD'24.
    Note:
        This is a simplified implementation under the SAITS framework (ORT+MIT).
        The timestamp encoding is also removed for ease of implementation.
    """

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_input_embed: 'int', d_learnable_embed: 'int', d_proj: 'int', d_ffn: 'int', n_temporal_heads: 'int', dropout: 'float'=0.0, input_dim: 'int'=1, output_dim: 'int'=1, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_nodes = n_features
        self.in_steps = n_steps
        self.out_steps = n_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = d_input_embed
        self.learnable_embedding_dim = d_learnable_embed
        self.model_dim = d_input_embed + d_learnable_embed
        self.n_temporal_heads = n_temporal_heads
        self.num_layers = n_layers
        self.input_proj = nn.Linear(input_dim, self.input_embedding_dim)
        self.d_proj = d_proj
        self.d_ffn = d_ffn
        self.learnable_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.in_steps, self.n_nodes, self.learnable_embedding_dim)))
        self.readout = MLP(self.model_dim, self.model_dim, output_dim, n_layers=2)
        self.attn_layers_t = nn.ModuleList([ProjectedAttentionLayer(self.n_nodes, self.d_proj, self.model_dim, self.n_temporal_heads, self.model_dim, dropout) for _ in range(self.num_layers)])
        self.attn_layers_s = nn.ModuleList([EmbeddedAttentionLayer(self.model_dim, self.learnable_embedding_dim, self.d_ffn) for _ in range(self.num_layers)])
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        x, missing_mask = inputs['X'], inputs['missing_mask']
        x = x.unsqueeze(-1)
        missing_mask = missing_mask.unsqueeze(-1)
        batch_size = x.shape[0]
        x = x * missing_mask
        x = self.input_proj(x)
        node_emb = self.learnable_embedding.expand(batch_size, *self.learnable_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)
        x = x.permute(0, 2, 1, 3)
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)
        x = x.permute(0, 2, 1, 3)
        reconstruction = self.readout(x)
        reconstruction = reconstruction.squeeze(-1)
        missing_mask = missing_mask.squeeze(-1)
        imputed_data = missing_mask * inputs['X'] + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class ConvLayer(nn.Module):

    def __init__(self, c_in, window_size):
        super().__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=window_size, stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class InformerEncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class ProbMask:

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


class ProbAttention(AttentionOperator):

    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, scale=None):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
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
        attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn)
        attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        return context_in, attns

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs):
        B, L_Q, H, D = q.shape
        _, L_K, _, _ = k.shape
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(v, L_Q)
        context, attn = self._update_context(context, v, scores_top, index, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn


class _Informer(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, n_heads, d_ffn, factor, dropout, ORT_weight: 'float'=1, MIT_weight: 'float'=1, distil=False, activation='relu'):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.encoder = InformerEncoder([InformerEncoderLayer(MultiHeadAttention(ProbAttention(False, factor, dropout), d_model, n_heads, d_model // n_heads, d_model // n_heads), d_model, d_ffn, dropout, activation) for _ in range(n_layers)], [ConvLayer(d_model) for _ in range(n_layers - 1)] if distil else None, norm_layer=nn.LayerNorm(d_model))
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, attns = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class _iTransformer(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_features = n_features
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.saits_embedding = SaitsEmbedding(n_steps, d_model, with_pos=False, dropout=dropout)
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)
        self.output_projection = nn.Linear(d_model, n_steps)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X.permute(0, 2, 1), missing_mask.permute(0, 2, 1)], dim=1)
        input_X = self.saits_embedding(input_X)
        enc_output, _ = self.encoder(input_X)
        reconstruction = self.output_projection(enc_output)
        reconstruction = reconstruction.permute(0, 2, 1)[:, :, :self.n_features]
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """

    def __init__(self, mask_spectrum):
        super().__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var
        return x_var, x_inv


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """

    def __init__(self, input_len=96, pred_len=96, dynamic_dim=128, encoder=None, decoder=None):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder
        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        res = x.transpose(1, 2)
        res = self.encoder(res)
        res = self.K(res)
        res = self.decoder(res)
        res = res.transpose(1, 2)
        return res


class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """

    def __init__(self):
        super().__init__()
        self.K = None

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(self.K).any():
            None
            self.K = torch.eye(self.K.shape[1]).unsqueeze(0).repeat(B, 1, 1)
        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred
        return z_pred

    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """

    def __init__(self):
        super().__init__()
        self.K = None
        self.K_step = None

    def forward(self, z, pred_len=1):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(self.K).any():
            None
            self.K = torch.eye(self.K.shape[1]).unsqueeze(0).repeat(B, 1, 1)
        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                None
                self.K_step = torch.eye(self.K_step.shape[1]).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                None
                self.K_step = torch.eye(self.K_step.shape[1]).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
        return z_rec, z_pred


class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """

    def __init__(self, enc_in=8, input_len=96, pred_len=96, seg_len=24, dynamic_dim=128, encoder=None, decoder=None, multistep=False):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)
        self.step = math.ceil(self.pred_len / self.seg_len)
        self.padding_len = self.seg_len * self.freq - self.input_len
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x):
        B, L, C = x.shape
        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)
        res = self.encoder(res)
        x_rec, x_pred = self.dynamics(res, self.step)
        x_rec = self.decoder(x_rec)
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]
        x_pred = self.decoder(x_pred)
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]
        return x_rec, x_pred


class BackboneKoopa(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_pred_steps: 'int', n_seg_steps: 'int', d_dynamic: 'int', d_hidden: 'int', n_hidden_layers: 'int', n_blocks: 'int', multistep: 'bool', alpha: 'int'=0.2):
        super().__init__()
        self.n_blocks = n_blocks
        self.alpha = alpha
        self.mask_spectrum = None
        self.disentanglement = FourierFilter(self.mask_spectrum)
        self.time_inv_encoder = MLP(d_in=n_steps, d_out=d_dynamic, activation='relu', d_hidden=d_hidden, n_hidden_layers=n_hidden_layers)
        self.time_inv_decoder = MLP(d_in=d_dynamic, d_out=n_pred_steps, activation='relu', d_hidden=d_hidden, n_hidden_layers=n_hidden_layers)
        self.time_inv_kps = nn.ModuleList([TimeInvKP(input_len=n_steps, pred_len=n_pred_steps, dynamic_dim=d_dynamic, encoder=self.time_inv_encoder, decoder=self.time_inv_decoder) for _ in range(n_blocks)])
        self.time_var_encoder = MLP(d_in=n_seg_steps * n_features, d_out=d_dynamic, activation='tanh', d_hidden=d_hidden, n_hidden_layers=n_hidden_layers)
        self.time_var_decoder = MLP(d_in=d_dynamic, d_out=n_seg_steps * n_features, activation='tanh', d_hidden=d_hidden, n_hidden_layers=n_hidden_layers)
        self.time_var_kps = nn.ModuleList([TimeVarKP(enc_in=n_features, input_len=n_steps, pred_len=n_pred_steps, seg_len=n_seg_steps, dynamic_dim=d_dynamic, encoder=self.time_var_encoder, decoder=self.time_var_decoder, multistep=multistep) for _ in range(n_blocks)])

    def _get_mask_spectrum(self, train_dataloader):
        """
        get shared frequency spectrums
        """
        amps = 0.0
        for _, data in enumerate(train_dataloader):
            lookback_window = data[1]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
        mask_spectrum = amps.topk(int(amps.shape[0] * self.alpha)).indices
        return mask_spectrum

    def init_mask_spectrum(self, train_dataloader: 'DataLoader'):
        self.mask_spectrum = self._get_mask_spectrum(train_dataloader)

    def forward(self, X):
        assert self.mask_spectrum is not None, 'Please initialize the mask spectrum first with init_mask_spectrum() method.'
        residual, output = X, None
        for i in range(self.n_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if output is None:
                output = time_inv_output + time_var_output
            else:
                output += time_inv_output + time_var_output
        return output


class _Koopa(nn.Module):

    def __init__(self, n_steps, n_features, n_seg_steps, d_dynamic, d_hidden, n_hidden_layers, n_blocks, multistep, alpha, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, n_features, with_pos=False)
        self.backbone = BackboneKoopa(n_steps, n_features, n_steps, n_seg_steps, d_dynamic, d_hidden, n_hidden_layers, n_blocks, multistep, alpha)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training=False) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        reconstruction = self.backbone(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if self.training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6]):
        super().__init__()
        self.conv_kernel = conv_kernel
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=1) for i in isometric_kernel])
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=i // 2, stride=i) for i in conv_kernel])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=i) for i in conv_kernel])
        self.decomp = nn.ModuleList([SeriesDecompositionBlock(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=(len(self.conv_kernel), 1))
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)
        x1 = self.drop(self.act(conv1d(x)))
        x = x1
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=input.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]
        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)
        mg = torch.tensor([], device=src.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):

    def __init__(self, embedding_size=512, d_layers=1, decomp_kernel=[32], c_out=1, conv_kernel=[2, 4], isometric_kernel=[18, 6]):
        super().__init__()
        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, decomp_kernel=decomp_kernel, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel) for _ in range(d_layers)])
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class BackboneMICN(nn.Module):

    def __init__(self, n_steps, n_features, n_pred_steps, n_pred_features, n_layers, d_model, decomp_kernel, isometric_kernel, conv_kernel: 'list'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.conv_trans = SeasonalPrediction(embedding_size=d_model, d_layers=n_layers, decomp_kernel=decomp_kernel, c_out=n_pred_features, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel)

    def forward(self, x):
        dec_out = self.conv_trans(x)
        return dec_out


class SeriesDecompositionMultiBlock(nn.Module):
    """
    Series decomposition block from FEDfromer,
    i.e. series_decomp_multi from https://github.com/MAZiqing/FEDformer

    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = [MovingAvgBlock(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class _MICN(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', dropout: 'float', conv_kernel: 'list'=None, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, dropout=dropout)
        decomp_kernel = []
        isometric_kernel = []
        for ii in conv_kernel:
            if ii % 2 == 0:
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((n_steps + n_steps + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((n_steps + n_steps + ii - 1) // ii)
        self.decomp_multi = SeriesDecompositionMultiBlock(decomp_kernel)
        self.backbone = BackboneMICN(n_steps, n_features, n_steps, n_features, n_layers, d_model, decomp_kernel, isometric_kernel, conv_kernel)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        seasonal_init, trend_init = self.decomp_multi(X)
        enc_out = self.saits_embedding(seasonal_init, missing_mask)
        reconstruction = self.backbone(enc_out)
        reconstruction = reconstruction + trend_init
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class FlattenHead(nn.Module):

    def __init__(self, d_input, d_output, n_features, head_dropout=0, individual=False):
        super().__init__()
        self.individual = individual
        self.n_features = n_features
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_features):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(d_input, d_output))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_input, d_output)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_features):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False, nvars=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel, stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):
        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dims=-1)
        x = torch.cat([x, pad_right], dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2, (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels, out_channels=self.lkb_origin.conv.out_channels, kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride, padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation, groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):

    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=large_size, stride=1, groups=nvars * dmodel, small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)
        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)
        x = input + x
        return x


class Stage(nn.Module):

    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class BackboneModernTCN(nn.Module):

    def __init__(self, n_steps, n_features, n_predict_features, patch_size, patch_stride, downsampling_ratio, ffn_ratio, num_blocks: 'list', large_size: 'list', small_size: 'list', dims: 'list', small_kernel_merged: 'bool'=False, backbone_dropout: 'float'=0.1, head_dropout: 'float'=0.1, use_multi_scale: 'bool'=True, individual: 'bool'=False, freq: 'str'='h'):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Linear(patch_size, dims[0])
        self.downsample_layers.append(stem)
        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(nn.BatchNorm1d(dims[i]), nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsampling_ratio, stride=downsampling_ratio))
                self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsampling_ratio
        if freq == 'h':
            time_feature_num = 4
        elif freq == 't':
            time_feature_num = 5
        else:
            raise NotImplementedError('time_feature_num should be 4 or 5')
        self.te_patch = nn.Sequential(nn.Conv1d(time_feature_num, time_feature_num, kernel_size=patch_size, stride=patch_stride, groups=time_feature_num), nn.Conv1d(time_feature_num, dims[0], kernel_size=1, stride=1, groups=1), nn.BatchNorm1d(dims[0]))
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx], nvars=n_features, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)
        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsampling_ratio
        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for i in range(self.num_stage):
            align_dim = dims[-1]
            lat = nn.Conv1d(dims[i], align_dim, kernel_size=1, stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)
            up_conv = nn.Sequential(nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.up_sample_ratio, stride=self.up_sample_ratio), nn.BatchNorm1d(align_dim))
            self.up_sample_conv.append(up_conv)
        patch_num = n_steps // patch_stride
        self.n_features = n_features
        self.individual = individual
        d_model = dims[self.num_stage - 1]
        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = FlattenHead(self.head_nf, n_predict_features, n_features, head_dropout, individual)
        else:
            if patch_num % pow(downsampling_ratio, self.num_stage - 1) == 0:
                self.head_nf = d_model * patch_num // pow(downsampling_ratio, self.num_stage - 1)
            else:
                self.head_nf = d_model * (patch_num // pow(downsampling_ratio, self.num_stage - 1) + 1)
            self.head = FlattenHead(self.head_nf, n_predict_features, n_features, head_dropout, individual)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    def forward(self, x):
        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(B, M, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)
            elif N % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - N % self.downsample_ratio
                x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                x = self.downsample_layers[i](x)
                _, D_, N_ = x.shape
                x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x


class _ModernTCN(nn.Module):

    def __init__(self, n_steps, n_features, patch_size, patch_stride, downsampling_ratio, ffn_ratio, num_blocks: 'list', large_size: 'list', small_size: 'list', dims: 'list', small_kernel_merged: 'bool'=False, backbone_dropout: 'float'=0.1, head_dropout: 'float'=0.1, use_multi_scale: 'bool'=True, individual: 'bool'=False, apply_nonstationary_norm: 'bool'=False):
        super().__init__()
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.backbone = BackboneModernTCN(n_steps, n_features, n_features, patch_size, patch_stride, downsampling_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, small_kernel_merged, backbone_dropout, head_dropout, use_multi_scale, individual)
        self.projection = FlattenHead(self.backbone.head_nf, n_steps, n_features, head_dropout, individual)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, missing_mask)
        in_X = X.permute(0, 2, 1)
        in_X = self.backbone(in_X)
        reconstruction = self.projection(in_X)
        reconstruction = reconstruction.permute(0, 2, 1)
        if self.apply_nonstationary_norm:
            reconstruction = nonstationary_denorm(reconstruction, means, stdev)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            loss = calc_mse(reconstruction, inputs['X_ori'], inputs['indicating_mask'])
            results['loss'] = loss
        return results


class MrnnFcnRegression(nn.Module):
    """M-RNN fully connection regression Layer"""

    def __init__(self, feature_num):
        super().__init__()
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))
        self.final_linear = nn.Linear(feature_num, feature_num)
        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x, missing_mask, target):
        h_t = torch.sigmoid(F.linear(x, self.U * self.m) + F.linear(target, self.V1 * self.m) + F.linear(missing_mask, self.V2) + self.beta)
        x_hat_t = torch.sigmoid(self.final_linear(h_t))
        return x_hat_t


class BackboneMRNN(nn.Module):

    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.f_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.b_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.fcn_regression = MrnnFcnRegression(n_features)

    def gene_hidden_states(self, inputs, feature_idx):
        X_f = inputs['forward']['X'][:, :, feature_idx].unsqueeze(dim=2)
        M_f = inputs['forward']['missing_mask'][:, :, feature_idx].unsqueeze(dim=2)
        D_f = inputs['forward']['deltas'][:, :, feature_idx].unsqueeze(dim=2)
        X_b = inputs['backward']['X'][:, :, feature_idx].unsqueeze(dim=2)
        M_b = inputs['backward']['missing_mask'][:, :, feature_idx].unsqueeze(dim=2)
        D_b = inputs['backward']['deltas'][:, :, feature_idx].unsqueeze(dim=2)
        device = X_f.device
        batch_size = X_f.size()[0]
        f_hidden_state_0 = torch.zeros((1, batch_size, self.rnn_hidden_size), device=device)
        b_hidden_state_0 = torch.zeros((1, batch_size, self.rnn_hidden_size), device=device)
        f_input = torch.cat([X_f, M_f, D_f], dim=2)
        b_input = torch.cat([X_b, M_b, D_b], dim=2)
        hidden_states_f, _ = self.f_rnn(f_input, f_hidden_state_0)
        hidden_states_b, _ = self.b_rnn(b_input, b_hidden_state_0)
        hidden_states_b = torch.flip(hidden_states_b, dims=[1])
        feature_estimation = self.concated_hidden_project(torch.cat([hidden_states_f, hidden_states_b], dim=2))
        return feature_estimation, hidden_states_f, hidden_states_b

    def forward(self, inputs: 'dict') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = inputs['forward']['X']
        M = inputs['forward']['missing_mask']
        feature_collector = []
        for f in range(self.n_features):
            feat_estimation, hid_states_f, hid_states_b = self.gene_hidden_states(inputs, f)
            feature_collector.append(feat_estimation)
        RNN_estimation = torch.concat(feature_collector, dim=2)
        RNN_imputed_data = M * X + (1 - M) * RNN_estimation
        FCN_estimation = self.fcn_regression(X, M, RNN_imputed_data)
        return RNN_estimation, RNN_imputed_data, FCN_estimation


def calc_rmse(predictions: 'Union[np.ndarray, torch.Tensor]', targets: 'Union[np.ndarray, torch.Tensor]', masks: 'Optional[Union[np.ndarray, torch.Tensor]]'=None) ->Union[float, torch.Tensor]:
    """Calculate the Root Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = calc_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = calc_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(calc_mse(predictions, targets, masks))


class _MRNN(nn.Module):

    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()
        self.backbone = BackboneMRNN(n_steps, n_features, rnn_hidden_size)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X = inputs['forward']['X']
        M = inputs['forward']['missing_mask']
        RNN_estimation, RNN_imputed_data, FCN_estimation = self.backbone(inputs)
        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {'imputed_data': imputed_data}
        if training:
            RNN_loss = calc_rmse(RNN_estimation, X, M)
            FCN_loss = calc_rmse(FCN_estimation, RNN_imputed_data)
            reconstruction_loss = RNN_loss + FCN_loss
            results['loss'] = reconstruction_loss
        return results


class DeStationaryAttention(AttentionOperator):
    """De-stationary Attention"""

    def __init__(self, temperature: 'float', attn_dropout: 'float'=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q: 'torch.Tensor', v: 'torch.Tensor', k: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        B, L, H, E = q.shape
        _, S, _, D = v.shape
        temperature = self.temperature or 1.0 / math.sqrt(E)
        tau, delta = kwargs['tau'], kwargs['delta']
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        scores = torch.einsum('blhe,bshe->bhls', q, k) * tau + delta
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -torch.inf)
        attn = self.dropout(torch.softmax(temperature * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', attn, v)
        output = V.contiguous()
        return output, attn


class NonstationaryTransformerEncoder(nn.Module):
    """NonstationaryTransformer encoder.
    Its arch is the same with the original Transformer encoder,
    but the attention operator is replaced by the DeStationaryAttention.

    Parameters
    ----------
    n_layers:
        The number of layers in the encoder.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer in the feed-forward network.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float'):
        super().__init__()
        self.enc_layer_stack = nn.ModuleList([TransformerEncoderLayer(DeStationaryAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, x: 'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Forward processing of the encoder.

        Parameters
        ----------
        x:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights_collector:
            A list containing the attention map from each encoder layer.

        """
        attn_weights_collector = []
        enc_output = x
        if src_mask is None:
            bz, n_steps, _ = x.shape
            mask_shape = [bz, n_steps, n_steps]
            src_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
        for layer in self.enc_layer_stack:
            enc_output, attn_weights = layer(enc_output, src_mask, **kwargs)
            attn_weights_collector.append(attn_weights)
        return enc_output, attn_weights_collector


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(self, d_in: 'int', n_steps: 'int', d_hidden: 'list', n_hidden_layers: 'int', d_output: 'int', kernel_size: 'int'=3):
        super().__init__()
        assert len(d_hidden) == n_hidden_layers, f'The length of d_hidden should be equal to n_hidden_layers, but got {len(d_hidden)} and {n_hidden_layers}.'
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=n_steps, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        layers = [nn.Linear(2 * d_in, d_hidden[0]), nn.ReLU()]
        for i in range(n_hidden_layers - 1):
            layers += [nn.Linear(d_hidden[i], d_hidden[i + 1]), nn.ReLU()]
        layers += [nn.Linear(d_hidden[-1], d_output, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y


class _NonstationaryTransformer(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_ffn: 'int', d_projector_hidden: 'list', n_projector_hidden_layers: 'int', dropout: 'float', attn_dropout: 'float', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        d_k = d_v = d_model // n_heads
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, dropout=dropout)
        self.encoder = NonstationaryTransformerEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)
        self.tau_learner = Projector(d_in=n_features, n_steps=n_steps, d_hidden=d_projector_hidden, n_hidden_layers=n_projector_hidden_layers, d_output=1)
        self.delta_learner = Projector(d_in=n_features, n_steps=n_steps, d_hidden=d_projector_hidden, n_hidden_layers=n_projector_hidden_layers, d_output=n_steps)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        X_enc, means, stdev = nonstationary_norm(X, missing_mask)
        tau = self.tau_learner(X, stdev).exp()
        delta = self.delta_learner(X, means)
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, attns = self.encoder(enc_out, tau=tau, delta=delta)
        reconstruction = self.output_projection(enc_out)
        reconstruction = nonstationary_denorm(reconstruction, means, stdev)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class PatchtstEncoder(nn.Module):

    def __init__(self, n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float'):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)

    def forward(self, x, attn_mask=None):
        enc_out, attns = self.encoder(x, attn_mask)
        enc_out = enc_out.reshape(-1, self.d_model, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        return enc_out, attns


class _PatchTST(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', patch_len: 'int', stride: 'int', dropout: 'float', attn_dropout: 'float', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        n_patches = int((n_steps - patch_len) / stride + 2)
        padding = stride
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        self.encoder = PatchtstEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)
        self.head = PredictionHead(d_model, n_patches, n_steps, dropout)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        input_X = self.saits_embedding(X, missing_mask)
        enc_out = self.patch_embedding(input_X.permute(0, 2, 1))
        enc_out, attns = self.encoder(enc_out)
        dec_out = self.head(enc_out)
        reconstruction = self.output_projection(dec_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([ConvLayer(d_inner, window_size), ConvLayer(d_inner, window_size), ConvLayer(d_inner, window_size)])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = nn.Linear(d_inner, d_model)
        self.down = nn.Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = self.norm(all_inputs)
        return all_inputs


def get_mask(input_size, window_size, inner_size):
    """Get the attention mask of PAM-Naive"""
    all_size = [input_size]
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length)
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = start - all_size[layer_idx - 1] + (i - start) * window_size[layer_idx - 1]
            if i == start + all_size[layer_idx] - 1:
                right_side = start
            else:
                right_side = start - all_size[layer_idx - 1] + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1
    mask = (1 - mask).bool()
    return mask, all_size


def refer_points(all_sizes, window_size):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes))
    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index
    indexes = indexes.unsqueeze(0).unsqueeze(3)
    return indexes.long()


class PyraformerEncoder(nn.Module):

    def __init__(self, n_steps: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float', window_size: 'list', inner_size: 'int'):
        super().__init__()
        d_bottleneck = d_model // 4
        d_k = d_v = d_model // n_heads
        self.mask, self.all_size = get_mask(n_steps, window_size, inner_size)
        self.indexes = refer_points(self.all_size, window_size)
        self.layer_stack = nn.ModuleList([TransformerEncoderLayer(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout=dropout) for _ in range(n_layers)])
        self.conv_layers = Bottleneck_Construct(d_model, window_size, d_bottleneck)

    def forward(self, x: 'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        mask = self.mask.repeat(len(x), 1, 1)
        x = self.conv_layers(x)
        attn_weights_collector = []
        for layer in self.layer_stack:
            x, attn_weights = layer(x, mask)
            attn_weights_collector.append(attn_weights)
        indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2))
        indexes = indexes.view(x.size(0), -1, x.size(2))
        all_enc = torch.gather(x, 1, indexes)
        enc_output = all_enc.view(x.size(0), self.all_size[0], -1)
        return enc_output, attn_weights_collector


class _Pyraformer(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float', window_size: 'list', inner_size: 'int', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, dropout=dropout)
        self.encoder = PyraformerEncoder(n_steps, n_layers, d_model, n_heads, d_ffn, dropout, attn_dropout, window_size, inner_size)
        self.output_projection = nn.Linear((len(window_size) + 1) * d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, attns = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


TOKEN_SELF_ATTN_VALUE = -50000.0


def default(val, default_val):
    return default_val if val is None else val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class FullQKAttention(nn.Module):

    def __init__(self, causal=False, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len
        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)
        dot = torch.einsum('bie,bje->bij', q, qk) * dim ** -0.5
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)
        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value
        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)
        out = torch.einsum('bij,bje->bie', dot, v)
        return out, dot, torch.empty(0)


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(qk, sinu_pos):
    sinu_pos = sinu_pos.type(qk.dtype)
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'n d -> n (d j)', j=2), (sin, cos))
    seq_len = sin.shape[0]
    qk, qk_pass = qk[:, :seq_len], qk[:, seq_len:]
    qk = qk * cos + rotate_every_two(qk) * sin
    return torch.cat((qk, qk_pass), dim=1)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):

    def inner_fn(fn):

        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'
            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def exists(val):
    return val is not None


def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


class LSHAttention(nn.Module):

    def __init__(self, dropout=0.0, bucket_size=64, n_hashes=8, causal=False, allow_duplicate_attention=True, attend_across_buckets=True, rehash_each_round=True, drop_for_hash_rate=0.0, random_rotations_per_head=False, return_attn=False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)
        assert rehash_each_round or allow_duplicate_attention, 'The setting {allow_duplicate_attention=False, rehash_each_round=False} is not implemented.'
        self.causal = causal
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head
        self._return_attn = return_attn
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device
        assert n_buckets % 2 == 0
        rot_size = n_buckets
        rotations_shape = batch_size if self._random_rotations_per_head else 1, vecs.shape[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)
            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[..., -self.n_hashes:].transpose(1, 2)
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1))
        return buckets

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, pos_emb=None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)
        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'
        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)
        assert int(buckets.shape[1]) == self.n_hashes * seqlen
        total_hashes = self.n_hashes
        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + ticker % seqlen
        buckets_and_t = buckets_and_t.detach()
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker
        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()
        if exists(pos_emb):
            qk = apply_rotary_pos_emb(qk, pos_emb)
        st = sticker % seqlen
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)
        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * dim ** -0.5
        masked_value = max_neg_value(dots)
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = (bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :]
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([torch.reshape(locs1, (batch_size, total_hashes, seqlen)), torch.reshape(locs2, (batch_size, total_hashes, seqlen))], 1).permute((0, 2, 1))
            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))
            b_locs1 = b_locs[:, :, :, None, :total_hashes]
            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)
            dup_counts = bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :]
            dup_counts = chunked_sum(dup_counts, chunks=total_hashes * batch_size)
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-09)
            del dup_counts
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)
        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1))
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)
        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))
        if query_len != seqlen:
            query_slice = slice(None), slice(None), slice(0, query_len)
            o, logits = o[query_slice], logits[query_slice]
        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)
        attn = torch.empty(0, device=device)
        if self._return_attn:
            attn_unsort = (bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :]
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)
        return out, attn, buckets


class SinusoidalEmbeddings(nn.Module):

    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent=False)

    @autocast(enabled=False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - seq_len // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)
    return normed.type(dtype)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:ind + t, ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


class LocalAttention(nn.Module):

    def __init__(self, window_size, causal=False, look_backward=1, look_forward=None, dropout=0.0, shared_qk=False, rel_pos_emb_config=None, dim=None, autopad=False, exact_windowsize=False, scale=None, use_rotary_pos_emb=True, use_xpos=False, xpos_scale_base=None):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        self.scale = scale
        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.rel_pos = None
        self.use_xpos = use_xpos
        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim, use_xpos=use_xpos, scale_base=default(xpos_scale_base, window_size // 2))

    def forward(self, q, k, v, mask=None, input_mask=None, attn_bias=None, window_size=None):
        mask = default(mask, input_mask)
        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'
        autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))
        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v))
        b, n, dim_head, device = *q.shape, q.device
        scale = default(self.scale, dim_head ** -0.5)
        assert n % window_size == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'
        windows = n // window_size
        if shared_qk:
            k = l2norm(k)
        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=window_size)
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w=windows), (q, k, v))
        bq = bq * scale
        look_around_kwargs = dict(backward=look_backward, forward=look_forward, pad_value=pad_value)
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')
        pad_mask = bq_k == pad_value
        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert b % heads == 0
            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b=b // heads)
            sim = sim + attn_bias
        mask_value = max_neg_value(sim)
        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask
        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                causal_mask = causal_mask | (bq_t > bq_k + max_causal_window_size)
            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask
        if not causal and self.exact_windowsize:
            max_backward_window_size = self.window_size * self.look_backward
            max_forward_window_size = self.window_size * self.look_forward
            window_mask = (bq_k - max_forward_window_size > bq_t) | (bq_t > bq_k + max_backward_window_size) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)
        if exists(mask):
            batch = mask.shape[0]
            assert b % batch == 0
            h = b // mask.shape[0]
            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim=-1, value=False)
            mask = rearrange(mask, '... (w n) -> (...) w n', w=windows, n=window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h=h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')
        if autopad:
            out = out[:, :orig_seq_len, :]
        out, *_ = unpack(out, packed_shape, '* n d')
        return out


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def process_inputs_chunk(fn, chunks=1, dim=0):

    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l_ = *pre_slices, slice(None, index)
    r_ = *pre_slices, slice(index, None)
    return t[l_], t[r_]


class LSHSelfAttention(nn.Module):

    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, causal=False, dim_head=None, attn_chunks=1, random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True, num_mem_kv=0, one_value_head=False, use_full_attn=False, full_attn_thres=None, return_attn=False, post_attn_dropout=0.0, dropout=0.0, n_local_attn_heads=0, **kwargs):
        super().__init__()
        assert dim_head or dim % heads == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'
        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)
        self.v_head_repeats = heads if one_value_head else 1
        v_dim = dim_heads // self.v_head_repeats
        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, dim)
        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets=attend_across_buckets, allow_duplicate_attention=allow_duplicate_attention, return_attn=return_attn, dropout=dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)
        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)
        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None
        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True, look_forward=1 if not causal else 0)
        self.callback = None

    def forward(self, x, keys=None, input_mask=None, input_attn_mask=None, context_mask=None, pos_emb=None, **kwargs):
        device, dtype = x.device, x.dtype
        b, t, e, h, m, l_h = *x.shape, self.heads, self.num_mem_kv, self.n_local_attn_heads
        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, -1)
        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]
        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres
        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()
        merge_batch_and_heads = partial(merge_dims, 0, 1)
        qk, v = map(merge_heads, (qk, v))
        has_local = l_h > 0
        lsh_h = h - l_h
        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))
        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask
        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask
        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, pos_emb=pos_emb, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.attn_chunks)
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)
        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))
        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)
        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out)


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network (FFN) in Transformer.

    Parameters
    ----------
    d_in:
        The dimension of the input tensor.

    d_hid:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, d_in: 'int', d_hid: 'int', dropout: 'float'=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward processing of the position-wise feed forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor.
        """
        residual = x
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class ReformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, bucket_size, n_hashes, causal, d_ffn, dropout):
        super().__init__()
        self.attn = LSHSelfAttention(dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes, causal=causal)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(self, enc_input: 'torch.Tensor'):
        enc_output = self.attn(enc_input)
        enc_output = self.dropout(enc_output)
        enc_output += enc_input
        enc_output = self.layer_norm(enc_output)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class ReformerEncoder(nn.Module):

    def __init__(self, n_steps, n_layers, d_model, n_heads, bucket_size, n_hashes, causal, d_ffn, dropout):
        super().__init__()
        assert n_steps % (bucket_size * 2) == 0, f'Sequence length ({n_steps}) needs to be divisible by target bucket size  x 2 - {bucket_size * 2}'
        self.enc_layer_stack = nn.ModuleList([ReformerLayer(d_model, n_heads, bucket_size, n_hashes, causal, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, x: 'torch.Tensor'):
        enc_output = x
        for layer in self.enc_layer_stack:
            enc_output = layer(enc_output)
        return enc_output


class _Reformer(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, n_heads, bucket_size, n_hashes, causal, d_ffn, dropout, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False, dropout=dropout)
        self.encoder = ReformerEncoder(n_steps, n_layers, d_model, n_heads, bucket_size, n_hashes, causal, d_ffn, dropout)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class Splitting(nn.Module):

    def __init__(self):
        super().__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        """Returns the odd and even part"""
        return self.even(x), self.odd(x)


class Interactor(nn.Module):

    def __init__(self, in_planes, splitting=True, kernel=5, dropout=0.5, groups=1, hidden_size=1, INN=True):
        super().__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1
            pad_r = self.dilation * self.kernel_size // 2 + 1
        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()
        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1
        size_hidden = self.hidden_size
        modules_P += [nn.ReplicationPad1d((pad_l, pad_r)), nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Dropout(self.dropout), nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups=self.groups), nn.Tanh()]
        modules_U += [nn.ReplicationPad1d((pad_l, pad_r)), nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Dropout(self.dropout), nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups=self.groups), nn.Tanh()]
        modules_phi += [nn.ReplicationPad1d((pad_l, pad_r)), nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Dropout(self.dropout), nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups=self.groups), nn.Tanh()]
        modules_psi += [nn.ReplicationPad1d((pad_l, pad_r)), nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Dropout(self.dropout), nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups=self.groups), nn.Tanh()]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            x_even, x_odd = self.split(x)
        else:
            x_even, x_odd = x
        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            return x_even_update, x_odd_update
        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return c, d


class InteractorLevel(nn.Module):

    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        super().__init__()
        self.level = Interactor(in_planes=in_planes, splitting=True, kernel=kernel, dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        x_even_update, x_odd_update = self.level(x)
        return x_even_update, x_odd_update


class LevelSCINet(nn.Module):

    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.interact = InteractorLevel(in_planes=in_planes, kernel=kernel_size, dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        x_even_update, x_odd_update = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class SCINet_Tree(nn.Module):

    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level
        self.workingblock = LevelSCINet(in_planes=in_planes, kernel_size=kernel_size, dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)
        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size, INN)
            self.SCINet_Tree_even = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size, INN)

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))


class EncoderTree(nn.Module):

    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels = num_levels
        self.SCINet_Tree = SCINet_Tree(in_planes=in_planes, current_level=num_levels - 1, kernel_size=kernel_size, dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        x = self.SCINet_Tree(x)
        return x


class BackboneSCINet(nn.Module):

    def __init__(self, n_out_steps, n_in_steps, n_in_features, d_hidden, n_stacks, n_levels, n_decoder_layers, n_groups, kernel_size=5, dropout: 'float'=0.5, concat_len: 'int'=0, pos_enc: 'bool'=False, modified: 'bool'=True, single_step_output_One: 'bool'=False):
        super().__init__()
        self.n_in_steps = n_in_steps
        self.n_in_features = n_in_features
        self.n_out_steps = n_out_steps
        self.d_hidden = d_hidden
        self.n_levels = n_levels
        self.n_groups = n_groups
        self.modified = modified
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.concat_len = concat_len
        self.pos_enc = pos_enc
        self.single_step_output_One = single_step_output_One
        self.n_decoder_layers = n_decoder_layers
        assert self.n_in_steps % np.power(2, self.n_levels) == 0
        self.blocks1 = EncoderTree(in_planes=self.n_in_features, num_levels=self.n_levels, kernel_size=self.kernel_size, dropout=self.dropout, groups=self.n_groups, hidden_size=self.d_hidden, INN=modified)
        if n_stacks == 2:
            self.blocks2 = EncoderTree(in_planes=self.n_in_features, num_levels=self.n_levels, kernel_size=self.kernel_size, dropout=self.dropout, groups=self.n_groups, hidden_size=self.d_hidden, INN=modified)
        self.stacks = n_stacks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.n_in_steps, self.n_out_steps, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.n_in_steps // 4
        self.div_len = self.n_in_steps // 6
        if self.n_decoder_layers > 1:
            self.projection1 = nn.Linear(self.n_in_steps, self.n_out_steps)
            for layer_idx in range(self.n_decoder_layers - 1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i * self.div_len + self.overlap_len, self.n_in_steps) - i * self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)
        if self.single_step_output_One:
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.n_out_steps, 1, kernel_size=1, bias=False)
                else:
                    self.projection2 = nn.Conv1d(self.n_in_steps + self.n_out_steps, 1, kernel_size=1, bias=False)
        elif self.stacks == 2:
            if self.concat_len:
                self.projection2 = nn.Conv1d(self.concat_len + self.n_out_steps, self.n_out_steps, kernel_size=1, bias=False)
            else:
                self.projection2 = nn.Conv1d(self.n_in_steps + self.n_out_steps, self.n_out_steps, kernel_size=1, bias=False)
        self.pe_hidden_size = n_in_features
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
        return signal

    def forward(self, x):
        if self.pos_enc:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.n_decoder_layers == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0, 2, 1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape, dtype=x.dtype)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:, :, i * self.div_len:min(i * self.div_len + self.overlap_len, self.n_in_steps)]
                    output[:, :, i * self.div_len:(i + 1) * self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0, 2, 1)
        if self.stacks == 1:
            return x, None
        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:, :], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            return x, MidOutPut


class RevIN(nn.Module):
    """RevIN: Reversible Inference Network.

    Parameters
    ----------
    n_features :
        the number of features or channels

    eps :
        a value added for numerical stability

    affine :
        if True, RevIN has learnable affine parameters

    """

    def __init__(self, n_features: 'int', eps: 'float'=1e-09, affine: 'bool'=True):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, missing_mask=None, mode: 'str'='norm'):
        if mode == 'norm':
            x = self._normalize(x, missing_mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.n_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.n_features))

    def _normalize(self, x, missing_mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if missing_mask is None:
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)
        else:
            missing_sum = torch.sum(missing_mask == 1, dim=dim2reduce, keepdim=True) + self.eps
            mean = torch.sum(x, dim=dim2reduce, keepdim=True) / missing_sum
            x_enc = x.masked_fill(missing_mask == 0, 0)
            variance = torch.sum(x_enc * x_enc, dim=dim2reduce, keepdim=True) + self.eps
            stdev = torch.sqrt(variance / missing_sum)
        self.mean = mean.detach()
        self.stdev = stdev.detach()
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class _RevIN_SCINet(nn.Module):

    def __init__(self, n_steps, n_features, n_stacks, n_levels, n_groups, n_decoder_layers, d_hidden, kernel_size, dropout, concat_len, pos_enc: 'bool', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, n_features, with_pos=False, dropout=dropout)
        self.backbone = BackboneSCINet(n_out_steps=n_steps, n_in_steps=n_steps, n_in_features=n_features, d_hidden=d_hidden, n_stacks=n_stacks, n_levels=n_levels, n_decoder_layers=n_decoder_layers, n_groups=n_groups, kernel_size=kernel_size, dropout=dropout, concat_len=concat_len, modified=True, pos_enc=pos_enc, single_step_output_One=False)
        self.revin = RevIN(n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        X = self.revin(X, missing_mask, mode='norm')
        enc_out = self.saits_embedding(X, missing_mask)
        reconstruction, _ = self.backbone(enc_out)
        reconstruction = self.revin(reconstruction, mode='denorm')
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class BackboneSAITS(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float'):
        super().__init__()
        actual_n_features = n_features * 2
        self.embedding_1 = SaitsEmbedding(actual_n_features, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.layer_stack_for_first_block = nn.ModuleList([TransformerEncoderLayer(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout) for _ in range(n_layers)])
        self.reduce_dim_z = nn.Linear(d_model, n_features)
        self.embedding_2 = SaitsEmbedding(actual_n_features, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.layer_stack_for_second_block = nn.ModuleList([TransformerEncoderLayer(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout) for _ in range(n_layers)])
        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def forward(self, X, missing_mask, attn_mask: 'Optional'=None) ->Tuple[torch.Tensor, ...]:
        enc_output = self.embedding_1(X, missing_mask)
        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)
        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = missing_mask * X + (1 - missing_mask) * X_tilde_1
        enc_output = self.embedding_2(X_prime, missing_mask)
        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)
        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))
        copy_second_DMSA_weights = second_DMSA_attn_weights.clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(dim=1)
        if len(copy_second_DMSA_weights.shape) == 4:
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)
        combining_weights = torch.sigmoid(self.weight_combine(torch.cat([missing_mask, copy_second_DMSA_weights], dim=2)))
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        return X_tilde_1, X_tilde_2, X_tilde_3, first_DMSA_attn_weights, second_DMSA_attn_weights, combining_weights


class _SAITS(nn.Module):

    def __init__(self, n_layers: 'int', n_steps: 'int', n_features: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float', diagonal_attention_mask: 'bool'=True, ORT_weight: 'float'=1, MIT_weight: 'float'=1, customized_loss_func: 'Callable'=calc_mae):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.customized_loss_func = customized_loss_func
        self.encoder = BackboneSAITS(n_steps, n_features, n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)

    def forward(self, inputs: 'dict', diagonal_attention_mask: 'bool'=True, training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if training and self.diagonal_attention_mask or not training and diagonal_attention_mask:
            diagonal_attention_mask = 1 - torch.eye(self.n_steps)
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None
        X_tilde_1, X_tilde_2, X_tilde_3, first_DMSA_attn_weights, second_DMSA_attn_weights, combining_weights = self.encoder(X, missing_mask, diagonal_attention_mask)
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3
        results = {'first_DMSA_attn_weights': first_DMSA_attn_weights, 'second_DMSA_attn_weights': second_DMSA_attn_weights, 'combining_weights': combining_weights, 'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            ORT_loss = 0
            ORT_loss += self.customized_loss_func(X_tilde_1, X, missing_mask)
            ORT_loss += self.customized_loss_func(X_tilde_2, X, missing_mask)
            ORT_loss += self.customized_loss_func(X_tilde_3, X, missing_mask)
            ORT_loss /= 3
            ORT_loss = self.ORT_weight * ORT_loss
            MIT_loss = self.MIT_weight * self.customized_loss_func(X_tilde_3, X_ori, indicating_mask)
            loss = ORT_loss + MIT_loss
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class _SCINet(nn.Module):

    def __init__(self, n_steps, n_features, n_stacks, n_levels, n_groups, n_decoder_layers, d_hidden, kernel_size, dropout, concat_len, pos_enc: 'bool', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.saits_embedding = SaitsEmbedding(n_features * 2, n_features, with_pos=False, dropout=dropout)
        self.backbone = BackboneSCINet(n_out_steps=n_steps, n_in_steps=n_steps, n_in_features=n_features, d_hidden=d_hidden, n_stacks=n_stacks, n_levels=n_levels, n_decoder_layers=n_decoder_layers, n_groups=n_groups, kernel_size=kernel_size, dropout=dropout, concat_len=concat_len, modified=True, pos_enc=pos_enc, single_step_output_One=False)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        reconstruction, _ = self.backbone(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class BackboneSegRNN(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', seg_len: 'int'=24, d_model: 'int'=512, dropout: 'float'=0.5):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.seg_len = seg_len
        self.d_model = d_model
        self.dropout = dropout
        if n_steps % seg_len:
            raise ValueError('The argument seg_len is necessary for SegRNN need to be divisible by the sequence length n_steps.')
        self.seg_num = self.n_steps // self.seg_len
        self.valueEmbedding = nn.Sequential(nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.n_features, self.d_model // 2))
        self.predict = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.seg_len))

    def forward(self, x):
        batch_size = x.size(0)
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)
        x = self.valueEmbedding(x.reshape(-1, self.seg_num, self.seg_len))
        _, hn = self.rnn(x)
        pos_emb = torch.cat([self.pos_emb.unsqueeze(0).repeat(self.n_features, 1, 1), self.channel_emb.unsqueeze(1).repeat(1, self.seg_num, 1)], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num).view(1, -1, self.d_model))
        y = self.predict(hy).view(-1, self.n_features, self.n_steps)
        y = y.permute(0, 2, 1) + seq_last
        return y


class _SegRNN(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', seg_len: 'int'=24, d_model: 'int'=512, dropout: 'float'=0.5, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.seg_len = seg_len
        self.d_model = d_model
        self.dropout = dropout
        self.backbone = BackboneSegRNN(n_steps, n_features, seg_len, d_model, dropout)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        reconstruction = self.backbone(X)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class GLU(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):

    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super().__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step))
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.view_as_real(torch.fft.fft(input, dim=1))
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.fft.irfft(torch.view_as_complex(time_step_as_inner), n=time_step_as_inner.shape[1], dim=1)
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class BackboneStemGNN(nn.Module):

    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2):
        super().__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend([StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(nn.Linear(int(self.time_step), int(self.time_step)), nn.LeakyReLU(), nn.Linear(int(self.time_step), self.horizon))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    @staticmethod
    def cheb_polynomial(laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = 2 * torch.matmul(laplacian, second_laplacian) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-07))
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    @staticmethod
    def graph_fft(input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = None
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            if stack_i == 0:
                result = forecast
            else:
                result += forecast
        forecast_result = self.fc(result)
        if forecast_result.size()[-1] == 1:
            return forecast_result.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast_result.permute(0, 2, 1).contiguous(), attention


class _StemGNN(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, n_stacks, d_model, dropout_rate=0.5, leaky_rate=0.2, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False)
        self.backbone = BackboneStemGNN(units=d_model, stack_cnt=n_stacks, time_step=n_steps, multi_layer=n_layers, horizon=n_steps, dropout_rate=dropout_rate, leaky_rate=leaky_rate)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out, _ = self.backbone(enc_out)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BackboneTCN(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class _TCN(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_levels: 'int', d_hidden: 'int', kernel_size: 'int', dropout: 'float'=0, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        channel_sizes = [d_hidden] * n_levels
        self.saits_embedding = SaitsEmbedding(n_features * 2, n_features, with_pos=False, dropout=dropout)
        self.backbone = BackboneTCN(n_features, channel_sizes, kernel_size, dropout)
        self.output_projection = nn.Linear(channel_sizes[-1], n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.backbone(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        reconstruction = self.output_projection(enc_out)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class EvidenceMachineKernel(nn.Module):

    def __init__(self, C, F):
        super().__init__()
        self.C = C
        self.F = 2 ** F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))

    def forward(self, x):
        x = torch.einsum('btc,cf->btcf', x, self.C_weight) + self.C_bias
        return x


class BackboneTEFN(nn.Module):

    def __init__(self, n_steps, n_features, n_fod):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_fod = n_fod
        self.T_model = EvidenceMachineKernel(self.n_steps, self.n_fod)
        self.C_model = EvidenceMachineKernel(self.n_features, self.n_fod)

    def forward(self, X) ->torch.Tensor:
        X = self.T_model(X.permute(0, 2, 1)).permute(0, 2, 1, 3) + self.C_model(X)
        X = torch.einsum('btcf->btc', X)
        return X


class _TEFN(nn.Module):

    def __init__(self, n_steps, n_features, n_fod, apply_nonstationary_norm):
        super().__init__()
        self.seq_len = n_steps
        self.n_fod = n_fod
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.model = BackboneTEFN(n_steps, n_features, n_fod)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, missing_mask)
        out = self.model(X)
        if self.apply_nonstationary_norm:
            out = nonstationary_denorm(out, means, stdev)
        imputed_data = missing_mask * X + (1 - missing_mask) * out
        results = {'imputed_data': imputed_data}
        if training:
            loss = calc_mse(out, inputs['X_ori'], inputs['indicating_mask'])
            results['loss'] = loss
        return results


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-05)


class ResBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TideDecoder(nn.Module):

    def __init__(self, n_steps: 'int', n_pred_steps: 'int', n_pred_features: 'int', n_layers: 'int', d_hidden: 'int', d_feature_encode, dropout: 'float'):
        super().__init__()
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.d_hidden = d_hidden
        res_hidden = d_hidden
        self.decoder_layers = nn.Sequential(*([ResBlock(d_hidden, res_hidden, d_hidden, dropout)] * (n_layers - 1)), ResBlock(d_hidden, res_hidden, n_pred_features * n_pred_steps, dropout))
        self.final_temporal_decoder = ResBlock(n_pred_features + d_feature_encode, d_hidden, 1, dropout)
        self.residual_proj = nn.Linear(self.n_steps, self.n_steps)

    def forward(self, X):
        dec_out = self.decoder_layers(X).reshape(X.shape[0], self.n_pred_steps, self.n_pred_features)
        return dec_out


class TideEncoder(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_flatten: 'int', d_hidden: 'int', dropout: 'float'):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.res_hidden = d_hidden
        self.dropout = dropout
        self.encoder_layers = nn.Sequential(ResBlock(d_flatten, self.res_hidden, self.d_hidden, dropout), *([ResBlock(self.d_hidden, self.res_hidden, self.d_hidden, dropout)] * (self.n_layers - 1)))

    def forward(self, X):
        hidden = self.encoder_layers(X)
        return hidden


class TiDE(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_hidden, d_feature_encode, d_temporal_decoder_hidden, dropout):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        n_output_steps = n_steps
        n_output_features = n_features
        d_flatten = n_steps * n_features + n_output_steps * d_feature_encode
        self.feature_encoder = ResBlock(n_features, d_hidden, d_feature_encode, dropout)
        self.encoder = TideEncoder(n_steps, n_features, n_layers, d_flatten, d_hidden, dropout)
        self.decoder = TideDecoder(n_steps, n_steps, n_output_features, n_layers, d_hidden, d_feature_encode, dropout)
        self.temporal_decoder = ResBlock(n_output_features + d_feature_encode, d_temporal_decoder_hidden, n_output_features, dropout)
        self.residual_proj = nn.Linear(n_features, n_output_features)

    def forward(self, X, dynamic):
        bz = X.shape[0]
        feature = self.feature_encoder(dynamic)
        enc_in = torch.cat([X.reshape(bz, -1), feature.reshape(bz, -1)], dim=-1)
        hidden = self.encoder(enc_in)
        decoded = self.decoder(hidden).reshape(hidden.shape[0], self.n_steps, self.n_features)
        temporal_decoder_input = torch.cat([feature, decoded], dim=-1)
        prediction = self.temporal_decoder(temporal_decoder_input)
        prediction += self.residual_proj(X)
        return prediction


class _TiDE(nn.Module):

    def __init__(self, n_steps, n_features, n_layers, d_model, d_hidden, d_feature_encode, d_temporal_decoder_hidden, dropout, ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_steps = n_steps
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=False, dropout=dropout)
        self.tide = TiDE(n_steps, n_features, n_layers, d_hidden, d_feature_encode, d_temporal_decoder_hidden, dropout)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        reconstruction = self.tide(X, missing_mask)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class FixedEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super().__init__()
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

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super().__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):

    def __init__(self, d_model, freq='h'):
        super().__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, with_pos=True, n_max_steps=1000):
        super().__init__()
        self.with_pos = with_pos
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if with_pos:
            self.position_embedding = PositionalEncoding(d_hid=d_model, n_positions=n_max_steps)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_timestamp=None):
        if x_timestamp is None:
            x = self.value_embedding(x)
            if self.with_pos:
                x += self.position_embedding(x, return_only_pos=True)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_timestamp)
            if self.with_pos:
                x += self.position_embedding(x, return_only_pos=True)
        return self.dropout(x)


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, n_steps, downsampling_window, downsampling_layers):
        super().__init__()
        self.downsampling_layers = torch.nn.ModuleList([nn.Sequential(torch.nn.Linear(n_steps // downsampling_window ** i, n_steps // downsampling_window ** (i + 1)), nn.GELU(), torch.nn.Linear(n_steps // downsampling_window ** (i + 1), n_steps // downsampling_window ** (i + 1))) for i in range(downsampling_layers)])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.downsampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, n_steps, downsampling_window, downsampling_layers):
        super().__init__()
        self.up_sampling_layers = torch.nn.ModuleList([nn.Sequential(torch.nn.Linear(n_steps // downsampling_window ** (i + 1), n_steps // downsampling_window ** i), nn.GELU(), torch.nn.Linear(n_steps // downsampling_window ** i, n_steps // downsampling_window ** i)) for i in reversed(range(downsampling_layers))])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):

    def __init__(self, n_steps, n_pred_steps, d_model, d_ffn, dropout, channel_independence, decomp_method, top_k, moving_avg, downsampling_layers, downsampling_window):
        super().__init__()
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.downsampling_window = downsampling_window
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence
        if decomp_method == 'moving_avg':
            self.decompsition = SeriesDecompositionBlock(moving_avg)
        elif decomp_method == 'dft_decomp':
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError('decompsition is error')
        if channel_independence == 0:
            self.cross_layer = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_ffn), nn.GELU(), nn.Linear(in_features=d_ffn, out_features=d_model))
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(n_steps, downsampling_window, downsampling_layers)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(n_steps, downsampling_window, downsampling_layers)
        self.out_cross_layer = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_ffn), nn.GELU(), nn.Linear(in_features=d_ffn, out_features=d_model))

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class BackboneTimeMixer(nn.Module):

    def __init__(self, task_name, n_steps, n_features, n_pred_steps, n_pred_features, n_layers, d_model, d_ffn, dropout, channel_independence, decomp_method, top_k, moving_avg, downsampling_layers, downsampling_window, downsampling_method, use_future_temporal_feature, embed='fixed', freq='h', n_classes=None):
        super().__init__()
        self.task_name = task_name
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.n_layers = n_layers
        self.channel_independence = channel_independence
        self.downsampling_window = downsampling_window
        self.downsampling_layers = downsampling_layers
        self.downsampling_method = downsampling_method
        self.use_future_temporal_feature = use_future_temporal_feature
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(n_steps, n_pred_steps, d_model, d_ffn, dropout, channel_independence, decomp_method, top_k, moving_avg, downsampling_layers, downsampling_window) for _ in range(n_layers)])
        self.preprocess = SeriesDecompositionBlock(moving_avg)
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding(1, d_model, embed, freq, dropout, with_pos=False)
        else:
            self.enc_embedding = DataEmbedding(n_features, d_model, embed, freq, dropout, with_pos=False)
        self.normalize_layers = torch.nn.ModuleList([RevIN(n_features) for _ in range(downsampling_layers + 1)])
        if task_name == 'long_term_forecast' or task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList([torch.nn.Linear(n_steps // downsampling_window ** i, n_pred_steps) for i in range(downsampling_layers + 1)])
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
                self.out_res_layers = torch.nn.ModuleList([torch.nn.Linear(n_steps // downsampling_window ** i, n_steps // downsampling_window ** i) for i in range(downsampling_layers + 1)])
                self.regression_layers = torch.nn.ModuleList([torch.nn.Linear(n_steps // downsampling_window ** i, n_pred_steps) for i in range(downsampling_layers + 1)])
        if task_name == 'imputation' or task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
        if task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * n_steps, n_classes)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return x_list, None
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return out1_list, out2_list

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.downsampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.downsampling_window, return_indices=False)
        elif self.downsampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.downsampling_window)
        elif self.downsampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=3, padding=padding, stride=self.downsampling_window, padding_mode='circular', bias=False)
        else:
            return x_enc, x_mark_enc
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)
        for i in range(self.downsampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.downsampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.downsampling_window, :]
        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc
        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.n_pred_steps).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.c_out, -1).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc):
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)
        for i in range(self.n_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.n_pred_features, -1).permute(0, 2, 1).contiguous()
        return dec_out


class _TimeMixer(nn.Module):

    def __init__(self, n_layers, n_steps, n_features, d_model, d_ffn, dropout, top_k, channel_independence, decomp_method, moving_avg, downsampling_layers, downsampling_window, apply_nonstationary_norm: 'bool'=False):
        super().__init__()
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.model = BackboneTimeMixer(task_name='imputation', n_steps=n_steps, n_features=n_features, n_pred_steps=None, n_pred_features=n_features, n_layers=n_layers, d_model=d_model, d_ffn=d_ffn, dropout=dropout, channel_independence=channel_independence, decomp_method=decomp_method, top_k=top_k, moving_avg=moving_avg, downsampling_layers=downsampling_layers, downsampling_window=downsampling_window, downsampling_method='avg', use_future_temporal_feature=False)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, missing_mask)
        dec_out = self.model.imputation(X, None)
        if self.apply_nonstationary_norm:
            dec_out = nonstationary_denorm(dec_out, means, stdev)
        imputed_data = missing_mask * X + (1 - missing_mask) * dec_out
        results = {'imputed_data': imputed_data}
        if training:
            loss = calc_mse(dec_out, inputs['X_ori'], inputs['indicating_mask'])
            results['loss'] = loss
        return results


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class InceptionBlockV1(nn.Module):

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):

    def __init__(self, seq_len, pred_len, top_k, d_model, d_ffn, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(InceptionBlockV1(d_model, d_ffn, num_kernels=num_kernels), nn.GELU(), InceptionBlockV1(d_ffn, d_model, num_kernels=num_kernels))

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]])
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class BackboneTimesNet(nn.Module):

    def __init__(self, n_layers, n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels):
        super().__init__()
        self.seq_len = n_steps
        self.n_layers = n_layers
        self.n_pred_steps = n_pred_steps
        self.model = nn.ModuleList([TimesBlock(n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X) ->torch.Tensor:
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](X))
        return enc_out


class _TimesNet(nn.Module):

    def __init__(self, n_layers, n_steps, n_features, top_k, d_model, d_ffn, n_kernels, dropout, apply_nonstationary_norm):
        super().__init__()
        self.seq_len = n_steps
        self.n_layers = n_layers
        self.apply_nonstationary_norm = apply_nonstationary_norm
        self.enc_embedding = DataEmbedding(n_features, d_model, dropout=dropout, n_max_steps=n_steps)
        self.model = BackboneTimesNet(n_layers, n_steps, 0, top_k, d_model, d_ffn, n_kernels)
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        if self.apply_nonstationary_norm:
            X, means, stdev = nonstationary_norm(X, missing_mask)
        input_X = self.enc_embedding(X)
        enc_out = self.model(input_X)
        dec_out = self.projection(enc_out)
        if self.apply_nonstationary_norm:
            dec_out = nonstationary_denorm(dec_out, means, stdev)
        imputed_data = missing_mask * X + (1 - missing_mask) * dec_out
        results = {'imputed_data': imputed_data}
        if training:
            loss = calc_mse(dec_out, inputs['X_ori'], inputs['indicating_mask'])
            results['loss'] = loss
        return results


class _Transformer(nn.Module):

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float', ORT_weight: 'float'=1, MIT_weight: 'float'=1):
        super().__init__()
        self.n_layers = n_layers
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.saits_embedding = SaitsEmbedding(n_features * 2, d_model, with_pos=True, n_max_steps=n_steps, dropout=dropout)
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ffn, dropout, attn_dropout)
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: 'dict', training: 'bool'=True) ->dict:
        X, missing_mask = inputs['X'], inputs['missing_mask']
        input_X = self.saits_embedding(X, missing_mask)
        enc_output, _ = self.encoder(input_X)
        reconstruction = self.output_projection(enc_output)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {'imputed_data': imputed_data}
        if training:
            X_ori, indicating_mask = inputs['X_ori'], inputs['indicating_mask']
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results['ORT_loss'] = ORT_loss
            results['MIT_loss'] = MIT_loss
            results['loss'] = loss
        return results


class UsganDiscriminator(nn.Module):
    """model Discriminator: built on BiRNN

    Parameters
    ----------
    n_features :
        the feature dimension of the input

    rnn_hidden_size :
        the hidden size of the RNN cell

    hint_rate :
        the hint rate for the input imputed_data

    dropout_rate :
        the dropout rate for the output layer

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(self, n_features: 'int', rnn_hidden_size: 'int', hint_rate: 'float'=0.7, dropout_rate: 'float'=0.0):
        super().__init__()
        self.hint_rate = hint_rate
        self.biRNN = nn.GRU(n_features * 2, rnn_hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.read_out = nn.Linear(rnn_hidden_size * 2, n_features)

    def forward(self, imputed_X: 'torch.Tensor', missing_mask: 'torch.Tensor') ->torch.Tensor:
        """Forward processing of USGAN Discriminator.

        Parameters
        ----------
        imputed_X : torch.Tensor,
            The original X with missing parts already imputed.

        missing_mask : torch.Tensor,
            The missing mask of X.

        Returns
        -------
        logits : torch.Tensor,
            the logits of the probability of being the true value.

        """
        device = imputed_X.device
        hint = torch.rand_like(missing_mask, dtype=torch.float, device=device) < self.hint_rate
        hint = hint.int()
        h = hint * missing_mask + (1 - hint) * 0.5
        x_in = torch.cat([imputed_X, h], dim=-1)
        out, _ = self.biRNN(x_in)
        logits = self.read_out(self.dropout(out))
        return logits


class BackboneUSGAN(nn.Module):
    """USGAN model"""

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int', lambda_mse: 'float', hint_rate: 'float'=0.7, dropout_rate: 'float'=0.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.generator = BackboneBRITS(n_steps, n_features, rnn_hidden_size)
        self.discriminator = UsganDiscriminator(n_features, rnn_hidden_size, hint_rate, dropout_rate)

    def forward(self, inputs: 'dict', training_object: 'str'='generator', training: 'bool'=True) ->Tuple[torch.Tensor, ...]:
        imputed_data, f_reconstruction, b_reconstruction, _, _, _, _ = self.generator(inputs)
        if training:
            forward_X = inputs['forward']['X']
            forward_missing_mask = inputs['forward']['missing_mask']
            if training_object == 'discriminator':
                discrimination = self.discriminator(imputed_data.detach(), forward_missing_mask)
                l_D = F.binary_cross_entropy_with_logits(discrimination, forward_missing_mask)
                discrimination_loss = l_D
                return imputed_data, discrimination_loss
            else:
                discrimination = self.discriminator(imputed_data, forward_missing_mask)
                l_G = -F.binary_cross_entropy_with_logits(discrimination, forward_missing_mask, weight=1 - forward_missing_mask)
                reconstruction = (f_reconstruction + b_reconstruction) / 2
                reconstruction_loss = calc_mse(forward_X, reconstruction, forward_missing_mask) + 0.1 * calc_mse(f_reconstruction, b_reconstruction)
                loss_gene = l_G + self.lambda_mse * reconstruction_loss
                generation_loss = loss_gene
                return imputed_data, generation_loss
        else:
            return imputed_data


class _USGAN(nn.Module):
    """USGAN model"""

    def __init__(self, n_steps: 'int', n_features: 'int', rnn_hidden_size: 'int', lambda_mse: 'float', hint_rate: 'float'=0.7, dropout_rate: 'float'=0.0):
        super().__init__()
        self.backbone = BackboneUSGAN(n_steps, n_features, rnn_hidden_size, lambda_mse, hint_rate, dropout_rate)

    def forward(self, inputs: 'dict', training_object: 'str'='generator', training: 'bool'=True) ->dict:
        assert training_object in ['generator', 'discriminator'], 'training_object should be "generator" or "discriminator"'
        results = {}
        if training:
            if training_object == 'discriminator':
                imputed_data, discrimination_loss = self.backbone(inputs, training_object, training)
                loss = discrimination_loss
            else:
                imputed_data, generation_loss = self.backbone(inputs, training_object, training)
                loss = generation_loss
            results['loss'] = loss
        else:
            imputed_data = self.backbone(inputs, training_object, training)
        results['imputed_data'] = imputed_data
        return results


class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attn_opt, cross_attn_opt, d_model, n_heads, d_out, d_ff=None, moving_avg=25, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = MultiHeadAttention(self_attn_opt, d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.cross_attention = MultiHeadAttention(cross_attn_opt, d_model, n_heads, d_model // n_heads, d_model // n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.series_decomp1 = SeriesDecompositionBlock(moving_avg)
        self.series_decomp2 = SeriesDecompositionBlock(moving_avg)
        self.series_decomp3 = SeriesDecompositionBlock(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=d_out, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.series_decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.series_decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.series_decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class CrossformerDecoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x, cross):
        final_predict = None
        i = 0
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
        return final_predict


class CrossformerDecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict


class FourierCrossAttention(AttentionOperator):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', activation='tanh', policy=0, num_heads=8):
        super().__init__()
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index_q), dtype=torch.float))

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag), torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, None]:
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = self.compl_mul1d('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = self.compl_mul1d('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        xqkvw = self.compl_mul1d('bhex,heox->bhox', xqkv_ft, torch.complex(self.weights1, self.weights2))
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return out, None


class InformerDecoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class FourierCrossAttentionW(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh', mode_select_method='random'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag), torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape
        xq = q.permute(0, 3, 2, 1)
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = self.compl_mul1d('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = self.compl_mul1d('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        return out, None


class MultiWaveletCross(AttentionOperator):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64, k=8, ich=512, L=0, base='legendre', mode_select_method='random', initializer=None, activation='tanh', **kwargs):
        super().__init__()
        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-08] = 0
        H1r[np.abs(H1r) < 1e-08] = 0
        G0r[np.abs(G0r) < 1e-08] = 0
        G1r[np.abs(G1r) < 1e-08] = 0
        self.max_item = 3
        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, modes=modes, activation=activation, mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))
        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, None]:
        B, N, H, E = q.shape
        _, S, _, _ = k.shape
        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)
        if N > S:
            zeros = torch.zeros_like(q[:, :N - S, :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)
        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])
        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        for i in range(ns - self.L):
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], attn_mask)[0] + self.attn2(dq[1], dk[1], dv[1], attn_mask)[0]]
            Us += [self.attn3(sq, sk, sv, attn_mask)[0]]
        v = self.attn4(q, k, v, attn_mask)[0]
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return v.contiguous(), None

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FEDformerDecoder(nn.Module):

    def __init__(self, n_steps, n_pred_steps, n_layers, n_heads, d_model, d_ffn, d_output, moving_avg_window_size, dropout, version='Fourier', modes=32, mode_select='random', activation='relu'):
        super().__init__()
        if version == 'Wavelets':
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=d_model, out_channels=d_model, seq_len_q=n_steps // 2 + n_pred_steps, seq_len_kv=n_steps, modes=modes, ich=d_model, base='legendre', activation='tanh')
        elif version == 'Fourier':
            decoder_self_att = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=n_steps // 2 + n_pred_steps, modes=modes, mode_select_method=mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=d_model, out_channels=d_model, seq_len_q=n_steps // 2 + n_pred_steps, seq_len_kv=n_steps, modes=modes, mode_select_method=mode_select, num_heads=n_heads)
        else:
            raise ValueError(f"Unsupported version: {version}. Please choose from ['Wavelets', 'Fourier'].")
        self.decoder = InformerDecoder([AutoformerDecoderLayer(decoder_self_att, decoder_cross_att, d_model, n_heads, d_output, d_ffn, moving_avg=moving_avg_window_size, dropout=dropout, activation=activation) for _ in range(n_layers)], norm_layer=SeasonalLayerNorm(d_model), projection=nn.Linear(d_model, d_output, bias=True))

    def forward(self, X, attn_mask=None):
        dec_out, attns = self.decoder(X, attn_mask)
        return dec_out, attns


class Dense(nn.Module):
    """A simple fully-connected layer."""

    def __init__(self, input_size, output_size, dropout=0.0, bias=True):
        super(Dense, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, output_size, bias=bias), nn.ReLU(), nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

    def forward(self, x):
        return self.layer(x)


class InformerDecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
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

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class SigmoidRange(nn.Module):

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class RegressionHead(nn.Module):

    def __init__(self, n_features, d_model, d_output, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_features * d_model, d_output)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        if self.y_range:
            y = SigmoidRange(*self.y_range)(y)
        return y


class ClassificationHead(nn.Module):

    def __init__(self, n_features, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_features * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class TransformerEncoder(nn.Module):
    """Transformer encoder.

    Parameters
    ----------
    n_layers:
        The number of layers in the encoder.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer in the feed-forward network.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float'):
        super().__init__()
        self.enc_layer_stack = nn.ModuleList([TransformerEncoderLayer(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, x: 'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Forward processing of the encoder.

        Parameters
        ----------
        x:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights_collector:
            A list containing the attention map from each encoder layer.

        """
        attn_weights_collector = []
        enc_output = x
        for layer in self.enc_layer_stack:
            enc_output, attn_weights = layer(enc_output, src_mask)
            attn_weights_collector.append(attn_weights)
        return enc_output, attn_weights_collector


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Parameters
    ----------
    slf_attn_opt:
        The attention operator for the multi-head attention module in the decoder layer.

    enc_attn_opt:
        The attention operator for the encoding multi-head attention module in the decoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, slf_attn_opt: 'AttentionOperator', enc_attn_opt: 'AttentionOperator', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float'=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(slf_attn_opt, d_model, n_heads, d_k, d_v)
        self.enc_attn = MultiHeadAttention(enc_attn_opt, d_model, n_heads, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(self, dec_input: 'torch.Tensor', enc_output: 'torch.Tensor', slf_attn_mask: 'Optional[torch.Tensor]'=None, dec_enc_attn_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward processing of the decoder layer.

        Parameters
        ----------
        dec_input:
            Input tensor.

        enc_output:
            Output tensor from the encoder.

        slf_attn_mask:
            Masking tensor for the self-attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        dec_enc_attn_mask:
            Masking tensor for the encoding attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        dec_output:
            Output tensor.

        dec_slf_attn:
            The self-attention map.

        dec_enc_attn:
            The encoding attention map.

        """
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask, **kwargs)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask, **kwargs)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class TransformerDecoder(nn.Module):
    """Transformer decoder.

    Parameters
    ----------
    n_steps:
        The number of time steps in the input tensor.

    n_features:
        The number of features in the input tensor.

    n_layers:
        The number of layers in the decoder.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer in the feed-forward network.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, n_steps: 'int', n_features: 'int', n_layers: 'int', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float', attn_dropout: 'float'):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        self.layer_stack = nn.ModuleList([TransformerDecoderLayer(ScaledDotProductAttention(d_k ** 0.5, attn_dropout), ScaledDotProductAttention(d_k ** 0.5, attn_dropout), d_model, n_heads, d_k, d_v, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, trg_seq: 'torch.Tensor', enc_output: 'torch.Tensor', trg_mask: 'Optional[torch.Tensor]'=None, src_mask: 'Optional[torch.Tensor]'=None, return_attn_weights: 'bool'=False) ->Union[torch.Tensor, Tuple[torch.Tensor, list, list]]:
        """Forward processing of the decoder.

        Parameters
        ----------
        trg_seq:
            Input tensor.

        enc_output:
            Output tensor from the encoder.

        trg_mask:
            Masking tensor for the self-attention module.

        src_mask:
            Masking tensor for the encoding attention module.

        return_attn_weights:
            Whether to return the attention map.

        Returns
        -------
        dec_output:
            Output tensor.

        dec_slf_attn_collector:
            A list containing the self-attention map from each decoder layer.

        dec_enc_attn_collector:
            A list containing the encoding attention map from each decoder layer.

        """
        trg_seq = self.embedding(trg_seq)
        dec_output = self.dropout(self.position_enc(trg_seq))
        dec_slf_attn_collector = []
        dec_enc_attn_collector = []
        for layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_collector.append(dec_slf_attn)
            dec_enc_attn_collector.append(dec_enc_attn)
        if return_attn_weights:
            return dec_output, dec_slf_attn_collector, dec_enc_attn_collector
        return dec_output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Parameters
    ----------
    attn_opt:
        The attention operator for the multi-head attention module in the encoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, attn_opt: 'AttentionOperator', d_model: 'int', n_heads: 'int', d_k: 'int', d_v: 'int', d_ffn: 'int', dropout: 'float'=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(attn_opt, d_model, n_heads, d_k, d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(self, enc_input: 'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the encoder layer.

        Parameters
        ----------
        enc_input:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights:
            The attention map.

        """
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=src_mask, **kwargs)
        enc_output = self.dropout(enc_output)
        enc_output += enc_input
        enc_output = self.layer_norm(enc_output)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AutoCorrelation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BackboneFreTS,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'embed_size': 4, 'n_pred_steps': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BackboneGPVAE,
     lambda: ([], {'input_dim': 4, 'time_length': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (BackboneSAITS,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'n_layers': 1, 'd_model': 4, 'n_heads': 4, 'd_k': 4, 'd_v': 4, 'd_ffn': 4, 'dropout': 0.5, 'attn_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (BackboneTCN,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BackboneTEFN,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'n_fod': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BackboneVaDER,
     lambda: ([], {'n_steps': 4, 'd_input': 4, 'n_clusters': 4, 'd_rnn_hidden': 4, 'd_mu_stddev': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClassificationHead,
     lambda: ([], {'n_features': 4, 'd_model': 4, 'n_classes': 4, 'head_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1dWithInit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'c_in': 4, 'window_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CrliDecoder,
     lambda: ([], {'n_steps': 4, 'd_input': 4, 'd_output': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (CrliGenerator,
     lambda: ([], {'n_layers': 1, 'n_features': 4, 'd_hidden': 4, 'cell_type': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CsdiDiffusionEmbedding,
     lambda: ([], {'n_diffusion_steps': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {})),
    (CsdiDiffusionModel,
     lambda: ([], {'n_diffusion_steps': 4, 'd_diffusion_embedding': 4, 'd_input': 4, 'd_side': 4, 'n_channels': 4, 'n_heads': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (CsdiResidualBlock,
     lambda: ([], {'d_side': 4, 'n_channels': 4, 'diffusion_embedding_dim': 4, 'nheads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (CustomConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DataEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Decay,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Decay_obs,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Dense,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EvidenceMachineKernel,
     lambda: ([], {'C': 4, 'F': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (FeatureRegression,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Feedforward,
     lambda: ([], {'d_model': 4, 'dim_feedforward': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FourierCrossAttentionW,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'seq_len_q': 4, 'seq_len_kv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FullQKAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (GLU,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GMMLayer,
     lambda: ([], {'d_hidden': 4, 'n_clusters': 4}),
     lambda: ([], {})),
    (GpvaeDecoder,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GpvaeEncoder,
     lambda: ([], {'input_size': 4, 'z_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (HiPPO_LegT,
     lambda: ([], {'N': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ImplicitImputation,
     lambda: ([], {'d_input': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (InceptionBlockV1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Interactor,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InteractorLevel,
     lambda: ([], {'in_planes': 4, 'kernel': 4, 'dropout': 0.5, 'groups': 1, 'hidden_size': 4, 'INN': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (KPLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (KPLayerApprox,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'ndim': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LevelSCINet,
     lambda: ([], {'in_planes': 4, 'kernel_size': 4, 'dropout': 0.5, 'groups': 1, 'hidden_size': 4, 'INN': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MovingAvgBlock,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MrnnFcnRegression,
     lambda: ([], {'feature_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiRNNCell,
     lambda: ([], {'cell_type': 4, 'n_layer': 1, 'd_input': 4, 'd_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbedding,
     lambda: ([], {'d_model': 4, 'patch_len': 4, 'stride': 1, 'padding': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PatchtstEncoder,
     lambda: ([], {'n_layers': 1, 'd_model': 4, 'n_heads': 4, 'd_k': 4, 'd_v': 4, 'd_ffn': 4, 'dropout': 0.5, 'attn_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PeepholeLSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (PositionWiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PredictionHead,
     lambda: ([], {'d_model': 4, 'n_patches': 4, 'n_steps_forecast': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProbAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RegressionHead,
     lambda: ([], {'n_features': 4, 'd_model': 4, 'd_output': 4, 'head_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReparamLargeKernelConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'groups': 1, 'small_kernel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RevIN,
     lambda: ([], {'n_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SaitsEmbedding,
     lambda: ([], {'d_in': 4, 'd_out': 4, 'with_pos': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SaitsLoss,
     lambda: ([], {'ORT_weight': 4, 'MIT_weight': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SeasonalLayerNorm,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SegMerging,
     lambda: ([], {'d_model': 4, 'win_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeriesDecompositionBlock,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4])], {})),
    (SeriesDecompositionMultiBlock,
     lambda: ([], {'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 2, 4])], {})),
    (SigmoidRange,
     lambda: ([], {'low': 4, 'high': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinusoidalEmbeddings,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpectralConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'seq_len': 4, 'modes1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Splitting,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StockBlockLayer,
     lambda: ([], {'time_step': 4, 'unit': 4, 'multi_layer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TemporalDecay,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TiDE,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'n_layers': 1, 'd_hidden': 4, 'd_feature_encode': 4, 'd_temporal_decoder_hidden': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (TideDecoder,
     lambda: ([], {'n_steps': 4, 'n_pred_steps': 4, 'n_pred_features': 4, 'n_layers': 1, 'd_hidden': 4, 'd_feature_encode': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (TideEncoder,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'n_layers': 1, 'd_flatten': 4, 'd_hidden': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimeFeatureEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TokenEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TransformerDecoder,
     lambda: ([], {'n_steps': 4, 'n_features': 4, 'n_layers': 1, 'd_model': 4, 'n_heads': 4, 'd_k': 4, 'd_v': 4, 'd_ffn': 4, 'dropout': 0.5, 'attn_dropout': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoder,
     lambda: ([], {'n_layers': 1, 'd_model': 4, 'n_heads': 4, 'd_k': 4, 'd_v': 4, 'd_ffn': 4, 'dropout': 0.5, 'attn_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (UsganDiscriminator,
     lambda: ([], {'n_features': 4, 'rnn_hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (_YourNewModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

