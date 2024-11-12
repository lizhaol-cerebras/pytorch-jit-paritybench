
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


from typing import List


from torch.utils.data import Dataset


from typing import Any


from typing import Callable


from typing import Dict


from typing import Tuple


from typing import Union


from torch.utils.data import DataLoader


import random


from time import time


import numpy as np


from scipy import signal


from torch import Tensor


import time


import warnings


import torch.nn as nn


from typing import Optional


from pandas import DataFrame


from torch.nn import Module


from torch.nn.functional import pad


import math


from copy import deepcopy


from functools import reduce


from torch.utils.tensorboard import SummaryWriter


def cumulative_normalization(original_signal_mag: 'Tensor', sliding_window_len: 'int'=192) ->Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    eps = 1e-10
    mu = 0
    mu_list = []
    batch_size, frame_num, freq_num = original_signal_mag.shape
    for frame_idx in range(frame_num):
        if frame_idx < sliding_window_len:
            alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
            mu = alp * mu + (1 - alp) * torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
        else:
            current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
            mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)
    XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1)
    return XrMM


def neg_si_sdr(prediction: 'Tensor', target: 'Tensor') ->Tensor:
    return -si_sdr(preds=prediction, target=target)


class McNetIO(Module):
    """网络的input，output以及loss相关的部分

    当前实现是：
        输入：全部通道的STFT系数；输出：参考通道的STFT系数。
    """

    def __init__(self, selected_channels: 'List[int]'=[2, 3, 4, 5], ref_channel: 'int'=5, loss_func: 'Callable'=neg_si_sdr, ft_len: 'int'=512, ft_hop: 'int'=256, sliding_window_len: 'int'=192, use_cumulative_normalization: 'bool'=False) ->None:
        super().__init__()
        self.register_buffer('window', torch.hann_window(ft_len))
        self.ft_len = ft_len
        self.ft_hop = ft_hop
        self.window = torch.hann_window(self.ft_len)
        self.selected_channels = selected_channels
        self.use_cumulative_normalization = use_cumulative_normalization
        self.sliding_window_len = sliding_window_len
        self.ref_chn_idx = selected_channels.index(ref_channel)
        self.loss_func = loss_func
        self._loss_name = loss_func.__name__
        self.freq_num = self.ft_len // 2 + 1

    def prepare_input(self, x: 'Tensor', *args, **kwargs) ->Dict[str, Any]:
        batch_size, chn_num, time = x.shape
        x = x.reshape((batch_size * chn_num, time))
        X = torch.stft(x, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=True)
        X = X.reshape((batch_size, chn_num, X.shape[-2], X.shape[-1]))
        X = X.permute(0, 3, 2, 1)
        frame_num, freq_num = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_chn_idx].clone()
        if self.use_cumulative_normalization == False:
            XrMM = torch.abs(Xr).mean(dim=(1, 2)).reshape(batch_size, 1, 1, 1)
        else:
            XrMM = cumulative_normalization(original_signal_mag=torch.abs(Xr), sliding_window_len=self.sliding_window_len)
        X /= XrMM + 1e-08
        input = torch.view_as_real(X).reshape(batch_size, frame_num, freq_num, chn_num * 2)
        XrMag = torch.abs(X[:, :, :, self.ref_chn_idx]).unsqueeze(-1)
        XMag = torch.abs(X)
        return {'input': input, 'X': X, 'XrMM': XrMM, 'XrMag': XrMag, 'XMag': XMag, 'original_time_len': time}

    def prepare_target(self, x: 'Tensor', yr: 'Tensor', XrMM: 'Tensor', *args, **kwargs) ->Any:
        """prepare target for loss function
        """
        yr_norm = yr / XrMM.reshape(XrMM.shape[0], 1)
        return yr_norm

    def prepare_prediction(self, X: 'Tensor', output: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = torch.view_as_complex(output.reshape(batch_size, frame_num, freq_num, 2))
        output = output.permute(0, 2, 1)
        pred = torch.istft(output, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)
        return pred

    def prepare_time_domain(self, x: 'Tensor', prediction: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        """prepare time domain prediction
        """
        return prediction * XrMM.reshape(XrMM.shape[0], 1)

    def loss(self, prediction: 'Tensor', target: 'Tensor', reduce_batch: 'bool'=False, *args, **kwargs) ->Tensor:
        """loss for prediction and target
        """
        if reduce_batch:
            return self.loss_func(prediction=prediction, target=target).mean()
        else:
            return self.loss_func(prediction=prediction, target=target)

    @property
    def loss_name(self) ->str:
        return self._loss_name


class CCIO(McNetIO):

    def prepare_target(self, x: 'Tensor', yr: 'Tensor', XrMM: 'Tensor', *args, **kwargs) ->Any:
        """prepare target for loss function
        """
        target = torch.stft(yr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=True)
        target /= XrMM.reshape(XrMM.shape[0], 1, 1) + 1e-08
        return target

    def prepare_prediction(self, X: 'Tensor', output: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = output.reshape(batch_size, frame_num, freq_num, 2).permute(0, 2, 1, 3)
        prediction = torch.view_as_complex(output)
        return prediction

    def prepare_time_domain(self, x: 'Tensor', prediction: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        """prepare time domain prediction
        """
        prediction = prediction * XrMM.reshape(XrMM.shape[0], 1, 1)
        wav = torch.istft(prediction, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)
        return wav


def compression_using_hyperbolic_tangent(mask, K=10, C=0.1):
    """
        (-inf, +inf) => [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def get_complex_ideal_ratio_mask(noisy_complex_tensor, clean_complex_tensor):
    assert noisy_complex_tensor.dim() == clean_complex_tensor.dim()
    noisy_real = noisy_complex_tensor[..., 0]
    noisy_imag = noisy_complex_tensor[..., 1]
    clean_real = clean_complex_tensor[..., 0]
    clean_imag = clean_complex_tensor[..., 1]
    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + 1e-10
    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator
    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)
    complex_ratio_mask = compression_using_hyperbolic_tangent(complex_ratio_mask, K=10, C=0.1)
    return complex_ratio_mask


def icIRM(pred_cIRM, noisy_complex_tensor):
    lim = 9.9
    pred_cIRM = lim * (pred_cIRM >= lim) - lim * (pred_cIRM <= -lim) + pred_cIRM * (torch.abs(pred_cIRM) < lim)
    pred_cIRM = -10 * torch.log((10 - pred_cIRM) / (10 + pred_cIRM))
    enhanced_real = pred_cIRM[..., 0] * noisy_complex_tensor[..., 0] - pred_cIRM[..., 1] * noisy_complex_tensor[..., 1]
    enhanced_imag = pred_cIRM[..., 1] * noisy_complex_tensor[..., 0] + pred_cIRM[..., 0] * noisy_complex_tensor[..., 1]
    enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
    return enhanced_complex


class cIRMIO(McNetIO):

    def prepare_target(self, x: 'Tensor', yr: 'Tensor', XrMM: 'Tensor', *args, **kwargs) ->Any:
        """prepare target for loss function
        """
        if len(x.shape) == 3:
            xr = x[:, self.ref_chn_idx, :]
        else:
            assert len(x.shape) == 2
            xr = x
        assert len(yr.shape) == 2
        stft_xr = torch.stft(xr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        stft_yr = torch.stft(yr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        assert stft_xr.dim() == stft_yr.dim()
        target = get_complex_ideal_ratio_mask(noisy_complex_tensor=stft_xr, clean_complex_tensor=stft_yr)
        target = torch.view_as_complex(target)
        return target

    def prepare_prediction(self, X: 'Tensor', output: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = output.reshape(batch_size, frame_num, freq_num, 2).permute(0, 2, 1, 3)
        prediction = torch.view_as_complex(output)
        return prediction

    def prepare_time_domain(self, x: 'Tensor', prediction: 'Tensor', XrMM: 'Tensor', original_time_len: 'int', *args, **kwargs) ->Tensor:
        if len(x.shape) == 3:
            xr = x[:, self.ref_chn_idx, :]
        else:
            assert len(x.shape) == 2
            xr = x
        stft_xr = torch.stft(xr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        prediction = torch.view_as_real(prediction)
        prediction_icIRM = icIRM(pred_cIRM=prediction, noisy_complex_tensor=stft_xr)
        wav = torch.istft(prediction_icIRM, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)
        return wav


class RNN_FC(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', hidden_size: 'int', num_layers: 'int'=2, bidirectional: 'bool'=True, act_funcs: 'Tuple[str, str]'=('SiLU', ''), use_FC: 'bool'=True):
        super().__init__()
        self.sequence_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.sequence_model.flatten_parameters()
        self.use_FC = use_FC
        if self.use_FC:
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)
        self.act_funcs = []
        for act_func in act_funcs:
            if act_func == 'SiLU' or act_func == 'swish':
                self.act_funcs.append(nn.SiLU())
            elif act_func == 'ReLU':
                self.act_funcs.append(nn.ReLU())
            elif act_func == 'Tanh':
                self.act_funcs.append(nn.Tanh())
            elif act_func == None or act_func == '':
                self.act_funcs.append(None)
            else:
                raise NotImplementedError(f'Not implemented activation function {act_func}')

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        o, _ = self.sequence_model(x)
        if self.act_funcs[0] != None:
            o = self.act_funcs[0](o)
        if self.use_FC:
            o = self.fc_output_layer(o)
            if self.act_funcs[1] != None:
                o = self.act_funcs[1](o)
        return o


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RNN_FC,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

