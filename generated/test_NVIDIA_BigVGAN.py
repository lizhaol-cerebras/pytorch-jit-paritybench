
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


from torch import nn


from torch import sin


from torch import pow


from torch.nn import Parameter


import torch.nn as nn


from torch.utils import cpp_extension


import torch.nn.functional as F


import math


from torch.nn import functional as F


from typing import Optional


from typing import Union


from typing import Dict


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


import pandas as pd


import numpy as np


from torch.nn import Conv2d


from torch.nn.utils import spectral_norm


import typing


from typing import List


from typing import Tuple


from scipy.io.wavfile import write


from scipy import signal


from collections import namedtuple


import functools


import random


import torch.utils.data


from time import time


import warnings


import itertools


import time


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DistributedSampler


from torch.utils.data import DataLoader


import torch.multiprocessing as mp


from torch.distributed import init_process_group


from torch.nn.parallel import DistributedDataParallel


import matplotlib


import matplotlib.pylab as plt


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + 1.0 / (alpha + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + 1.0 / (beta + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):

    def __init__(self, cutoff=0.5, half_width=0.6, stride: 'int'=1, padding: 'bool'=True, padding_mode: 'str'='replicate', kernel_size: 'int'=12):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError('Minimum cutoff must be larger than zero.')
        if cutoff > 0.5:
            raise ValueError('A cutoff above 0.5 does not make sense.')
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer('filter', filter)

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return out


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer('filter', filter)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class Activation1d(nn.Module):

    def __init__(self, activation, up_ratio: 'int'=2, down_ratio: 'int'=2, up_kernel_size: 'int'=12, down_kernel_size: 'int'=12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(self, h: 'AttrDict', channels: 'int', kernel_size: 'int'=3, dilation: 'tuple'=(1, 3, 5), activation: 'str'=None):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))) for _ in range(len(dilation))])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        if self.h.get('use_cuda_kernel', False):
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d
        if activation == 'snake':
            self.activations = nn.ModuleList([Activation1d(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(self, h: 'AttrDict', channels: 'int', kernel_size: 'int'=3, dilation: 'tuple'=(1, 3, 5), activation: 'str'=None):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs.apply(init_weights)
        self.num_layers = len(self.convs)
        if self.h.get('use_cuda_kernel', False):
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d
        if activation == 'snake':
            self.activations = nn.ModuleList([Activation1d(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_hparams_from_json(path) ->AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class DiscriminatorP(torch.nn.Module):

    def __init__(self, h: 'AttrDict', period: 'List[int]', kernel_size: 'int'=5, stride: 'int'=3, use_spectral_norm: 'bool'=False):
        super().__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, int(32 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(32 * self.d_mult), int(128 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(128 * self.d_mult), int(512 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(512 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(1024 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, h: 'AttrDict'):
        super().__init__()
        self.mpd_reshapes = h.mpd_reshapes
        None
        self.discriminators = nn.ModuleList([DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes])

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor') ->Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):

    def __init__(self, cfg: 'AttrDict', resolution: 'List[List[int]]'):
        super().__init__()
        self.resolution = resolution
        assert len(self.resolution) == 3, f'MRD layer requires list with len=3, got {self.resolution}'
        self.lrelu_slope = 0.1
        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, 'mrd_use_spectral_norm'):
            None
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, 'mrd_channel_mult'):
            None
            self.d_mult = cfg.mrd_channel_mult
        self.convs = nn.ModuleList([norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 3), padding=(1, 1)))])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x: 'torch.Tensor') ->torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)
        mag = torch.norm(x, p=2, dim=-1)
        return mag


class MultiResolutionDiscriminator(nn.Module):

    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3, f'MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}'
        self.discriminators = nn.ModuleList([DiscriminatorR(cfg, resolution) for resolution in self.resolutions])

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor') ->Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorB(nn.Module):

    def __init__(self, window_length: 'int', channels: 'int'=32, hop_factor: 'float'=0.25, bands: 'Tuple[Tuple[float, float], ...]'=((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None)
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda : nn.ModuleList([weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)))])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x: 'torch.Tensor') ->List[torch.Tensor]:
        x = x - x.mean(dim=-1, keepdims=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-09)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1)
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        return x, fmap


class MultiBandDiscriminator(nn.Module):

    def __init__(self, h):
        """
        Multi-band multi-scale STFT discriminator, with the architecture based on https://github.com/descriptinc/descript-audio-codec.
        and the modified code adapted from https://github.com/gemelo-ai/vocos.
        """
        super().__init__()
        self.fft_sizes = h.get('mbd_fft_sizes', [2048, 1024, 512])
        self.discriminators = nn.ModuleList([DiscriminatorB(window_length=w) for w in self.fft_sizes])

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor') ->Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorCQT(nn.Module):

    def __init__(self, cfg: 'AttrDict', hop_length: 'int', n_octaves: 'int', bins_per_octave: 'int'):
        super().__init__()
        self.cfg = cfg
        self.filters = cfg['cqtd_filters']
        self.max_filters = cfg['cqtd_max_filters']
        self.filters_scale = cfg['cqtd_filters_scale']
        self.kernel_size = 3, 9
        self.dilations = cfg['cqtd_dilations']
        self.stride = 1, 2
        self.in_channels = cfg['cqtd_in_channels']
        self.out_channels = cfg['cqtd_out_channels']
        self.fs = cfg['sampling_rate']
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.cqt_transform = features.cqt.CQT2010v2(sr=self.fs * 2, hop_length=self.hop_length, n_bins=self.bins_per_octave * self.n_octaves, bins_per_octave=self.bins_per_octave, output_format='Complex', pad_mode='constant')
        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=self.kernel_size, padding=self.get_2d_padding(self.kernel_size)))
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(self.in_channels * 2, self.filters, kernel_size=self.kernel_size, padding=self.get_2d_padding(self.kernel_size)))
        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(self.filters_scale ** (i + 1) * self.filters, self.max_filters)
            self.convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=self.kernel_size, stride=self.stride, dilation=(dilation, 1), padding=self.get_2d_padding(self.kernel_size, (dilation, 1)))))
            in_chs = out_chs
        out_chs = min(self.filters_scale ** (len(self.dilations) + 1) * self.filters, self.max_filters)
        self.convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=(self.kernel_size[0], self.kernel_size[0]), padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])))))
        self.conv_post = weight_norm(nn.Conv2d(out_chs, self.out_channels, kernel_size=(self.kernel_size[0], self.kernel_size[0]), padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0]))))
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)
        self.cqtd_normalize_volume = self.cfg.get('cqtd_normalize_volume', False)
        if self.cqtd_normalize_volume:
            None

    def get_2d_padding(self, kernel_size: 'typing.Tuple[int, int]', dilation: 'typing.Tuple[int, int]'=(1, 1)):
        return (kernel_size[0] - 1) * dilation[0] // 2, (kernel_size[1] - 1) * dilation[1] // 2

    def forward(self, x: 'torch.tensor') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        if self.cqtd_normalize_volume:
            x = x - x.mean(dim=-1, keepdims=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-09)
        x = self.resample(x)
        z = self.cqt_transform(x)
        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))
        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(self.conv_pres[i](z[:, :, :, i * self.bins_per_octave:(i + 1) * self.bins_per_octave]))
        latent_z = torch.cat(latent_z, dim=-1)
        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)
        latent_z = self.conv_post(latent_z)
        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):

    def __init__(self, cfg: 'AttrDict'):
        super().__init__()
        self.cfg = cfg
        self.cfg['cqtd_filters'] = self.cfg.get('cqtd_filters', 32)
        self.cfg['cqtd_max_filters'] = self.cfg.get('cqtd_max_filters', 1024)
        self.cfg['cqtd_filters_scale'] = self.cfg.get('cqtd_filters_scale', 1)
        self.cfg['cqtd_dilations'] = self.cfg.get('cqtd_dilations', [1, 2, 4])
        self.cfg['cqtd_in_channels'] = self.cfg.get('cqtd_in_channels', 1)
        self.cfg['cqtd_out_channels'] = self.cfg.get('cqtd_out_channels', 1)
        self.cfg['cqtd_hop_lengths'] = self.cfg.get('cqtd_hop_lengths', [512, 256, 256])
        self.cfg['cqtd_n_octaves'] = self.cfg.get('cqtd_n_octaves', [9, 9, 9])
        self.cfg['cqtd_bins_per_octaves'] = self.cfg.get('cqtd_bins_per_octaves', [24, 36, 48])
        self.discriminators = nn.ModuleList([DiscriminatorCQT(self.cfg, hop_length=self.cfg['cqtd_hop_lengths'][i], n_octaves=self.cfg['cqtd_n_octaves'][i], bins_per_octave=self.cfg['cqtd_bins_per_octaves'][i]) for i in range(len(self.cfg['cqtd_hop_lengths']))])

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor') ->Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """
    Wrapper of chaining multiple discrimiantor architectures.
    Example: combine mbd and cqtd as a single class
    """

    def __init__(self, list_discriminator: 'List[nn.Module]'):
        super().__init__()
        self.discrimiantor = nn.ModuleList(list_discriminator)

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor') ->Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discrimiantor:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(self, sampling_rate: 'int', n_mels: 'List[int]'=[5, 10, 20, 40, 80, 160, 320], window_lengths: 'List[int]'=[32, 64, 128, 256, 512, 1024, 2048], loss_fn: 'typing.Callable'=nn.L1Loss(), clamp_eps: 'float'=1e-05, mag_weight: 'float'=0.0, log_weight: 'float'=1.0, pow: 'float'=1.0, weight: 'float'=1.0, match_stride: 'bool'=False, mel_fmin: 'List[float]'=[0, 0, 0, 0, 0, 0, 0], mel_fmax: 'List[float]'=[None, None, None, None, None, None, None], window_type: 'str'='hann'):
        super().__init__()
        self.sampling_rate = sampling_rate
        STFTParams = namedtuple('STFTParams', ['window_length', 'hop_length', 'window_type', 'match_stride'])
        self.stft_params = [STFTParams(window_length=w, hop_length=w // 4, match_stride=match_stride, window_type=window_type) for w in window_lengths]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(window_type, window_length):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(self, wav, n_mels, fmin, fmax, window_length, hop_length, match_stride, window_type):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape
        if match_stride:
            assert hop_length == window_length // 4, 'For match_stride, hop must equal n_fft // 4'
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0
        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode='reflect')
        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).float()
        stft = torch.stft(wav.reshape(-1, T), n_fft=window_length, hop_length=hop_length, window=window, return_complex=True, center=True)
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)
        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax)
        mel_basis = torch.from_numpy(mel_basis)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)
        return mel_spectrogram

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params):
            kwargs = {'n_mels': n_mels, 'fmin': fmin, 'fmax': fmax, 'window_length': s.window_length, 'hop_length': s.hop_length, 'match_stride': s.match_stride, 'window_type': s.window_type}
            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(x_mels.clamp(min=self.clamp_eps).pow(self.pow)) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(y_mels.clamp(min=self.clamp_eps).pow(self.pow)) / torch.log(torch.tensor(10.0))
            loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Snake,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SnakeBeta,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

