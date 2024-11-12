
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


import torch.nn as nn


from torch.nn import Conv1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import spectral_norm


from scipy.io.wavfile import write


import numpy as np


from torch.nn import ConvTranspose1d


from torch.nn.utils import remove_weight_norm


import math


import random


import torch.utils.data


from scipy.io.wavfile import read


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


LRELU_SLOPE = 0.1


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-07)).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window', use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))), norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))])
        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        with torch.no_grad():
            y = y.squeeze(1)
            y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window'):
        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window), SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window), SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)])

    def forward(self, y, y_hat):
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


Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']


class DWT_1D(nn.Module):

    def __init__(self, pad_type='reflect', wavename='haar', stride=2, in_channels=1, out_channels=None, groups=None, kernel_size=None, trainable=False):
        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band
        a = (self.kernel_size - length_band) // 2
        b = -(self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low
            self.filter_high = self.filter_high
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad=self.pad_sizes, mode=self.pad_type)
        return F.conv1d(input, self.filter_low, stride=self.stride, groups=self.groups), F.conv1d(input, self.filter_high, stride=self.stride, groups=self.groups)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_proj1 = norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.dwt_proj2 = norm_f(Conv2d(1, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv3 = norm_f(Conv1d(8, 1, 1))
        self.dwt_proj3 = norm_f(Conv2d(1, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        b, c, t = x_d1.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x_d1 = F.pad(x_d1, (0, n_pad), 'reflect')
            t = t + n_pad
        x_d1 = x_d1.view(b, c, t // self.period, self.period)
        x_d1 = self.dwt_proj1(x_d1)
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        b, c, t = x_d2.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x_d2 = F.pad(x_d2, (0, n_pad), 'reflect')
            t = t + n_pad
        x_d2 = x_d2.view(b, c, t // self.period, self.period)
        x_d2 = self.dwt_proj2(x_d2)
        x_d3_high1, x_d3_low1 = self.dwt1d(x_d2_high1)
        x_d3_high2, x_d3_low2 = self.dwt1d(x_d2_low1)
        x_d3_high3, x_d3_low3 = self.dwt1d(x_d2_high2)
        x_d3_high4, x_d3_low4 = self.dwt1d(x_d2_low2)
        x_d3 = self.dwt_conv3(torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4], dim=1))
        b, c, t = x_d3.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x_d3 = F.pad(x_d3, (0, n_pad), 'reflect')
            t = t + n_pad
        x_d3 = x_d3.view(b, c, t // self.period, self.period)
        x_d3 = self.dwt_proj3(x_d3)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        i = 0
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            elif i == 1:
                x = torch.cat([x, x_d2], dim=2)
            elif i == 2:
                x = torch.cat([x, x_d3], dim=2)
            else:
                x = x
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class ResWiseMultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super(ResWiseMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11)])

    def forward(self, y, y_hat):
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


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 128, 15, 1, padding=7))
        self.dwt_conv2 = norm_f(Conv1d(4, 128, 41, 2, padding=20))
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        i = 0
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            if i == 1:
                x = torch.cat([x, x_d2], dim=2)
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class ResWiseMultiScaleDiscriminator(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(ResWiseMultiScaleDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_hi, y_lo = self.dwt1d(y)
        y_1 = self.dwt_conv1(torch.cat([y_hi, y_lo], dim=1))
        x_d1_high1, x_d1_low1 = self.dwt1d(y_hat)
        y_hat_1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        x_d2_high1, x_d2_low1 = self.dwt1d(y_hi)
        x_d2_high2, x_d2_low2 = self.dwt1d(y_lo)
        y_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        y_hat_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        for i, d in enumerate(self.discriminators):
            if i == 1:
                y = y_1
                y_hat = y_hat_1
            if i == 2:
                y = y_2
                y_hat = y_hat_2
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5, 7)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[3], padding=get_padding(kernel_size, dilation[3])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class FreGAN(torch.nn.Module):

    def __init__(self, h, top_k=4):
        super(FreGAN, self).__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_rates = h.upsample_rates
        self.up_kernels = h.upsample_kernel_sizes
        self.cond_level = self.num_upsamples - top_k
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1
        self.ups = nn.ModuleList()
        self.cond_up = nn.ModuleList()
        self.res_output = nn.ModuleList()
        upsample_ = 1
        kr = 80
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.up_kernels)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
            if i > self.num_upsamples - top_k:
                self.res_output.append(nn.Sequential(nn.Upsample(scale_factor=u, mode='nearest'), weight_norm(nn.Conv1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), 1))))
            if i >= self.num_upsamples - top_k:
                self.cond_up.append(weight_norm(ConvTranspose1d(kr, h.upsample_initial_channel // 2 ** i, self.up_kernels[i - 1], self.upsample_rates[i - 1], padding=(self.up_kernels[i - 1] - self.upsample_rates[i - 1]) // 2)))
                kr = h.upsample_initial_channel // 2 ** i
            upsample_ *= u
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond_up.apply(init_weights)
        self.res_output.apply(init_weights)

    def forward(self, x):
        mel = x
        x = self.conv_pre(x)
        output = None
        for i in range(self.num_upsamples):
            if i >= self.cond_level:
                mel = self.cond_up[i - self.cond_level](mel)
                x += mel
            if i > self.cond_level:
                if output is None:
                    output = self.res_output[i - self.cond_level - 1](x)
                else:
                    output = self.res_output[i - self.cond_level - 1](output)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if output is not None:
                output = output + x
        x = F.leaky_relu(output)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.cond_up:
            remove_weight_norm(l)
        for l in self.res_output:
            remove_weight_norm(l[1])
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class KernelPredictor(torch.nn.Module):
    """ Kernel predictor for the location-variable convolutions
    """

    def __init__(self, cond_channels, conv_in_channels, conv_out_channels, conv_layers, conv_kernel_size=3, kpnet_hidden_channels=64, kpnet_conv_size=3, kpnet_dropout=0.0, kpnet_nonlinear_activation='LeakyReLU', kpnet_nonlinear_activation_params={'negative_slope': 0.1}):
        """
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        """
        super().__init__()
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        l_b = conv_out_channels * conv_layers
        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = torch.nn.Sequential(torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=(5 - 1) // 2, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params))
        self.residual_conv = torch.nn.Sequential(torch.nn.Dropout(kpnet_dropout), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), torch.nn.Dropout(kpnet_dropout), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), torch.nn.Dropout(kpnet_dropout), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True), getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params))
        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_w, kpnet_conv_size, padding=padding, bias=True)
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_b, kpnet_conv_size, padding=padding, bias=True)

    def forward(self, c):
        """
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        Returns:
        """
        batch, cond_channels, cond_length = c.shape
        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(batch, self.conv_layers, self.conv_in_channels, self.conv_out_channels, self.conv_kernel_size, cond_length)
        bias = b.contiguous().view(batch, self.conv_layers, self.conv_out_channels, cond_length)
        return kernels, bias


class LVCBlock(torch.nn.Module):
    """ the location-variable convolutions
    """

    def __init__(self, in_channels, cond_channels, upsample_ratio, conv_layers=4, conv_kernel_size=3, cond_hop_length=256, kpnet_hidden_channels=64, kpnet_conv_size=3, kpnet_dropout=0.0):
        super().__init__()
        self.cond_hop_length = cond_hop_length
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = torch.nn.ModuleList()
        self.upsample = torch.nn.ConvTranspose1d(in_channels, in_channels, kernel_size=upsample_ratio * 2, stride=upsample_ratio, padding=upsample_ratio // 2 + upsample_ratio % 2, output_padding=upsample_ratio % 2)
        self.kernel_predictor = KernelPredictor(cond_channels=cond_channels, conv_in_channels=in_channels, conv_out_channels=2 * in_channels, conv_layers=conv_layers, conv_kernel_size=conv_kernel_size, kpnet_hidden_channels=kpnet_hidden_channels, kpnet_conv_size=kpnet_conv_size, kpnet_dropout=kpnet_dropout)
        for i in range(conv_layers):
            padding = 3 ** i * int((conv_kernel_size - 1) / 2)
            conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=conv_kernel_size, padding=padding, dilation=3 ** i)
            self.convs.append(conv)

    def forward(self, x, c):
        """ forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        """
        batch, in_channels, in_length = x.shape
        kernels, bias = self.kernel_predictor(c)
        x = F.leaky_relu(x, 0.2)
        x = self.upsample(x)
        for i in range(self.conv_layers):
            y = F.leaky_relu(x, 0.2)
            y = self.convs[i](y)
            y = F.leaky_relu(y, 0.2)
            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            x = x + torch.sigmoid(y[:, :in_channels, :]) * torch.tanh(y[:, in_channels:, :])
        return x

    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        """ perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        """
        batch, in_channels, in_length = x.shape
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == kernel_length * hop_size, 'length of (x, kernel) is not matched'
        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)
        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)
        x = x.unfold(4, kernel_size, 1)
        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.contiguous().view(batch, out_channels, -1)
        return o


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window'):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (KernelPredictor,
     lambda: ([], {'cond_channels': 4, 'conv_in_channels': 4, 'conv_out_channels': 4, 'conv_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LogSTFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ResBlock1,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SpectralConvergengeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

