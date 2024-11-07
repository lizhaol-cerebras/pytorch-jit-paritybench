
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


import numpy as np


import torch


import re


import warnings


import math


import random


from torch import nn


import torch.nn.functional as F


import torch.utils.data


from scipy.signal import get_window


from scipy.io.wavfile import read


from torch.nn.utils.rnn import pad_sequence


import copy


import torch.nn as nn


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


from scipy.interpolate import interp1d


from torch.autograd import Variable


from typing import Optional


from scipy.stats import betabinom


from torch.optim.lr_scheduler import *


from torch.optim.lr_scheduler import _LRScheduler


import logging


import matplotlib.pyplot as plt


from typing import List


import time


import itertools


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
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


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.initial_channel, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
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
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Discriminator(nn.Module):

    def __init__(self, config) ->None:
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        if config.pretrained_discriminator:
            state_dict_do = torch.load(config.pretrained_discriminator, map_location='cpu')
            self.mpd.load_state_dict(state_dict_do['mpd'])
            self.msd.load_state_dict(state_dict_do['msd'])
            None

    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        return y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g


def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(data=fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class AlignmentModule(nn.Module):

    def __init__(self, adim, odim, cache_prior=True):
        super().__init__()
        self.cache_prior = cache_prior
        self._cache = {}
        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)
        self.f_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, text_lengths, feats_lengths, x_masks=None):
        text = text.transpose(1, 2)
        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose(1, 2)
        feats = feats.transpose(1, 2)
        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose(1, 2)
        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = torch.norm(dist, p=2, dim=3)
        score = -dist
        if x_masks is not None:
            x_masks = x_masks.unsqueeze(-2)
            score = score.masked_fill(x_masks, -np.inf)
        log_p_attn = F.log_softmax(score, dim=-1)
        bb_prior = self._generate_prior(text_lengths, feats_lengths)
        log_p_attn = log_p_attn + bb_prior
        return log_p_attn

    def _generate_prior(self, text_lengths, feats_lengths, w=1) ->torch.Tensor:
        B = len(text_lengths)
        T_text = text_lengths.max()
        T_feats = feats_lengths.max()
        bb_prior = torch.full((B, T_feats, T_text), fill_value=-np.inf)
        for bidx in range(B):
            T = feats_lengths[bidx].item()
            N = text_lengths[bidx].item()
            key = str(T) + ',' + str(N)
            if self.cache_prior and key in self._cache:
                prob = self._cache[key]
            else:
                alpha = w * np.arange(1, T + 1, dtype=float)
                beta = w * np.array([(T - t + 1) for t in alpha])
                k = np.arange(N)
                batched_k = k[..., None]
                prob = betabinom.logpmf(batched_k, N, alpha, beta)
            if self.cache_prior and key not in self._cache:
                self._cache[key] = prob
            prob = torch.from_numpy(prob).transpose(0, 1)
            bb_prior[bidx, :T, :N] = prob
        return bb_prior


class LayerNorm(torch.nn.LayerNorm):

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class DurationPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1))
        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs.squeeze(-1)

    def forward(self, xs, x_masks=None):
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        return self._forward(xs, x_masks, True)


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False, stochastic_depth_rate=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
        skip_layer = False
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, mask


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class MultiLayeredConv1d(torch.nn.Module):

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.gelu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: 'torch.Tensor'):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiSequential(torch.nn.Sequential):

    def __init__(self, *args, layer_drop_rate=0.0):
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or _probs[idx] >= self.layer_drop_rate:
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)


class Encoder(torch.nn.Module):

    def __init__(self, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0, pos_enc_class=ScaledPositionalEncoding, normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1, stochastic_depth_rate=0.0):
        super(Encoder, self).__init__()
        self.embed = torch.nn.Sequential(pos_enc_class(attention_dim, positional_dropout_rate))
        self.normalize_before = normalize_before
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = [(attention_heads, attention_dim, attention_dropout_rate)] * num_blocks
        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]), positionwise_layer(*positionwise_layer_args), dropout_rate, normalize_before, concat_after, stochastic_depth_rate * float(1 + lnum) / num_blocks))
        self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        xs = self.after_norm(xs)
        return xs, masks


class GaussianUpsampling(torch.nn.Module):

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None, alpha=1.0):
        ds = ds * alpha
        B = ds.size(0)
        device = ds.device
        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1
        if h_masks is None:
            mel_lenghs = torch.sum(ds, dim=-1).int()
            T_feats = mel_lenghs.max().item()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).float()
        if h_masks is not None:
            t = t * h_masks.float()
        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(~d_masks.unsqueeze(1).repeat(1, T_feats, 1), -float('inf'))
        p_attn = torch.softmax(energy, dim=2)
        hs = torch.matmul(p_attn, hs)
        return hs


class VariancePredictor(torch.nn.Module):

    def __init__(self, idim: 'int', n_layers: 'int'=2, n_chans: 'int'=384, kernel_size: 'int'=3, bias: 'bool'=True, dropout_rate: 'float'=0.5):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: 'torch.Tensor', x_masks: 'torch.Tensor'=None) ->torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).

        """
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, 2))
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs.squeeze(-1)


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg)
    return xs_avg


def initialize(model: 'torch.nn.Module', init: 'str'):
    for p in model.parameters():
        if p.dim() > 1:
            if init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(p.data)
            elif init == 'xavier_normal':
                torch.nn.init.xavier_normal_(p.data)
            elif init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity='relu')
            elif init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(p.data, nonlinearity='relu')
            else:
                raise ValueError('Unknown initialization: ' + init)
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            m.reset_parameters()
        if hasattr(m, 'espnet_initialization_fn'):
            m.espnet_initialization_fn()
    if getattr(model, 'encoder', None) and getattr(model.encoder, 'reload_pretrained_parameters', None):
        model.encoder.reload_pretrained_parameters()
    if getattr(model, 'frontend', None) and getattr(model.frontend, 'reload_pretrained_parameters', None):
        model.frontend.reload_pretrained_parameters()
    if getattr(model, 'postencoder', None) and getattr(model.postencoder, 'reload_pretrained_parameters', None):
        model.postencoder.reload_pretrained_parameters()


def viterbi_decode(log_p_attn, text_lengths, feats_lengths):
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device
    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, :feats_lengths[b], :text_lengths[b]]
        viterbi = _monotonic_alignment_search(cur_log_p_attn.detach().cpu().numpy())
        _ds = np.bincount(viterbi)
        ds[b, :len(_ds)] = torch.from_numpy(_ds)
        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


class PromptTTS(nn.Module):

    def __init__(self, config) ->None:
        super().__init__()
        self.encoder = Encoder(attention_dim=config.model.encoder_n_hidden, attention_heads=config.model.encoder_n_heads, linear_units=config.model.encoder_n_hidden * 4, num_blocks=config.model.encoder_n_layers, dropout_rate=config.model.encoder_p_dropout, positional_dropout_rate=config.model.encoder_p_dropout, attention_dropout_rate=config.model.encoder_p_dropout, normalize_before=True, concat_after=False, positionwise_conv_kernel_size=config.model.encoder_kernel_size_conv_mod, stochastic_depth_rate=0.0)
        self.decoder = Encoder(attention_dim=config.model.decoder_n_hidden, attention_heads=config.model.decoder_n_heads, linear_units=config.model.decoder_n_hidden * 4, num_blocks=config.model.decoder_n_layers, dropout_rate=config.model.decoder_p_dropout, positional_dropout_rate=config.model.decoder_p_dropout, attention_dropout_rate=config.model.decoder_p_dropout, normalize_before=True, concat_after=False, positionwise_conv_kernel_size=config.model.decoder_kernel_size_conv_mod, stochastic_depth_rate=0.0)
        self.duration_predictor = DurationPredictor(idim=config.model.encoder_n_hidden, n_layers=config.model.duration_n_layers, n_chans=config.model.variance_n_hidden, kernel_size=config.model.duration_kernel_size, dropout_rate=config.model.duration_p_dropout)
        self.pitch_predictor = VariancePredictor(idim=config.model.encoder_n_hidden, n_layers=config.model.variance_n_layers, n_chans=config.model.variance_n_hidden, kernel_size=config.model.variance_kernel_size, dropout_rate=config.model.variance_p_dropout)
        self.pitch_embed = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=config.model.encoder_n_hidden, kernel_size=config.model.variance_embed_kernel_size, padding=(config.model.variance_embed_kernel_size - 1) // 2), torch.nn.Dropout(config.model.variance_embde_p_dropout))
        self.energy_predictor = VariancePredictor(idim=config.model.encoder_n_hidden, n_layers=2, n_chans=config.model.variance_n_hidden, kernel_size=3, dropout_rate=config.model.variance_p_dropout)
        self.energy_embed = torch.nn.Sequential(torch.nn.Conv1d(in_channels=1, out_channels=config.model.encoder_n_hidden, kernel_size=config.model.variance_embed_kernel_size, padding=(config.model.variance_embed_kernel_size - 1) // 2), torch.nn.Dropout(config.model.variance_embde_p_dropout))
        self.length_regulator = GaussianUpsampling()
        self.alignment_module = AlignmentModule(config.model.encoder_n_hidden, config.n_mels)
        self.to_mel = nn.Linear(in_features=config.model.decoder_n_hidden, out_features=config.n_mels)
        self.spk_tokenizer = nn.Embedding(config.n_speaker, config.model.encoder_n_hidden)
        self.src_word_emb = nn.Embedding(config.n_vocab, config.model.encoder_n_hidden)
        self.embed_projection1 = nn.Linear(config.model.encoder_n_hidden * 2 + config.model.bert_embedding * 2, config.model.encoder_n_hidden)
        initialize(self, 'xavier_uniform')

    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding, inputs_content_embedding, mel_targets=None, output_lengths=None, pitch_targets=None, energy_targets=None, alpha=1.0):
        B = inputs_ling.size(0)
        T = inputs_ling.size(1)
        src_mask = self.get_mask_from_lengths(input_lengths)
        token_embed = self.src_word_emb(inputs_ling)
        x, _ = self.encoder(token_embed, ~src_mask.unsqueeze(-2))
        speaker_embedding = self.spk_tokenizer(inputs_speaker)
        x = torch.concat([x, speaker_embedding.unsqueeze(1).expand(B, T, -1), inputs_style_embedding.unsqueeze(1).expand(B, T, -1), inputs_content_embedding.unsqueeze(1).expand(B, T, -1)], dim=-1)
        x = self.embed_projection1(x)
        if mel_targets is not None:
            log_p_attn = self.alignment_module(text=x, feats=mel_targets.transpose(1, 2), text_lengths=input_lengths, feats_lengths=output_lengths, x_masks=src_mask)
            ds, bin_loss = viterbi_decode(log_p_attn, input_lengths, output_lengths)
            ps = average_by_duration(ds, pitch_targets.squeeze(-1), input_lengths, output_lengths)
            es = average_by_duration(ds, energy_targets.squeeze(-1), input_lengths, output_lengths)
        p_outs = self.pitch_predictor(x, src_mask.unsqueeze(-1))
        e_outs = self.energy_predictor(x, src_mask.unsqueeze(-1))
        if mel_targets is not None:
            d_outs = self.duration_predictor(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(ps.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        else:
            log_p_attn, ds, bin_loss, ps, es = None, None, None, None, None
            d_outs = self.duration_predictor.inference(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(p_outs.unsqueeze(1)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.unsqueeze(1)).transpose(1, 2)
        x = x + p_embs + e_embs
        if mel_targets is not None:
            h_masks_upsampling = self.make_non_pad_mask(output_lengths)
            x = self.length_regulator(x, ds, h_masks_upsampling, ~src_mask, alpha=alpha)
            h_masks = self.make_non_pad_mask(output_lengths).unsqueeze(-2)
        else:
            x = self.length_regulator(x, d_outs, None, ~src_mask)
            mel_lenghs = torch.sum(d_outs, dim=-1).int()
            h_masks = None
        x, _ = self.decoder(x, h_masks)
        x = self.to_mel(x)
        return {'mel_targets': mel_targets, 'dec_outputs': x, 'postnet_outputs': None, 'pitch_predictions': p_outs.squeeze(), 'pitch_targets': ps, 'energy_predictions': e_outs.squeeze(), 'energy_targets': es, 'log_duration_predictions': d_outs, 'duration_targets': ds, 'input_lengths': input_lengths, 'output_lengths': output_lengths, 'log_p_attn': log_p_attn, 'bin_loss': bin_loss}

    def get_mask_from_lengths(self, lengths: 'torch.Tensor') ->torch.Tensor:
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

    def average_utterance_prosody(self, u_prosody_pred: 'torch.Tensor', src_mask: 'torch.Tensor') ->torch.Tensor:
        lengths = (~src_mask * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                None

    def make_pad_mask(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).int()
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

    def make_non_pad_mask(self, length, max_len=None):
        return ~self.make_pad_mask(length, max_len)


def get_segments(x: 'torch.Tensor', start_idxs: 'torch.Tensor', segment_size: 'int'):
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    if t < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - t), 'constant')
    for i, start_idx in enumerate(start_idxs):
        segment = x[i, :, start_idx:start_idx + segment_size]
        segments[i, :, :segment.size(1)] = segment
    return segments


def get_random_segments(x: 'torch.Tensor', x_lengths: 'torch.Tensor', segment_size: 'int'):
    b, d, t = x.size()
    max_start_idx = x_lengths - segment_size
    max_start_idx = torch.clamp(max_start_idx, min=0)
    start_idxs = torch.rand([b]).to(x.device) * max_start_idx
    segments = get_segments(x, start_idxs, segment_size)
    return segments, start_idxs, segment_size


class JETSGenerator(nn.Module):

    def __init__(self, config) ->None:
        super().__init__()
        self.upsample_factor = int(np.prod(config.model.upsample_rates))
        self.segment_size = config.segment_size
        self.am = PromptTTS(config)
        self.generator = HiFiGANGenerator(config.model)
        self.config = config

    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding, inputs_content_embedding, mel_targets=None, output_lengths=None, pitch_targets=None, energy_targets=None, alpha=1.0, cut_flag=True):
        outputs = self.am(inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding, inputs_content_embedding, mel_targets, output_lengths, pitch_targets, energy_targets, alpha)
        if mel_targets is not None and cut_flag:
            z_segments, z_start_idxs, segment_size = get_random_segments(outputs['dec_outputs'].transpose(1, 2), output_lengths, self.segment_size)
        else:
            z_segments = outputs['dec_outputs'].transpose(1, 2)
            z_start_idxs = None
            segment_size = self.segment_size
        wav = self.generator(z_segments)
        outputs['wav_predictions'] = wav
        outputs['z_start_idxs'] = z_start_idxs
        outputs['segment_size'] = segment_size
        return outputs


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


class MelReconLoss(torch.nn.Module):

    def __init__(self, loss_type='mae'):
        super(MelReconLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'mae':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif loss_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError('Unknown loss type: {}'.format(loss_type))

    def forward(self, output_lengths, mel_targets, dec_outputs, postnet_outputs=None):
        """
        mel_targets: B, C, T
        """
        output_masks = get_mask_from_lengths(output_lengths, max_len=mel_targets.size(1))
        output_masks = ~output_masks
        valid_outputs = output_masks.sum()
        mel_loss_ = torch.sum(self.criterion(mel_targets, dec_outputs) * output_masks.unsqueeze(-1)) / (valid_outputs * mel_targets.size(-1))
        if postnet_outputs is not None:
            mel_loss = torch.sum(self.criterion(mel_targets, postnet_outputs) * output_masks.unsqueeze(-1)) / (valid_outputs * mel_targets.size(-1))
        else:
            mel_loss = 0.0
        return mel_loss_, mel_loss


class ForwardSumLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, log_p_attn: 'torch.Tensor', ilens: 'torch.Tensor', olens: 'torch.Tensor', blank_prob: 'float'=np.e ** -1) ->torch.Tensor:
        B = log_p_attn.size(0)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))
        loss = 0
        for bidx in range(B):
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[bidx, :olens[bidx], :ilens[bidx] + 1].unsqueeze(1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(log_probs=cur_log_p_attn_pd, targets=target_seq, input_lengths=olens[bidx:bidx + 1], target_lengths=ilens[bidx:bidx + 1], zero_infinity=True)
        loss = loss / B
        return loss


class ProsodyReconLoss(torch.nn.Module):

    def __init__(self, loss_type='mae'):
        super(ProsodyReconLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'mae':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif loss_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError('Unknown loss type: {}'.format(loss_type))

    def forward(self, input_lengths, duration_targets, pitch_targets, energy_targets, log_duration_predictions, pitch_predictions, energy_predictions):
        input_masks = get_mask_from_lengths(input_lengths, max_len=duration_targets.size(1))
        input_masks = ~input_masks
        valid_inputs = input_masks.sum()
        dur_loss = torch.sum(self.criterion(torch.log(duration_targets.float() + 1), log_duration_predictions) * input_masks) / valid_inputs
        pitch_loss = torch.sum(self.criterion(pitch_targets, pitch_predictions) * input_masks) / valid_inputs
        energy_loss = torch.sum(self.criterion(energy_targets, energy_predictions) * input_masks) / valid_inputs
        return dur_loss, pitch_loss, energy_loss


class TTSLoss(torch.nn.Module):

    def __init__(self, loss_type='mae') ->None:
        super().__init__()
        self.Mel_Loss = MelReconLoss()
        self.Prosodu_Loss = ProsodyReconLoss(loss_type)
        self.ForwardSum_Loss = ForwardSumLoss()

    def forward(self, outputs):
        dec_outputs = outputs['dec_outputs']
        postnet_outputs = outputs['postnet_outputs']
        log_duration_predictions = outputs['log_duration_predictions']
        pitch_predictions = outputs['pitch_predictions']
        energy_predictions = outputs['energy_predictions']
        duration_targets = outputs['duration_targets']
        pitch_targets = outputs['pitch_targets']
        energy_targets = outputs['energy_targets']
        output_lengths = outputs['output_lengths']
        input_lengths = outputs['input_lengths']
        mel_targets = outputs['mel_targets'].transpose(1, 2)
        log_p_attn = outputs['log_p_attn']
        bin_loss = outputs['bin_loss']
        dec_mel_loss, postnet_mel_loss = self.Mel_Loss(output_lengths, mel_targets, dec_outputs, postnet_outputs)
        dur_loss, pitch_loss, energy_loss = self.Prosodu_Loss(input_lengths, duration_targets, pitch_targets, energy_targets, log_duration_predictions, pitch_predictions, energy_predictions)
        forwardsum_loss = self.ForwardSum_Loss(log_p_attn, input_lengths, output_lengths)
        res = {'dec_mel_loss': dec_mel_loss, 'postnet_mel_loss': postnet_mel_loss, 'dur_loss': dur_loss, 'pitch_loss': pitch_loss, 'energy_loss': energy_loss, 'forwardsum_loss': forwardsum_loss, 'bin_loss': bin_loss}
        return res


class ClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels, dropout_rate=0.1) ->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        return self.classifier(self.dropout(pooled_output))


class DownSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class LearnedDownSample(nn.Module):

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):

    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


class StylePretrainLoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, outputs):
        pitch_loss = self.loss(outputs['pitch_outputs'], inputs['pitch'])
        energy_loss = self.loss(outputs['energy_outputs'], inputs['energy'])
        speed_loss = self.loss(outputs['speed_outputs'], inputs['speed'])
        emotion_loss = self.loss(outputs['emotion_outputs'], inputs['emotion'])
        return {'pitch_loss': pitch_loss, 'energy_loss': energy_loss, 'speed_loss': speed_loss, 'emotion_loss': emotion_loss}


class StylePretrainLoss2(StylePretrainLoss):

    def __init__(self) ->None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, outputs):
        res = super().forward(inputs, outputs)
        speaker_loss = self.loss(outputs['speaker_outputs'], inputs['speaker'])
        res['speaker_loss'] = speaker_loss
        return res


class LearnedUpSample(nn.Module):

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class CosineSimilarityLoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def forward(self, output1, output2):
        B = output1.size(0)
        target = torch.ones(B, device=output1.device, requires_grad=False)
        loss = self.loss_fn(output1, output2, target)
        return loss


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ClassificationHead,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (CosineSimilarityLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (DurationPredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256]), torch.rand([4, 4, 4])], {})),
    (GaussianUpsampling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearNorm,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MelReconLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MultiLayeredConv1d,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (MultiSequential,
     lambda: ([], {}),
     lambda: ([], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProsodyReconLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ResBlk,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StyleEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 128, 128])], {})),
    (VariancePredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

