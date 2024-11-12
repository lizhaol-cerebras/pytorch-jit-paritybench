
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


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


import torch


from torch.utils.data import ConcatDataset


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


import numpy as np


from torchvision.datasets import MNIST


from torchvision.transforms import transforms


import math


import typing as tp


import warnings


from torch import nn


from torch.nn import functional as F


from torch.nn.utils import spectral_norm


from torch.nn.utils import weight_norm


from typing import List


import torch.nn.functional as F


from functools import partial


from itertools import cycle


from itertools import zip_longest


from torch.nn import Module


from torch.nn import ModuleList


import torch.nn as nn


import typing as T


from abc import abstractmethod


import matplotlib.pyplot as plt


CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: 'nn.Module', norm: 'str'='none') ->nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        return module


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self, normalized_shape: 'tp.Union[int, tp.List[int], torch.Size]', **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


def get_norm_module(module: 'nn.Module', causal: 'bool'=False, norm: 'str'='none', **norm_kwargs) ->nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, causal: bool=False, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, causal: bool=False, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


def get_extra_padding_for_conv1d(x: 'torch.Tensor', kernel_size: 'int', stride: 'int', padding_total: 'int'=0) ->int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: 'torch.Tensor', paddings: 'tp.Tuple[int, int]', mode: 'str'='zero', value: 'float'=0.0):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True, causal: 'bool'=False, norm: 'str'='none', norm_kwargs: 'tp.Dict[str, tp.Any]'={}, pad_mode: 'str'='reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(f'SConv1d has been initialized with stride > 1 and dilation > 1 (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias, causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


def unpad1d(x: 'torch.Tensor', paddings: 'tp.Tuple[int, int]'):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert padding_left + padding_right <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, causal: 'bool'=False, norm: 'str'='none', trim_right_ratio: 'float'=1.0, norm_kwargs: 'tp.Dict[str, tp.Any]'={}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, '`trim_right_ratio` != 1.0 only makes sense for causal convolutions'
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


def get_2d_padding(kernel_size: 'tp.Tuple[int, int]', dilation: 'tp.Tuple[int, int]'=(1, 1)):
    return (kernel_size[0] - 1) * dilation[0] // 2, (kernel_size[1] - 1) * dilation[1] // 2


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self, filters: 'int', in_channels: 'int'=1, out_channels: 'int'=1, n_fft: 'int'=1024, hop_length: 'int'=256, win_length: 'int'=1024, max_filters: 'int'=1024, filters_scale: 'int'=1, kernel_size: 'tp.Tuple[int, int]'=(3, 9), dilations: 'tp.List'=[1, 2, 4], stride: 'tp.Tuple[int, int]'=(1, 2), normalized: 'bool'=True, norm: 'str'='weight_norm', activation: 'str'='LeakyReLU', activation_params: 'dict'={'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window, normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min(filters_scale ** (i + 1) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)), norm=norm))
            in_chs = out_chs
        out_chs = min(filters_scale ** (len(dilations) + 1) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]), padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels, kernel_size=(kernel_size[0], kernel_size[0]), padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)

    def forward(self, x: 'torch.Tensor'):
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


FeatureMapType = tp.List[torch.Tensor]


LogitsType = torch.Tensor


DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self, filters: 'int', in_channels: 'int'=1, out_channels: 'int'=1, n_ffts: 'tp.List[int]'=[1024, 2048, 512], hop_lengths: 'tp.List[int]'=[256, 512, 128], win_lengths: 'tp.List[int]'=[1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels, n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs) for i in range(len(n_ffts))])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: 'torch.Tensor') ->DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class Discriminator(nn.Module):

    def __init__(self, filters: 'int'=32):
        super().__init__()
        self.msstftd = MultiScaleSTFTDiscriminator(filters=filters)

    def forward(self, y: 'torch.Tensor', y_hat: 'torch.Tensor'):
        y_d_rs, fmap_rs = self.msstftd(y)
        y_d_gs, fmap_gs = self.msstftd(y_hat)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SqueezeExcite(Module):

    def __init__(self, dim, reduction_factor=4, dim_minimum=8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(nn.Conv1d(dim, dim_inner, 1), nn.SiLU(), nn.Conv1d(dim_inner, dim, 1), nn.Sigmoid())

    def forward(self, x):
        seq, device = x.shape[-2], x.device
        cum_sum = x.cumsum(dim=-2)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')
        gate = self.net(cum_mean)
        return x * gate


class Residual(Module):

    def __init__(self, fn: 'Module'):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ChannelTranspose(Module):

    def __init__(self, fn: 'Module'):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c n -> b n c')
        out = self.fn(x, **kwargs) + x
        return rearrange(out, 'b n c -> b c n')


class CausalConv1d(Module):

    def __init__(self, chan_in, chan_out, kernel_size, pad_mode='reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(Module):

    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]
        out = self.conv(x)
        out = out[..., :n * self.upsample_factor]
        return out


class FiLM(Module):

    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)


def exists(val):
    return val is not None


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7, squeeze_excite=False, pad_mode='reflect'):
    return Residual(Sequential(CausalConv1d(chan_in, chan_out, kernel_size, dilation=dilation, pad_mode=pad_mode), nn.ELU(), CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode), nn.ELU(), SqueezeExcite(chan_out) if squeeze_excite else None))


class EncoderBlock(Module):

    def __init__(self, chan_in, chan_out, stride, cycle_dilations=(1, 3, 9), squeeze_excite=False, pad_mode='reflect') ->None:
        super().__init__()
        it = cycle(cycle_dilations)
        residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)
        self.layers = nn.Sequential(residual_unit(chan_in, chan_in, next(it)), residual_unit(chan_in, chan_in, next(it)), residual_unit(chan_in, chan_in, next(it)), CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride))

    def forward(self, x):
        return self.layers(x)


class FiLMDecoderBlock(Module):

    def __init__(self, chan_in, chan_out, stride, cond_channels, cycle_dilations=(1, 3, 9), squeeze_excite=False, pad_mode='reflect'):
        super().__init__()
        even_stride = stride % 2 == 0
        residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)
        it = cycle(cycle_dilations)
        self.upsample = CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride)
        self.films = ModuleList([FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity(), FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity(), FiLM(chan_out, cond_channels) if cond_channels > 0 else nn.Identity()])
        self.residual_units = ModuleList([residual_unit(chan_out, chan_out, next(it)), residual_unit(chan_out, chan_out, next(it)), residual_unit(chan_out, chan_out, next(it))])

    def forward(self, x, cond):
        x = self.upsample(x)
        for film, res in zip(self.films, self.residual_units):
            x = film(x, cond)
            x = res(x)
        return x


class LocalTransformer(Module):

    def __init__(self, *, dim, depth, heads, window_size, dynamic_pos_bias=False, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.layers = ModuleList([])
        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)
        for _ in range(depth):
            self.layers.append(ModuleList([LocalMHA(dim=dim, heads=heads, qk_rmsnorm=True, window_size=window_size, use_rotary_pos_emb=not dynamic_pos_bias, gate_values_per_head=True, use_xpos=True, **kwargs), FeedForward(dim=dim)]))

    def forward(self, x):
        w = self.window_size
        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x


class Encoder(Module):

    def __init__(self, *, channels=64, strides=(2, 4, 5, 8), channel_mults=(2, 4, 8, 16), embedding_dim=64, input_channels=1, cycle_dilations=(1, 3, 9), use_gate_loop_layers=False, squeeze_excite=False, pad_mode='reflect'):
        super().__init__()
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = channels, *layer_channels
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))
        encoder_blocks = []
        for (chan_in, chan_out), layer_stride in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, cycle_dilations, squeeze_excite, pad_mode))
            if use_gate_loop_layers:
                encoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_out, use_heinsen=False))))
        self.encoder = nn.Sequential(CausalConv1d(input_channels, channels, 7, pad_mode=pad_mode), *encoder_blocks, CausalConv1d(layer_channels[-1], embedding_dim, 3, pad_mode=pad_mode))

    def forward(self, x):
        return self.encoder(x)


class SpeakerEncoder(Module):

    def __init__(self, *, channels=32, strides=(2, 4, 5, 8), channel_mults=(2, 4, 8, 16), embedding_dim=64, input_channels=1, cycle_dilations=(1, 3, 9), use_gate_loop_layers=False, squeeze_excite=False, pad_mode='reflect'):
        super().__init__()
        self.encoder = Encoder(channels=channels, strides=strides, channel_mults=channel_mults, embedding_dim=embedding_dim, input_channels=input_channels, cycle_dilations=cycle_dilations, use_gate_loop_layers=use_gate_loop_layers, squeeze_excite=squeeze_excite, pad_mode=pad_mode)
        self.learnable_query = nn.Parameter(torch.randn((1, 1, embedding_dim)), requires_grad=True)

    def forward(self, x, mask=None):
        """speaker encoder

        Args:
            x (torch.FloatTensor): [B, 1, T]
            mask (torch.FloatTensor, optional): mask for attention. Defaults to None.

        Returns:
            spkemb (torch.FloatTensor): [B, C]
        """
        emb = self.encoder(x)
        B, d_k, _ = emb.shape
        query = self.learnable_query.expand(B, -1, -1)
        key = emb
        value = emb.transpose(1, 2)
        score = torch.matmul(query, key)
        score = score / d_k ** 0.5
        if exists(mask):
            score.masked_fill_(mask == 0, -1000000000.0)
        probs = F.softmax(score, dim=-1)
        out = torch.matmul(probs, value)
        out = out.squeeze(1)
        return out


class Decoder(Module):

    def __init__(self, *, channels=40, strides=(2, 4, 5, 8), channel_mults=(2, 4, 8, 16), embedding_dim=64, input_channels=1, cond_channels=64, cycle_dilations=(1, 3, 9), use_gate_loop_layers=False, squeeze_excite=False, pad_mode='reflect', use_local_attn=True, attn_window_size=128, attn_dim_head=64, attn_heads=8, attn_depth=1, attn_xpos_scale_base=None, attn_dynamic_pos_bias=False) ->None:
        super().__init__()
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = channels, *layer_channels
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))
        attn_kwargs = dict(dim=embedding_dim, dim_head=attn_dim_head, heads=attn_heads, depth=attn_depth, window_size=attn_window_size, xpos_scale_base=attn_xpos_scale_base, dynamic_pos_bias=attn_dynamic_pos_bias, prenorm=True, causal=True)
        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None
        self.decoder_init = CausalConv1d(embedding_dim, layer_channels[-1], 7, pad_mode=pad_mode)
        decoder_blocks = []
        for (chan_in, chan_out), layer_stride in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(FiLMDecoderBlock(chan_out, chan_in, layer_stride, cond_channels, cycle_dilations, squeeze_excite, pad_mode))
            if use_gate_loop_layers:
                decoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_in))))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_out = CausalConv1d(channels, input_channels, 7, pad_mode=pad_mode)

    def forward(self, x, cond):
        """StreamVC Decoder

        Args:
            x (_type_): [B, C, N]
            cond (_type_): [B, C, 1]

        Returns:
            _type_: _description_
        """
        if exists(self.decoder_attn):
            x = rearrange(x, 'b c n -> b n c')
            x = self.decoder_attn(x)
            x = rearrange(x, 'b n c -> b c n')
        x = self.decoder_init(x)
        for block in self.decoder_blocks:
            x = block(x, cond)
        out = self.decoder_out(x)
        return out


class StreamVC(Module):

    def __init__(self) ->None:
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder(embedding_dim=64 + 10)
        self.spk_enc = SpeakerEncoder()
        self.norm = nn.LayerNorm(64)
        self.proj = nn.Linear(64, 100)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pitch, energy, train=False):
        h = self.enc(x)
        contents = h.detach()
        spk = self.spk_enc(x)
        logits = None
        if train:
            h = self.norm(h.transpose(1, 2))
            h = self.proj(h)
            logits = self.softmax(h)
        dec_inp = torch.cat([contents, pitch, energy], dim=1)
        out = self.dec(dec_inp, spk)
        return out, logits


class Audio2Mel(nn.Module):

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None):
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window', window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), 'reflect').squeeze(1)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=False, return_complex=True)
        magnitude = torch.abs(fft)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-05))
        return log_mel_spec


class ReconstructionLoss(nn.Module):

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=16000, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.fft = Audio2Mel(n_fft=n_fft, hop_length=hop_length, win_length=win_length, sampling_rate=sampling_rate, n_mel_channels=n_mel_channels, mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    def forward(self, x, G_x):
        S_x = self.fft(x)
        S_G_x = self.fft(G_x)
        loss = F.l1_loss(S_x, S_G_x)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CausalConv1d,
     lambda: ([], {'chan_in': 4, 'chan_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (CausalConvTranspose1d,
     lambda: ([], {'chan_in': 4, 'chan_out': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (FiLM,
     lambda: ([], {'dim': 4, 'dim_cond': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (NormConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (NormConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (NormConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

