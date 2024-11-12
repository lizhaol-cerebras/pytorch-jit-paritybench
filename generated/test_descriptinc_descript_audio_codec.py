
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


import math


from typing import Union


import numpy as np


from torch import nn


from typing import List


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils import weight_norm


import typing


import warnings


from torch.utils.tensorboard import SummaryWriter


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-09).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResidualUnit(nn.Module):

    def __init__(self, dim: 'int'=16, dilation: 'int'=1):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.Sequential(Snake1d(dim), WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad), Snake1d(dim), WNConv1d(dim, dim, kernel_size=1))

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):

    def __init__(self, dim: 'int'=16, stride: 'int'=1):
        super().__init__()
        self.block = nn.Sequential(ResidualUnit(dim // 2, dilation=1), ResidualUnit(dim // 2, dilation=3), ResidualUnit(dim // 2, dilation=9), Snake1d(dim // 2), WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, d_model: 'int'=64, strides: 'list'=[2, 4, 8, 8], d_latent: 'int'=64):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class DecoderBlock(nn.Module):

    def __init__(self, input_dim: 'int'=16, output_dim: 'int'=8, stride: 'int'=1):
        super().__init__()
        self.block = nn.Sequential(Snake1d(input_dim), WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)), ResidualUnit(output_dim, dilation=1), ResidualUnit(output_dim, dilation=3), ResidualUnit(output_dim, dilation=9))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):

    def __init__(self, input_channel, channels, rates, d_out: 'int'=1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def WNConv2d(*args, **kwargs):
    act = kwargs.pop('act', True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)), WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)), WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)), WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)), WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0))])
        self.conv_post = WNConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode='reflect')
        return x

    def forward(self, x):
        fmap = []
        x = self.pad_to_period(x)
        x = rearrange(x, 'b c (l p) -> b c l p', p=self.period)
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MSD(nn.Module):

    def __init__(self, rate: 'int'=1, sample_rate: 'int'=44100):
        super().__init__()
        self.convs = nn.ModuleList([WNConv1d(1, 16, 15, 1, padding=7), WNConv1d(16, 64, 41, 4, groups=4, padding=20), WNConv1d(64, 256, 41, 4, groups=16, padding=20), WNConv1d(256, 1024, 41, 4, groups=64, padding=20), WNConv1d(1024, 1024, 41, 4, groups=256, padding=20), WNConv1d(1024, 1024, 5, 1, padding=2)])
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data
        fmap = []
        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):

    def __init__(self, window_length: 'int', hop_factor: 'float'=0.25, sample_rate: 'int'=44100, bands: 'list'=BANDS):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(window_length=window_length, hop_length=int(window_length * hop_factor), match_stride=True)
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        ch = 32
        convs = lambda : nn.ModuleList([WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)), WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)), WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)), WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)), WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1))])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, 'b 1 f t c -> (b 1) c t f')
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: 'str'='audio_data', weight: 'float'=1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: 'AudioSignal', y: 'AudioSignal'):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, scaling: 'int'=True, reduction: 'str'='mean', zero_mean: 'int'=True, clip_min: 'int'=None, weight: 'float'=1.0):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: 'AudioSignal', y: 'AudioSignal'):
        eps = 1e-08
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y
        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0
        _references = references - mean_reference
        _estimates = estimates - mean_estimate
        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps
        scale = (references_on_estimates / references_projection).unsqueeze(1) if self.scaling else 1
        e_true = scale * _references
        e_res = _estimates - e_true
        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)
        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)
        if self.reduction == 'mean':
            sdr = sdr.mean()
        elif self.reduction == 'sum':
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(self, window_lengths: 'List[int]'=[2048, 512], loss_fn: 'typing.Callable'=nn.L1Loss(), clamp_eps: 'float'=1e-05, mag_weight: 'float'=1.0, log_weight: 'float'=1.0, pow: 'float'=2.0, weight: 'float'=1.0, match_stride: 'bool'=False, window_type: 'str'=None):
        super().__init__()
        self.stft_params = [STFTParams(window_length=w, hop_length=w // 4, match_stride=match_stride, window_type=window_type) for w in window_lengths]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: 'AudioSignal', y: 'AudioSignal'):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(), y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10())
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(self, n_mels: 'List[int]'=[150, 80], window_lengths: 'List[int]'=[2048, 512], loss_fn: 'typing.Callable'=nn.L1Loss(), clamp_eps: 'float'=1e-05, mag_weight: 'float'=1.0, log_weight: 'float'=1.0, pow: 'float'=2.0, weight: 'float'=1.0, match_stride: 'bool'=False, mel_fmin: 'List[float]'=[0.0, 0.0], mel_fmax: 'List[float]'=[None, None], window_type: 'str'=None):
        super().__init__()
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

    def forward(self, x: 'AudioSignal', y: 'AudioSignal'):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params):
            kwargs = {'window_length': s.window_length, 'hop_length': s.hop_length, 'window_type': s.window_type}
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            loss += self.log_weight * self.loss_fn(x_mels.clamp(self.clamp_eps).pow(self.pow).log10(), y_mels.clamp(self.clamp_eps).pow(self.pow).log10())
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)
        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)
        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: 'int', codebook_size: 'int', codebook_dim: 'int'):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction='none').mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction='none').mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, 'b d t -> (b t) d')
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t()
        indices = rearrange((-dist).max(1)[1], '(b t) -> b t', b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(self, input_dim: 'int'=512, n_codebooks: 'int'=9, codebook_size: 'int'=1024, codebook_dim: 'Union[int, list]'=8, quantizer_dropout: 'float'=0.0):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList([VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)])
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: 'int'=None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []
        latents = []
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers
        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: 'torch.Tensor'):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: 'torch.Tensor'):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])
        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Decoder,
     lambda: ([], {'input_channel': 4, 'channels': 4, 'rates': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DecoderBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 16, 4])], {})),
    (EncoderBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 4])], {})),
    (ResidualUnit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 16, 4])], {})),
    (Snake1d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

