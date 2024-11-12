
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


from typing import Callable


from typing import Optional


from typing import Sequence


import torch


import torch.nn.functional as F


from torch import Tensor


from torch import nn


from math import pi


from typing import Any


from typing import Tuple


import torch.nn as nn


from abc import ABC


from abc import abstractmethod


from math import floor


from typing import Union


from torch import Generator


from functools import reduce


from inspect import isfunction


from math import ceil


from math import log2


from typing import Dict


from typing import List


from typing import TypeVar


class MelSpectrogram(nn.Module):

    def __init__(self, n_fft: 'int', hop_length: 'int', win_length: 'int', sample_rate: 'int', n_mel_channels: 'int', center: 'bool'=False, normalize: 'bool'=False, normalize_log: 'bool'=False):
        super().__init__()
        self.padding = (n_fft - hop_length) // 2
        self.normalize = normalize
        self.normalize_log = normalize_log
        self.hop_length = hop_length
        self.to_spectrogram = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, power=None)
        self.to_mel_scale = transforms.MelScale(n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate)

    def forward(self, waveform: 'Tensor') ->Tensor:
        waveform, ps = pack([waveform], '* t')
        waveform = F.pad(waveform, [self.padding] * 2, mode='reflect')
        spectrogram = self.to_spectrogram(waveform)
        spectrogram = torch.abs(spectrogram)
        mel_spectrogram = self.to_mel_scale(spectrogram)
        if self.normalize:
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-05))
        return unpack(mel_spectrogram, ps, '* f l')[0]


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""
    pass


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: 'int', device: 'torch.device'):
        raise NotImplementedError()


class UniformDistribution(Distribution):

    def __init__(self, vmin: 'float'=0.0, vmax: 'float'=1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: 'int', device: 'torch.device'=torch.device('cpu')):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


def extend_dim(x: 'Tensor', dim: 'int'):
    return x.view(*(x.shape + (1,) * (dim - x.ndim)))


class VDiffusion(Diffusion):

    def __init__(self, net: 'nn.Module', sigma_distribution: 'Distribution'=UniformDistribution(), loss_fn: 'Any'=F.mse_loss):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: 'Tensor') ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: 'Tensor', **kwargs) ->Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        noise = torch.randn_like(x)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss_fn(v_pred, v_target)


class ARVDiffusion(Diffusion):

    def __init__(self, net: 'nn.Module', length: 'int', num_splits: 'int', loss_fn: 'Any'=F.mse_loss):
        super().__init__()
        assert length % num_splits == 0, 'length must be divisible by num_splits'
        self.net = net
        self.length = length
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: 'Tensor') ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: 'Tensor', **kwargs) ->Tensor:
        """Returns diffusion loss of v-objective with different noises per split"""
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        assert t == self.length, 'input length must match length'
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        sigmas = repeat(sigmas, 'b 1 n -> b 1 (n l)', l=self.split_length)
        noise = torch.randn_like(x)
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        channels = torch.cat([x_noisy, sigmas], dim=1)
        v_pred = self.net(channels, **kwargs)
        return self.loss_fn(v_pred, v_target)


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: 'int', device: 'torch.device') ->Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):

    def __init__(self, start: 'float'=1.0, end: 'float'=0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: 'int', device: 'Any') ->Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)


class Sampler(nn.Module):
    pass


class VSampler(Sampler):
    diffusion_types = [VDiffusion]

    def __init__(self, net: 'nn.Module', schedule: 'Schedule'=LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: 'Tensor') ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(self, x_noisy: 'Tensor', num_steps: 'int', show_progress: 'bool'=False, **kwargs) ->Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, 'i -> i b', b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f'Sampling (noise={sigmas[i + 1, 0]:.2f})')
        return x_noisy


class ARVSampler(Sampler):

    def __init__(self, net: 'nn.Module', in_channels: 'int', length: 'int', num_splits: 'int'):
        super().__init__()
        assert length % num_splits == 0, 'length must be divisible by num_splits'
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: 'Tensor') ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: 'int', num_steps_per_split: 'int') ->Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, '(n i) -> i b 1 (n l)', b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(self, current: 'Tensor', sigmas: 'Tensor', show_progress: 'bool'=False, **kwargs) ->Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        for i in progress_bar:
            channels = torch.cat([current, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f'Sampling (noise={sigmas[i + 1, 0, 0, 0]:.2f})')
        return current

    def sample_start(self, num_items: 'int', num_steps: 'int', **kwargs) ->Tensor:
        b, c, t = num_items, self.in_channels, self.length
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, 'i -> i b 1 t', b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    @torch.no_grad()
    def forward(self, num_items: 'int', num_chunks: 'int', num_steps: 'int', start: 'Optional[Tensor]'=None, show_progress: 'bool'=False, **kwargs) ->Tensor:
        assert_message = f'required at least {self.num_splits} chunks'
        assert num_chunks >= self.num_splits, assert_message
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        if num_chunks == self.num_splits:
            return start
        b, n = num_items, self.num_splits
        assert num_steps >= n, 'num_steps must be greater than num_splits'
        sigmas = self.get_sigmas_ladder(num_items=b, num_steps_per_split=num_steps // self.num_splits)
        alphas, betas = self.get_alpha_beta(sigmas)
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))
        num_shifts = num_chunks
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)
        for j in progress_bar:
            updated = self.sample_loop(current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs)
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            shape = b, self.in_channels, self.split_length
            chunks += [torch.randn(shape, device=self.device)]
        return torch.cat(chunks[:num_chunks], dim=-1)


class Inpainter(nn.Module):
    pass


T = TypeVar('T')


def default(val: 'Optional[T]', d: 'Union[Callable[..., T], T]') ->T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


class VInpainter(Inpainter):
    diffusion_types = [VDiffusion]

    def __init__(self, net: 'nn.Module', schedule: 'Schedule'=LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule

    def get_alpha_beta(self, sigmas: 'Tensor') ->Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(self, source: 'Tensor', mask: 'Tensor', num_steps: 'int', num_resamples: 'int', show_progress: 'bool'=False, x_noisy: 'Optional[Tensor]'=None, **kwargs) ->Tensor:
        x_noisy = default(x_noisy, lambda : torch.randn_like(source))
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, 'i -> i b', b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)
        for i in progress_bar:
            for r in range(num_resamples):
                v_pred = self.net(x_noisy, sigmas[i], **kwargs)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                j = r == num_resamples - 1
                x_noisy = alphas[i + j] * x_pred + betas[i + j] * noise_pred
                s_noisy = alphas[i + j] * source + betas[i + j] * torch.randn_like(source)
                x_noisy = s_noisy * mask + x_noisy * ~mask
            progress_bar.set_description(f'Inpainting (noise={sigmas[i + 1, 0]:.2f})')
        return x_noisy


def group_dict_by_prefix(prefix: 'str', d: 'Dict') ->Tuple[Dict, Dict]:
    return_dicts: 'Tuple[Dict, Dict]' = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: 'str', d: 'Dict', keep_prefix: 'bool'=False) ->Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix):]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


class DiffusionModel(nn.Module):

    def __init__(self, net_t: 'Callable', diffusion_t: 'Callable'=VDiffusion, sampler_t: 'Callable'=VSampler, loss_fn: 'Callable'=torch.nn.functional.mse_loss, dim: 'int'=1, **kwargs):
        super().__init__()
        diffusion_kwargs, kwargs = groupby('diffusion_', kwargs)
        sampler_kwargs, kwargs = groupby('sampler_', kwargs)
        self.net = net_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, loss_fn=loss_fn, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) ->Tensor:
        return self.diffusion(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs) ->Tensor:
        return self.sampler(*args, **kwargs)


class EncoderBase(nn.Module, ABC):
    """Abstract class for DiffusionAE encoder"""

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.out_channels = None
        self.downsample_factor = None


class AdapterBase(nn.Module, ABC):
    """Abstract class for DiffusionAE encoder"""

    @abstractmethod
    def encode(self, x: 'Tensor') ->Tensor:
        pass

    @abstractmethod
    def decode(self, x: 'Tensor') ->Tensor:
        pass


def closest_power_2(x: 'float') ->int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


class DiffusionAE(DiffusionModel):
    """Diffusion Auto Encoder"""

    def __init__(self, in_channels: 'int', channels: 'Sequence[int]', encoder: 'EncoderBase', inject_depth: 'int', latent_factor: 'Optional[int]'=None, adapter: 'Optional[AdapterBase]'=None, **kwargs):
        context_channels = [0] * len(channels)
        context_channels[inject_depth] = encoder.out_channels
        super().__init__(in_channels=in_channels, channels=channels, context_channels=context_channels, **kwargs)
        self.in_channels = in_channels
        self.encoder = encoder
        self.inject_depth = inject_depth
        self.latent_factor = default(latent_factor, self.encoder.downsample_factor)
        self.adapter = adapter.requires_grad_(False) if exists(adapter) else None

    def forward(self, x: 'Tensor', with_info: 'bool'=False, **kwargs) ->Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        channels = [None] * self.inject_depth + [latent]
        x = self.adapter.encode(x) if exists(self.adapter) else x
        loss = super().forward(x, channels=channels, **kwargs)
        return (loss, info) if with_info else loss

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @torch.no_grad()
    def decode(self, latent: 'Tensor', generator: 'Optional[Generator]'=None, **kwargs) ->Tensor:
        b = latent.shape[0]
        noise_length = closest_power_2(latent.shape[2] * self.latent_factor)
        noise = torch.randn((b, self.in_channels, noise_length), device=latent.device, dtype=latent.dtype, generator=generator)
        channels = [None] * self.inject_depth + [latent]
        out = super().sample(noise, channels=channels, **kwargs)
        return self.adapter.decode(out) if exists(self.adapter) else out


def AppendChannelsPlugin(net_t: 'Callable', channels: 'int'):

    def Net(in_channels: 'int', out_channels: 'Optional[int]'=None, **kwargs) ->nn.Module:
        out_channels = default(out_channels, in_channels)
        net = net_t(in_channels=in_channels + channels, out_channels=out_channels, **kwargs)

        def forward(x: 'Tensor', *args, append_channels: Tensor, **kwargs):
            x = torch.cat([x, append_channels], dim=1)
            return net(x, *args, **kwargs)
        return Module([net], forward)
    return Net


def resample(waveforms: 'Tensor', factor_in: 'int', factor_out: 'int', rolloff: 'float'=0.99, lowpass_filter_width: 'int'=6) ->Tensor:
    """Resamples a waveform using sinc interpolation, adapted from torchaudio"""
    b, _, length = waveforms.shape
    length_target = int(factor_out * length / factor_in)
    d = dict(device=waveforms.device, dtype=waveforms.dtype)
    base_factor = min(factor_in, factor_out) * rolloff
    width = ceil(lowpass_filter_width * factor_in / base_factor)
    idx = torch.arange(-width, width + factor_in, **d)[None, None] / factor_in
    t = torch.arange(0, -factor_out, step=-1, **d)[:, None, None] / factor_out + idx
    t = (t * base_factor).clamp(-lowpass_filter_width, lowpass_filter_width) * pi
    window = torch.cos(t / lowpass_filter_width / 2) ** 2
    scale = base_factor / factor_in
    kernels = torch.where(t == 0, torch.tensor(1.0), t.sin() / t)
    kernels *= window * scale
    waveforms = rearrange(waveforms, 'b c t -> (b c) t')
    waveforms = F.pad(waveforms, (width, width + factor_in))
    resampled = F.conv1d(waveforms[:, None], kernels, stride=factor_in)
    resampled = rearrange(resampled, '(b c) k l -> b c (l k)', b=b)
    return resampled[..., :length_target]


def downsample(waveforms: 'Tensor', factor: 'int', **kwargs) ->Tensor:
    return resample(waveforms, factor_in=factor, factor_out=1, **kwargs)


def randn_like(tensor: 'Tensor', *args, generator: Optional[Generator]=None, **kwargs):
    """randn_like that supports generator"""
    return torch.randn(tensor.shape, *args, generator=generator, **kwargs)


def upsample(waveforms: 'Tensor', factor: 'int', **kwargs) ->Tensor:
    return resample(waveforms, factor_in=1, factor_out=factor, **kwargs)


class DiffusionUpsampler(DiffusionModel):

    def __init__(self, in_channels: 'int', upsample_factor: 'int', net_t: 'Callable', **kwargs):
        self.upsample_factor = upsample_factor
        super().__init__(net_t=AppendChannelsPlugin(net_t, channels=in_channels), in_channels=in_channels, **kwargs)

    def reupsample(self, x: 'Tensor') ->Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample_factor)
        x = upsample(x, factor=self.upsample_factor)
        return x

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        reupsampled = self.reupsample(x)
        return super().forward(x, *args, append_channels=reupsampled, **kwargs)

    @torch.no_grad()
    def sample(self, downsampled: 'Tensor', generator: 'Optional[Generator]'=None, **kwargs) ->Tensor:
        reupsampled = upsample(downsampled, factor=self.upsample_factor)
        noise = randn_like(reupsampled, generator=generator)
        return super().sample(noise, append_channels=reupsampled, **kwargs)


class DiffusionVocoder(DiffusionModel):

    def __init__(self, net_t: 'Callable', mel_channels: 'int', mel_n_fft: 'int', mel_hop_length: 'Optional[int]'=None, mel_win_length: 'Optional[int]'=None, in_channels: 'int'=1, **kwargs):
        mel_hop_length = default(mel_hop_length, floor(mel_n_fft) // 4)
        mel_win_length = default(mel_win_length, mel_n_fft)
        mel_kwargs, kwargs = groupby('mel_', kwargs)
        super().__init__(net_t=AppendChannelsPlugin(net_t, channels=1), in_channels=1, **kwargs)
        self.to_spectrogram = MelSpectrogram(n_fft=mel_n_fft, hop_length=mel_hop_length, win_length=mel_win_length, n_mel_channels=mel_channels, **mel_kwargs)
        self.to_flat = nn.ConvTranspose1d(in_channels=mel_channels, out_channels=1, kernel_size=mel_win_length, stride=mel_hop_length, padding=(mel_win_length - mel_hop_length) // 2, bias=False)

    def forward(self, x: 'Tensor', *args, **kwargs) ->Tensor:
        spectrogram = rearrange(self.to_spectrogram(x), 'b c f l -> (b c) f l')
        spectrogram_flat = self.to_flat(spectrogram)
        x = rearrange(x, 'b c t -> (b c) 1 t')
        return super().forward(x, *args, append_channels=spectrogram_flat, **kwargs)

    @torch.no_grad()
    def sample(self, spectrogram: 'Tensor', generator: 'Optional[Generator]'=None, **kwargs) ->Tensor:
        spectrogram, ps = pack([spectrogram], '* f l')
        spectrogram_flat = self.to_flat(spectrogram)
        noise = randn_like(spectrogram_flat, generator=generator)
        waveform = super().sample(noise, append_channels=spectrogram_flat, **kwargs)
        waveform = rearrange(waveform, '... 1 t -> ... t')
        waveform = unpack(waveform, ps, '* t')[0]
        return waveform


class DiffusionAR(DiffusionModel):

    def __init__(self, in_channels: 'int', length: 'int', num_splits: 'int', diffusion_t: 'Callable'=ARVDiffusion, sampler_t: 'Callable'=ARVSampler, **kwargs):
        super().__init__(in_channels=in_channels + 1, out_channels=in_channels, diffusion_t=diffusion_t, diffusion_length=length, diffusion_num_splits=num_splits, sampler_t=sampler_t, sampler_in_channels=in_channels, sampler_length=length, sampler_num_splits=num_splits, use_time_conditioning=False, use_modulation=False, **kwargs)

