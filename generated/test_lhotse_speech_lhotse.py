
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


import logging


import re


from functools import lru_cache


from typing import Any


from typing import Generator


from typing import List


from typing import NamedTuple


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import torch


from math import sqrt


from math import ceil


from math import isclose


from typing import Callable


from typing import Dict


from typing import Sequence


import warnings


from typing import Literal


from typing import Set


from abc import ABCMeta


from abc import abstractmethod


from typing import Iterable


from functools import partial


from functools import reduce


from itertools import groupby


import itertools


import random


from collections import defaultdict


from itertools import chain


from itertools import islice


from typing import FrozenSet


from typing import Type


from typing import TypeVar


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.nn import CrossEntropyLoss


from torch import distributed as dist


from torch.utils.data import Dataset


import copy


from copy import deepcopy


from torch.utils.data import Sampler


import time


from queue import Queue


from typing import Deque


import math


import torch.nn.functional as F


from torch.utils.data import IterableDataset


from torch import nn


from abc import ABC


import types


from torch.hub import download_url_to_file


import functools


import inspect


import uuid


from typing import Iterator


import abc


from functools import partialmethod


from numpy.testing import assert_array_almost_equal


from torch.utils.data import DataLoader


from collections import Counter


import torch.utils.data


import torch.testing


from uuid import uuid4


from torch import tensor


import torch.distributed


import torch.multiprocessing as mp


class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization"""

    def __init__(self, feature_dim: 'int'):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer('norm_means', torch.zeros(feature_dim))
        self.register_buffer('norm_stds', torch.ones(feature_dim))

    @classmethod
    def from_cuts(cls, cuts: 'CutSet', max_cuts: 'Optional[int]'=None, extractor: 'Optional[FeatureExtractor]'=None) ->'GlobalMVN':
        stats = cuts.compute_global_feature_stats(max_cuts=max_cuts, extractor=extractor)
        stats = {name: torch.as_tensor(value) for name, value in stats.items()}
        feature_dim, = stats['norm_means'].shape
        global_mvn = cls(feature_dim)
        global_mvn.load_state_dict(stats)
        return global_mvn

    @classmethod
    def from_file(cls, stats_file: 'Pathlike') ->'GlobalMVN':
        stats = torch.load(stats_file)
        feature_dim, = stats['norm_means'].shape
        global_mvn = cls(feature_dim)
        global_mvn.load_state_dict(stats)
        return global_mvn

    def to_file(self, stats_file: 'Pathlike'):
        torch.save(self.state_dict(), stats_file)

    def forward(self, features: 'torch.Tensor', supervision_segments: 'Optional[torch.IntTensor]'=None) ->torch.Tensor:
        return (features - self.norm_means) / self.norm_stds

    def inverse(self, features: 'torch.Tensor') ->torch.Tensor:
        return features * self.norm_stds + self.norm_means


def random_mask_along_batch_axis(tensor: 'torch.Tensor', p: 'float'=0.5) ->torch.Tensor:
    """
    For a given tensor with shape (N, d1, d2, d3, ...), returns a mask with shape (N, 1, 1, 1, ...),
    that randomly masks the samples in a batch.

    E.g. for a 2D input matrix it looks like:

        >>> [[0., 0., 0., ...],
        ...  [1., 1., 1., ...],
        ...  [0., 0., 0., ...]]

    :param tensor: the input tensor.
    :param p: the probability of masking an element.
    """
    mask_shape = (tensor.shape[0],) + tuple(1 for _ in tensor.shape[1:])
    mask = torch.rand(mask_shape) > p
    return mask


T = TypeVar('T')


def schedule_value_for_step(schedule: 'Sequence[Tuple[int, T]]', step: 'int') ->T:
    milestones, values = zip(*schedule)
    assert milestones[0] <= step, f'Cannot determine the scheduled value for step {step} with schedule: {schedule}. Did you forget to add the first part of the schedule for steps below {milestones[0]}?'
    idx = bisect.bisect_right(milestones, step) - 1
    return values[idx]


class RandomizedSmoothing(torch.nn.Module):
    """
    Randomized smoothing - gaussian noise added to an input waveform, or a batch of waveforms.
    The summed audio is clipped to ``[-1.0, 1.0]`` before returning.
    """

    def __init__(self, sigma: 'Union[float, Sequence[Tuple[int, float]]]'=0.1, sample_sigma: 'bool'=True, p: 'float'=0.3):
        """
        RandomizedSmoothing's constructor.

        :param sigma: standard deviation of the gaussian noise. Either a constant float, or a schedule,
            i.e. a list of tuples that specify which value to use from which step.
            For example, ``[(0, 0.01), (1000, 0.1)]`` means that from steps 0-999, the sigma value
            will be 0.01, and from step 1000 onwards, it will be 0.1.
        :param sample_sigma: when ``False``, then sigma is used as the standard deviation in each forward step.
            When ``True``, the standard deviation is sampled from a uniform distribution of
            ``[-sigma, sigma]`` for each forward step.
        :param p: the probability of applying this transform.
        """
        super().__init__()
        self.sigma = sigma
        self.sample_sigma = sample_sigma
        self.p = p
        self.step = 0

    def forward(self, audio: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        if isinstance(self.sigma, float):
            sigma = self.sigma
        else:
            sigma = schedule_value_for_step(self.sigma, self.step)
            self.step += 1
        if self.sample_sigma:
            mask_shape = (audio.shape[0],) + tuple(1 for _ in audio.shape[1:])
            sigma = sigma * (2 * torch.rand(mask_shape) - 1)
        noise = sigma * torch.randn_like(audio)
        noise_mask = random_mask_along_batch_axis(noise, p=1.0 - self.p)
        noise = noise * noise_mask
        return torch.clip(audio + noise, min=-1.0, max=1.0)


def mask_along_axis_optimized(features: 'torch.Tensor', mask_size: 'int', mask_times: 'int', mask_value: 'float', axis: 'int') ->torch.Tensor:
    """
    Apply Frequency and Time masking along axis.
    Frequency and Time masking as described in the SpecAugment paper.

    :param features: input tensor of shape ``(T, F)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    :axis: Axis to apply masking on (1 -> time, 2 -> frequency)
    """
    if axis not in [1, 2]:
        raise ValueError('Only Frequency and Time masking are supported!')
    features = features.unsqueeze(0)
    features = features.reshape([-1] + list(features.size()[-2:]))
    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (features.size(axis) - values)
    mask_starts = min_values.long().squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()
    if axis == 1:
        if mask_times == 1:
            features[:, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            features[:, mask_start:mask_end] = mask_value
    else:
        if mask_times == 1:
            features[:, :, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for mask_start, mask_end in zip(mask_starts, mask_ends):
            features[:, :, mask_start:mask_end] = mask_value
    features = features.squeeze(0)
    return features


def time_warp(features: 'torch.Tensor', factor: 'int') ->torch.Tensor:
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T, F)``
    """
    t = features.size(0)
    if t - factor <= factor + 1:
        return features
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(features[:, :, :center, :], size=(warped, features.size(3)), mode='bicubic', align_corners=False)
    right = torch.nn.functional.interpolate(features[:, :, center:, :], size=(t - warped, features.size(3)), mode='bicubic', align_corners=False)
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)


class SpecAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(self, time_warp_factor: 'Optional[int]'=80, num_feature_masks: 'int'=2, features_mask_size: 'int'=27, num_frame_masks: 'int'=10, frames_mask_size: 'int'=100, max_frames_mask_fraction: 'float'=0.15, p=0.9):
        """
        SpecAugment's constructor.

        :param time_warp_factor: parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
        :param num_feature_masks: how many feature masks should be applied. Set to ``0`` to disable.
        :param features_mask_size: the width of the feature mask (expressed in the number of masked feature bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: the number of masking regions for utterances. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_feature_masks >= 0
        assert num_frame_masks >= 0
        assert features_mask_size > 0
        assert frames_mask_size > 0
        self.time_warp_factor = time_warp_factor
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def forward(self, features: 'torch.Tensor', supervision_segments: 'Optional[torch.IntTensor]'=None, *args, **kwargs) ->torch.Tensor:
        """
        Computes SpecAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param supervision_segments: an int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features`` -- there may be either
            less or more than the batch size.
            The second dimension encoder three kinds of information:
            the sequence index of the corresponding feature matrix in `features`,
            the start frame index, and the number of frames for each segment.
        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        assert len(features.shape) == 3, 'SpecAugment only supports batches of single-channel feature matrices.'
        features = features.clone()
        if supervision_segments is None:
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx])
        else:
            for sequence_idx, start_frame, num_frames in supervision_segments:
                end_frame = start_frame + num_frames
                features[sequence_idx, start_frame:end_frame] = self._forward_single(features[sequence_idx, start_frame:end_frame], warp=True, mask=False)
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx], warp=False, mask=True)
        return features

    def _forward_single(self, features: 'torch.Tensor', warp: 'bool'=True, mask: 'bool'=True) ->torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            return features
        if warp:
            if self.time_warp_factor is not None and self.time_warp_factor >= 1:
                features = time_warp(features, factor=self.time_warp_factor)
        if mask:
            mean = features.mean()
            features = mask_along_axis_optimized(features, mask_size=self.features_mask_size, mask_times=self.num_feature_masks, mask_value=mean, axis=2)
            max_tot_mask_frames = self.max_frames_mask_fraction * features.size(0)
            num_frame_masks = min(self.num_frame_masks, math.ceil(max_tot_mask_frames / self.frames_mask_size))
            max_mask_frames = min(self.frames_mask_size, max_tot_mask_frames // num_frame_masks)
            features = mask_along_axis_optimized(features, mask_size=max_mask_frames, mask_times=num_frame_masks, mask_value=mean, axis=1)
        return features

    def state_dict(self, **kwargs) ->Dict[str, Any]:
        return dict(time_warp_factor=self.time_warp_factor, num_feature_masks=self.num_feature_masks, features_mask_size=self.features_mask_size, num_frame_masks=self.num_frame_masks, frames_mask_size=self.frames_mask_size, max_frames_mask_fraction=self.max_frames_mask_fraction, p=self.p)

    def load_state_dict(self, state_dict: 'Dict[str, Any]'):
        self.time_warp_factor = state_dict.get('time_warp_factor', self.time_warp_factor)
        self.num_feature_masks = state_dict.get('num_feature_masks', self.num_feature_masks)
        self.features_mask_size = state_dict.get('features_mask_size', self.features_mask_size)
        self.num_frame_masks = state_dict.get('num_frame_masks', self.num_frame_masks)
        self.frames_mask_size = state_dict.get('frames_mask_size', self.frames_mask_size)
        self.max_frames_mask_fraction = state_dict.get('max_frames_mask_fraction', self.max_frames_mask_fraction)
        self.p = state_dict.get('p', self.p)


def is_module_available(*modules: str) ->bool:
    """Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).

    Note: "borrowed" from torchaudio:
    https://github.com/pytorch/audio/blob/6bad3a66a7a1c7cc05755e9ee5931b7391d2b94c/torchaudio/_internal/module_utils.py#L9
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def dereverb_wpe_torch(audio: 'torch.Tensor', n_fft: 'int'=512, hop_length: 'int'=128, taps: 'int'=10, delay: 'int'=3, iterations: 'int'=3, statistics_mode: 'str'='full') ->torch.Tensor:
    """
    Applies WPE-based dereverberation using nara_wpe's wpe_v6 function with PyTorch backend.
    The parameter defaults follow the ones in nara_wpe.

    .. caution:: The PyTorch backend is known to sometimes be less stable than the numpy backend.
    """
    if not is_module_available('nara_wpe'):
        raise ImportError("Please install nara_wpe first using 'pip install git+https://github.com/fgnt/nara_wpe'")
    assert audio.ndim == 2
    window = torch.blackman_window(n_fft)
    Y = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)
    Y = Y.permute(1, 0, 2)
    Z = wpe_v6(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode=statistics_mode)
    z = torch.istft(Z.permute(1, 0, 2), n_fft=n_fft, hop_length=hop_length, window=window)
    return z


class DereverbWPE(torch.nn.Module):
    """
    Dereverberation with Weighted Prediction Error (WPE).
    The implementation and default values are borrowed from `nara_wpe` package:
    https://github.com/fgnt/nara_wpe

    The method and library are described in the following paper:
    https://groups.uni-paderborn.de/nt/pubs/2018/ITG_2018_Drude_Paper.pdf
    """

    def __init__(self, n_fft: 'int'=512, hop_length: 'int'=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, audio: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        """
        Expects audio to be 2D or 3D tensor.
        2D means a batch of single-channel audio, shape (B, T).
        3D means a batch of multi-channel audio, shape (B, D, T).
        B => batch size; D => number of channels; T => number of audio samples.
        """
        if audio.ndim == 2:
            return torch.cat([dereverb_wpe_torch(a.unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop_length) for a in audio], dim=0)
        assert audio.ndim == 3
        return torch.stack([dereverb_wpe_torch(a, n_fft=self.n_fft, hop_length=self.hop_length) for a in audio], dim=0)


EPSILON = 1e-10


def _get_log_energy(x: 'torch.Tensor', energy_floor: 'float') ->torch.Tensor:
    """
    Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()
    if energy_floor > 0.0:
        log_energy = torch.max(log_energy, torch.tensor(math.log(energy_floor), dtype=log_energy.dtype))
    return log_energy


def _get_strided_batch(waveform: 'torch.Tensor', window_length: 'int', window_shift: 'int', snip_edges: 'bool') ->torch.Tensor:
    """Given a waveform (2D tensor of size ``(batch_size, num_samples)``,
    it returns a 2D tensor ``(batch_size, num_frames, window_length)``
    representing how the window is shifted along the waveform. Each row is a frame.
    Args:
        waveform (torch.Tensor): Tensor of size ``(batch_size, num_samples)``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
    Returns:
        torch.Tensor: 3D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)
    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        num_frames = (num_samples + window_shift // 2) // window_shift
        new_num_samples = (num_frames - 1) * window_shift + window_length
        npad = new_num_samples - num_samples
        npad_left = int((window_length - window_shift) // 2)
        npad_right = npad - npad_left
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        if npad_right >= 0:
            pad_right = torch.flip(waveform[:, -npad_right:], (1,))
        else:
            pad_right = torch.zeros(0, dtype=waveform.dtype, device=waveform.device)
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)
    strides = waveform.stride(0), window_shift * waveform.stride(1), waveform.stride(1)
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides)


def _get_strided_batch_streaming(waveform: 'torch.Tensor', window_shift: 'int', window_length: 'int', prev_remainder: 'Optional[torch.Tensor]'=None, snip_edges: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    A variant of _get_strided_batch that creates short frames of a batch of audio signals
    in a way suitable for streaming. It accepts a waveform, window size parameters, and
    an optional buffer of previously unused samples. It returns a pair of waveform windows tensor,
    and unused part of the waveform to be passed as ``prev_remainder`` in the next call to this
    function.

    Example usage::

        >>> # get the first buffer of audio and make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ... )
        >>>
        >>> process(frames)  # do sth with the frames
        >>>
        >>> # get the next buffer and use previous remainder to make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ...     prev_remainder=prev_remainder,
        ... )

    :param waveform: A waveform tensor of shape ``(batch_size, num_samples)``.
    :param window_shift: The shift between frames measured in the number of samples.
    :param window_length: The number of samples in each window (frame).
    :param prev_remainder: An optional waveform tensor of shape ``(batch_size, num_samples)``.
        Can be ``None`` which indicates the start of a recording.
    :param snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
        in the file, and the number of frames depends on the frame_length.  If False, the number of frames
        depends only on the frame_shift, and we reflect the data at the ends.
    :return: a pair of tensors with shapes ``(batch_size, num_frames, window_length)`` and
        ``(batch_size, remainder_len)``.
    """
    assert window_shift <= window_length
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    if prev_remainder is None:
        if not snip_edges:
            npad_left = int((window_length - window_shift) // 2)
            pad_left = torch.flip(waveform[:, :npad_left], (1,))
            waveform = torch.cat((pad_left, waveform), dim=1)
    else:
        assert prev_remainder.dim() == 2
        assert prev_remainder.size(0) == batch_size
        waveform = torch.cat((prev_remainder, waveform), dim=1)
    num_samples = waveform.size(-1)
    if snip_edges:
        if num_samples < window_length:
            return torch.empty((batch_size, 0, 0)), waveform
        num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        window_remainder = window_length - window_shift
        num_frames = (num_samples - window_remainder) // window_shift
    remainder = waveform[:, num_frames * window_shift:]
    strides = waveform.stride(0), window_shift * waveform.stride(1), waveform.stride(1)
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides), remainder


BLACKMAN = 'blackman'


HAMMING = 'hamming'


HANNING = 'hanning'


POVEY = 'povey'


RECTANGULAR = 'rectangular'


def create_frame_window(window_size, window_type: 'str'='povey', blackman_coeff=0.42):
    """Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return blackman_coeff - 0.5 * torch.cos(a * window_function) + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
    else:
        raise Exception(f'Invalid window type: {window_type}')


class Wav2Win(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and partition them into overlapping frames (of audio samples).
    Note: no feature extraction happens in here, the output is still a time-domain signal.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Win()
        >>> t(x).shape
        torch.Size([1, 100, 400])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, window_length)``.
    When ``return_log_energy==True``, returns a tuple where the second element
    is a log-energy tensor of shape ``(batch_size, num_frames)``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, pad_length: 'Optional[int]'=None, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, return_log_energy: 'bool'=False) ->None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.return_log_energy = return_log_energy
        if snip_edges:
            warnings.warn('Setting snip_edges=True is generally incompatible with Lhotse -- you might experience mismatched duration/num_frames errors.')
        N = int(math.floor(frame_length * sampling_rate))
        self._length = N
        self._shift = int(math.floor(frame_shift * sampling_rate))
        self._window = nn.Parameter(create_frame_window(N, window_type=window_type), requires_grad=False)
        self.pad_length = N if pad_length is None else pad_length
        assert self.pad_length >= N, f'pad_length (or fft_length) = {pad_length} cannot be smaller than N = {N}'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{}(sampling_rate={}, frame_length={}, frame_shift={}, pad_length={}, remove_dc_offset={}, preemph_coeff={}, window_type={} dither={}, snip_edges={}, energy_floor={}, raw_energy={}, return_log_energy={})'.format(self.__class__.__name__, self.sampling_rate, self.frame_length, self.frame_shift, self.pad_length, self.remove_dc_offset, self.preemph_coeff, self.window_type, self.dither, self.snip_edges, self.energy_floor, self.raw_energy, self.return_log_energy)
        return s

    def _forward_strided(self, x_strided: 'torch.Tensor') ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.remove_dc_offset:
            mu = torch.mean(x_strided, dim=2, keepdim=True)
            x_strided = x_strided - mu
        log_energy: 'Optional[torch.Tensor]' = None
        if self.return_log_energy and self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)
        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(x_strided, (1, 0), mode='replicate')
            x_strided = x_strided - self.preemph_coeff * x_offset[:, :, :-1]
        x_strided = x_strided * self._window
        if self.pad_length != self._length:
            pad = self.pad_length - self._length
            x_strided = torch.nn.functional.pad(x_strided.unsqueeze(1), [0, pad], mode='constant', value=0.0).squeeze(1)
        if self.return_log_energy and not self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)
        return x_strided, log_energy

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n
        x_strided = _get_strided_batch(x, self._length, self._shift, self.snip_edges)
        return self._forward_strided(x_strided)

    @torch.jit.export
    def online_inference(self, x: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None) ->Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        The same as the ``forward()`` method, except it accepts an extra argument with the
        remainder waveform from the previous call of ``online_inference()``, and returns
        a tuple of ``((frames, log_energy), remainder)``.
        """
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n
        x_strided, remainder = _get_strided_batch_streaming(x, window_length=self._length, window_shift=self._shift, prev_remainder=context, snip_edges=self.snip_edges)
        x_strided, log_energy = self._forward_strided(x_strided)
        return (x_strided, log_energy), remainder


Seconds = float


def next_power_of_2(x: 'int') ->int:
    """
    Returns the smallest power of 2 that is greater than x.

    Original source: TorchAudio (torchaudio/compliance/kaldi.py)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class Wav2FFT(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The output is a complex-valued tensor.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2FFT()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``
    with dtype ``torch.complex64``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, round_to_power_of_two: 'bool'=True, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, use_energy: 'bool'=True) ->None:
        super().__init__()
        self.use_energy = use_energy
        N = int(math.floor(frame_length * sampling_rate))
        self.fft_length = next_power_of_2(N) if round_to_power_of_two else N
        self.wav2win = Wav2Win(sampling_rate, frame_length, frame_shift, pad_length=self.fft_length, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, return_log_energy=use_energy)

    @property
    def sampling_rate(self) ->int:
        return self.wav2win.sampling_rate

    @property
    def frame_length(self) ->Seconds:
        return self.wav2win.frame_length

    @property
    def frame_shift(self) ->Seconds:
        return self.wav2win.frame_shift

    @property
    def remove_dc_offset(self) ->bool:
        return self.wav2win.remove_dc_offset

    @property
    def preemph_coeff(self) ->float:
        return self.wav2win.preemph_coeff

    @property
    def window_type(self) ->str:
        return self.wav2win.window_type

    @property
    def dither(self) ->float:
        return self.wav2win.dither

    def _forward_strided(self, x_strided: 'torch.Tensor', log_e: 'Optional[torch.Tensor]') ->torch.Tensor:
        X = _rfft(x_strided)
        if self.use_energy and log_e is not None:
            X[:, :, 0] = log_e
        return X

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x_strided, log_e = self.wav2win(x)
        return self._forward_strided(x_strided=x_strided, log_e=log_e)

    @torch.jit.export
    def online_inference(self, x: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        (x_strided, log_e), remainder = self.wav2win.online_inference(x, context=context)
        return self._forward_strided(x_strided=x_strided, log_e=log_e), remainder


class Wav2Spec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a magnitude spectrum (``use_fft_mag=True``)
    or a power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Spec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, round_to_power_of_two: 'bool'=True, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, use_energy: 'bool'=True, use_fft_mag: 'bool'=False) ->None:
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(self, x_strided: 'torch.Tensor', log_e: 'Optional[torch.Tensor]') ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e
        return pow_spec


class Wav2LogSpec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a log-magnitude spectrum (``use_fft_mag=True``)
    or a log-power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogSpec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, round_to_power_of_two: 'bool'=True, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, use_energy: 'bool'=True, use_fft_mag: 'bool'=False) ->None:
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(self, x_strided: 'torch.Tensor', log_e: 'Optional[torch.Tensor]') ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = (pow_spec + 1e-15).log()
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e
        return pow_spec


def lin2mel(x):
    return 1127.0 * np.log(1 + x / 700)


def create_mel_scale(num_filters: 'int', fft_length: 'int', sampling_rate: 'int', low_freq: 'float'=0, high_freq: 'Optional[float]'=None, norm_filters: 'bool'=True) ->torch.Tensor:
    if high_freq is None or high_freq == 0:
        high_freq = sampling_rate / 2
    if high_freq < 0:
        high_freq = sampling_rate / 2 + high_freq
    mel_low_freq = lin2mel(low_freq)
    mel_high_freq = lin2mel(high_freq)
    melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
    mels = lin2mel(np.linspace(0, sampling_rate, fft_length))
    B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=np.float32)
    for k in range(num_filters):
        left_mel = melfc[k]
        center_mel = melfc[k + 1]
        right_mel = melfc[k + 2]
        for j in range(int(fft_length / 2)):
            mel_j = mels[j]
            if left_mel < mel_j < right_mel:
                if mel_j <= center_mel:
                    B[j, k] = (mel_j - left_mel) / (center_mel - left_mel)
                else:
                    B[j, k] = (right_mel - mel_j) / (right_mel - center_mel)
    if norm_filters:
        B = B / np.sum(B, axis=0, keepdims=True)
    return torch.from_numpy(B)


class Wav2LogFilterBank(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their log-Mel filter bank energies (also known as "fbank").

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogFilterBank()
        >>> t(x).shape
        torch.Size([1, 100, 80])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_filters)``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, round_to_power_of_two: 'bool'=True, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, use_energy: 'bool'=False, use_fft_mag: 'bool'=False, low_freq: 'float'=20.0, high_freq: 'float'=-400.0, num_filters: 'int'=80, norm_filters: 'bool'=False, torchaudio_compatible_mel_scale: 'bool'=True):
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self._eps = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram
        if torchaudio_compatible_mel_scale:
            fb, _ = get_mel_banks(num_bins=num_filters, window_length_padded=self.fft_length, sample_freq=sampling_rate, low_freq=low_freq, high_freq=high_freq, vtln_warp_factor=1.0, vtln_low=100.0, vtln_high=-500.0)
            fb = torch.nn.functional.pad(fb, (0, 1), mode='constant', value=0).T
        else:
            fb = create_mel_scale(num_filters=num_filters, fft_length=self.fft_length, sampling_rate=sampling_rate, low_freq=low_freq, high_freq=high_freq, norm_filters=norm_filters)
        self._fb = nn.Parameter(fb, requires_grad=False)

    def _forward_strided(self, x_strided: 'torch.Tensor', log_e: 'Optional[torch.Tensor]') ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()
        if self.use_energy and log_e is not None:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)
        return pow_spec


class Wav2MFCC(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Mel-Frequency Cepstral Coefficients (MFCC).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2MFCC()
        >>> t(x).shape
        torch.Size([1, 100, 13])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_ceps)``.
    """

    def __init__(self, sampling_rate: 'int'=16000, frame_length: 'Seconds'=0.025, frame_shift: 'Seconds'=0.01, round_to_power_of_two: 'bool'=True, remove_dc_offset: 'bool'=True, preemph_coeff: 'float'=0.97, window_type: 'str'='povey', dither: 'float'=0.0, snip_edges: 'bool'=False, energy_floor: 'float'=EPSILON, raw_energy: 'bool'=True, use_energy: 'bool'=False, use_fft_mag: 'bool'=False, low_freq: 'float'=20.0, high_freq: 'float'=-400.0, num_filters: 'int'=23, norm_filters: 'bool'=False, num_ceps: 'int'=13, cepstral_lifter: 'int'=22, torchaudio_compatible_mel_scale: 'bool'=True):
        super().__init__(sampling_rate, frame_length, frame_shift, round_to_power_of_two=round_to_power_of_two, remove_dc_offset=remove_dc_offset, preemph_coeff=preemph_coeff, window_type=window_type, dither=dither, snip_edges=snip_edges, energy_floor=energy_floor, raw_energy=raw_energy, use_energy=use_energy)
        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.num_ceps = num_ceps
        self.cepstral_lifter = cepstral_lifter
        self._eps = nn.Parameter(torch.tensor(torch.finfo(torch.float).eps), requires_grad=False)
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram
        if torchaudio_compatible_mel_scale:
            fb, _ = get_mel_banks(num_bins=num_filters, window_length_padded=self.fft_length, sample_freq=sampling_rate, low_freq=low_freq, high_freq=high_freq, vtln_warp_factor=1.0, vtln_low=100.0, vtln_high=-500.0)
            fb = torch.nn.functional.pad(fb, (0, 1), mode='constant', value=0).T
        else:
            fb = create_mel_scale(num_filters=num_filters, fft_length=self.fft_length, sampling_rate=sampling_rate, low_freq=low_freq, high_freq=high_freq, norm_filters=norm_filters)
        self._fb = nn.Parameter(fb, requires_grad=False)
        self._dct = nn.Parameter(self.make_dct_matrix(self.num_ceps, self.num_filters), requires_grad=False)
        self._lifter = nn.Parameter(self.make_lifter(self.num_ceps, self.cepstral_lifter), requires_grad=False)

    @staticmethod
    def make_lifter(N, Q):
        """Makes the liftering function

        Args:
          N: Number of cepstral coefficients.
          Q: Liftering parameter
        Returns:
          Liftering vector.
        """
        if Q == 0:
            return 1
        return 1 + 0.5 * Q * torch.sin(math.pi * torch.arange(N, dtype=torch.get_default_dtype()) / Q)

    @staticmethod
    def make_dct_matrix(num_ceps, num_filters):
        n = torch.arange(float(num_filters)).unsqueeze(1)
        k = torch.arange(float(num_ceps))
        dct = torch.cos(math.pi / float(num_filters) * (n + 0.5) * k)
        dct[:, 0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(num_filters))
        return dct

    def _forward_strided(self, x_strided: 'torch.Tensor', log_e: 'Optional[torch.Tensor]') ->torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()
        mfcc = torch.matmul(pow_spec, self._dct)
        if self.cepstral_lifter > 0:
            mfcc *= self._lifter
        if self.use_energy and log_e is not None:
            mfcc[:, 0] = log_e
        return mfcc


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GlobalMVN,
     lambda: ([], {'feature_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomizedSmoothing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

