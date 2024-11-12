
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


from torch.utils.data import Dataset


from typing import Tuple


from typing import Union


from torch.utils.data import ConcatDataset


from typing import Dict


from typing import TypeVar


import random


import torch


import warnings


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


import scipy.stats


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from itertools import accumulate


from typing import Mapping


import math


from torch.autograd import Function


from math import ceil


from typing import cast


import types


from collections import OrderedDict


from torch.cuda import amp


import torch.optim as optim


from typing import Type


from torch import optim


from torch.optim import lr_scheduler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import DataLoader


import abc


import time


from collections import defaultdict


from torchvision import transforms


from itertools import starmap


from torch.utils.model_zoo import tqdm


from torch.hub import load_state_dict_from_url


from enum import Enum


from typing import IO


from typing import NamedTuple


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


from torchvision.transforms import Compose


import itertools


import copy


def lower_bound_bwd(x: 'Tensor', bound: 'Tensor', grad_output: 'Tensor'):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


def lower_bound_fwd(x: 'Tensor', bound: 'Tensor') ->Tensor:
    return torch.max(x, bound)


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    bound: 'Tensor'

    def __init__(self, bound: 'float'):
        super().__init__()
        self.register_buffer('bound', torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


_entropy_coder = 'ans'


_available_entropy_coders = [_entropy_coder]


def available_entropy_coders():
    """
    Return the list of available entropy coders.
    """
    return _available_entropy_coders


class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')
        if method not in available_entropy_coders():
            methods = ', '.join(available_entropy_coders())
            raise ValueError(f'Unknown entropy coder "{method}" (available: {methods})')
        if method == 'ans':
            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == 'rangecoder':
            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()
        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def _forward(self, *args: Any) ->Any:
    raise NotImplementedError()


def get_entropy_coder():
    """
    Return the name of the default entropy coder used to encode the bit-streams.
    """
    return _entropy_coder


def default_entropy_coder():
    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: 'Tensor', precision: 'int'=16) ->Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class EntropyModel(nn.Module):
    """Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(self, likelihood_bound: 'float'=1e-09, entropy_coder: 'Optional[str]'=None, entropy_coder_precision: 'int'=16):
        super().__init__()
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.register_buffer('_offset', torch.IntTensor())
        self.register_buffer('_quantized_cdf', torch.IntTensor())
        self.register_buffer('_cdf_length', torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes['entropy_coder'] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop('entropy_coder'))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length
    forward: 'Callable[..., Any]' = _forward

    def quantize(self, inputs: 'Tensor', mode: 'str', means: 'Optional[Tensor]'=None) ->Tensor:
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if mode == 'noise':
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs
        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs

    def _quantize(self, inputs: 'Tensor', mode: 'str', means: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('_quantize is deprecated. Use quantize instead.', stacklevel=2)
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(inputs: 'Tensor', means: 'Optional[Tensor]'=None, dtype: 'torch.dtype'=torch.float) ->Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: 'Tensor', means: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('_dequantize. Use dequantize instead.', stacklevel=2)
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() first')
        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f'Invalid CDF size {self._quantized_cdf.size()}')

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError('Uninitialized offsets. Run update() first')
        if len(self._offset.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._offset.size()}')

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError('Uninitialized CDF lengths. Run update() first')
        if len(self._cdf_length.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._cdf_length.size()}')

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, 'symbols', means)
        if len(inputs.size()) < 2:
            raise ValueError('Invalid `inputs` size. Expected a tensor with at least 2 dimensions.')
        if inputs.size() != indexes.size():
            raise ValueError('`inputs` and `indexes` should have the same size.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(symbols[i].reshape(-1).int().tolist(), indexes[i].reshape(-1).int().tolist(), self._quantized_cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            strings.append(rv)
        return strings

    def decompress(self, strings: 'str', indexes: 'torch.IntTensor', dtype: 'torch.dtype'=torch.float, means: 'torch.Tensor'=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """
        if not isinstance(strings, (tuple, list)):
            raise ValueError('Invalid `strings` parameter type.')
        if not len(strings) == indexes.size(0):
            raise ValueError('Invalid strings or indexes parameters')
        if len(indexes.size()) < 2:
            raise ValueError('Invalid `indexes` size. Expected a tensor with at least 2 dimensions.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError('Invalid means or indexes parameters')
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError('Invalid means parameters')
        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())
        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(s, indexes[i].reshape(-1).int().tolist(), cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype)
        return outputs


class EntropyBottleneck(EntropyModel):
    """Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """
    _offset: 'Tensor'

    def __init__(self, channels: 'int', *args: Any, tail_mass: float=1e-09, init_scale: float=10, filters: Tuple[int, ...]=(3, 3, 3, 3), **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.matrices.append(nn.Parameter(matrix))
            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(nn.Parameter(bias))
            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.factors.append(nn.Parameter(factor))
        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer('target', torch.Tensor([-target, 0, target]))

    def _get_medians(self) ->Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: 'bool'=False, update_quantiles: 'bool'=False) ->bool:
        if self._offset.numel() > 0 and not force:
            return False
        if update_quantiles:
            self._update_quantiles()
        medians = self.quantiles[:, 0, 1]
        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)
        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)
        self._offset = -minima
        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]
        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) ->Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: 'Tensor', stop_gradient: 'bool') ->Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)
            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias
            if i < len(self.filters):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(self, inputs: 'Tensor', stop_gradient: 'bool'=False) ->Tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper

    def forward(self, x: 'Tensor', training: 'Optional[bool]'=None) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if not torch.jit.is_scripting():
            perm = torch.cat((torch.tensor([1, 0], dtype=torch.long, device=x.device), torch.arange(2, x.ndim, dtype=torch.long, device=x.device)))
            inv_perm = perm
        else:
            raise NotImplementedError()
        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)
        outputs = self.quantize(values, 'noise' if training else 'dequantize', self._get_medians())
        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]
        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()
        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    @torch.no_grad()
    def _update_quantiles(self, search_radius=100000.0, rtol=0.0001, atol=0.001):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        device = self.quantiles.device
        shape = self.channels, 1, 1
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)
        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=0.0001, atol=0.001, strict=False):
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = len(strings), self._quantized_cdf.size(0), *size
        indexes = self._build_indexes(output_size)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditional(EntropyModel):
    """Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(self, scale_table: 'Optional[Union[List, Tuple]]', *args: Any, scale_bound: float=0.11, tail_mass: float=1e-09, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')
        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')
        if scale_table and (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')
        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError('Invalid parameters')
        self.lower_bound_scale = LowerBound(scale_bound)
        self.register_buffer('scale_table', self._prepare_scale_table(scale_table) if scale_table else torch.Tensor())
        self.register_buffer('scale_bound', torch.Tensor([float(scale_bound)]) if scale_bound is not None else None)

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: 'Tensor') ->Tensor:
        half = float(0.5)
        const = float(-2 ** -0.5)
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()
        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower
        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs: 'Tensor', scales: 'Tensor', means: 'Optional[Tensor]'=None) ->Tensor:
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.lower_bound_scale(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def forward(self, inputs: 'Tensor', scales: 'Tensor', means: 'Optional[Tensor]'=None, training: 'Optional[bool]'=None) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, 'noise' if training else 'dequantize', means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales: 'Tensor') ->Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes


class GaussianMixtureConditional(GaussianConditional):

    def __init__(self, K=3, scale_table: 'Optional[Union[List, Tuple]]'=None, *args: Any, **kwargs: Any):
        super().__init__(scale_table, *args, **kwargs)
        self.K = K

    def _likelihood(self, inputs: 'Tensor', scales: 'Tensor', means: 'Tensor', weights: 'Tensor') ->Tensor:
        likelihood = torch.zeros_like(inputs)
        M = inputs.size(1)
        for k in range(self.K):
            likelihood += super()._likelihood(inputs, scales[:, M * k:M * (k + 1)], means[:, M * k:M * (k + 1)]) * weights[:, M * k:M * (k + 1)]
        return likelihood

    def forward(self, inputs: 'Tensor', scales: 'Tensor', means: 'Tensor', weights: 'Tensor', training: 'Optional[bool]'=None) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, 'noise' if training else 'dequantize', means=None)
        likelihood = self._likelihood(outputs, scales, means, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    @torch.no_grad()
    def _build_cdf(self, scales, means, weights, abs_max):
        num_latents = scales.size(1)
        num_samples = abs_max * 2 + 1
        TINY = 1e-10
        device = scales.device
        scales = scales.clamp_(0.11, 256)
        means += abs_max
        scales_ = scales.unsqueeze(-1).expand(-1, -1, num_samples)
        means_ = means.unsqueeze(-1).expand(-1, -1, num_samples)
        weights_ = weights.unsqueeze(-1).expand(-1, -1, num_samples)
        samples = torch.arange(num_samples).unsqueeze(0).expand(num_latents, -1)
        pmf = torch.zeros_like(samples).float()
        for k in range(self.K):
            pmf += (0.5 * (1 + torch.erf((samples + 0.5 - means_[k]) / ((scales_[k] + TINY) * 2 ** 0.5))) - 0.5 * (1 + torch.erf((samples - 0.5 - means_[k]) / ((scales_[k] + TINY) * 2 ** 0.5)))) * weights_[k]
        cdf_limit = 2 ** self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)
        cdf = F.pad(torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(), (1, 0), 'constant', 0)
        pmf_quantized = torch.diff(cdf, dim=1)
        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)
        _, pmf_first_stealable_indices = torch.min(torch.where(pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples), pmf_quantized, torch.tensor(cdf_limit + 1).int()), dim=1)
        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1
        pmf_real_steal_indices = torch.cat((torch.arange(num_latents).unsqueeze(-1), pmf_first_stealable_indices.unsqueeze(-1)), dim=1).transpose(0, 1)
        pmf_quantized[pmf_real_steal_indices[0], pmf_real_steal_indices[1]] -= pmf_zero_count
        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), 'constant', 0)
        cdf = F.pad(cdf, (0, 1), 'constant', cdf_limit + 1)
        return cdf

    def reshape_entropy_parameters(self, scales, means, weights, nonzero):
        reshape_size = scales.size(0), self.K, scales.size(1) // self.K, -1
        scales = scales.reshape(*reshape_size)[:, :, nonzero].permute(1, 0, 2, 3).reshape(self.K, -1)
        means = means.reshape(*reshape_size)[:, :, nonzero].permute(1, 0, 2, 3).reshape(self.K, -1)
        weights = weights.reshape(*reshape_size)[:, :, nonzero].permute(1, 0, 2, 3).reshape(self.K, -1)
        return scales, means, weights

    def compress(self, y, scales, means, weights):
        abs_max = max(torch.abs(y.max()).int().item(), torch.abs(y.min()).int().item()) + 1
        abs_max = 1 if abs_max < 1 else abs_max
        y_quantized = torch.round(y)
        zero_bitmap = torch.where(torch.sum(torch.abs(y_quantized), (3, 2)).squeeze(0) == 0, 0, 1)
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y_quantized[:, nonzero] + abs_max
        cdf = self._build_cdf(*self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max)
        num_latents = cdf.size(0)
        rv = self.entropy_coder._encoder.encode_with_indexes(symbols.reshape(-1).int().tolist(), torch.arange(num_latents).int().tolist(), cdf.cpu().tolist(), torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(), torch.tensor(0).repeat(num_latents).int().tolist())
        return (rv, abs_max, zero_bitmap), y_quantized

    def decompress(self, strings, abs_max, zero_bitmap, scales, means, weights):
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        cdf = self._build_cdf(*self.reshape_entropy_parameters(scales, means, weights, nonzero), abs_max)
        num_latents = cdf.size(0)
        values = self.entropy_coder._decoder.decode_with_indexes(strings, torch.arange(num_latents).int().tolist(), cdf.cpu().tolist(), torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(), torch.tensor(0).repeat(num_latents).int().tolist())
        symbols = torch.tensor(values) - abs_max
        symbols = symbols.reshape(scales.size(0), -1, scales.size(2), scales.size(3))
        y_hat = torch.zeros(scales.size(0), zero_bitmap.size(0), scales.size(2), scales.size(3))
        y_hat[:, nonzero] = symbols.float()
        return y_hat


class EntropyModelVbr(nn.Module):
    """Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(self, likelihood_bound: 'float'=1e-09, entropy_coder: 'Optional[str]'=None, entropy_coder_precision: 'int'=16):
        super().__init__()
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.register_buffer('_offset', torch.IntTensor())
        self.register_buffer('_quantized_cdf', torch.IntTensor())
        self.register_buffer('_cdf_length', torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes['entropy_coder'] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop('entropy_coder'))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length
    forward: 'Callable[..., Any]' = _forward

    def quantize(self, inputs: 'Tensor', mode: 'str', means: 'Optional[Tensor]'=None) ->Tensor:
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if mode == 'noise':
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs
        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs

    def quantize_variable(self, inputs: 'Tensor', mode: 'str', means: 'Optional[Tensor]'=None, qs: 'Optional[Tensor]'=None) ->Tensor:
        if mode not in ('noise', 'ste', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if qs is not None:
            assert qs.shape == torch.Size([])
        if mode == 'noise':
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            if qs is None:
                inputs = inputs + noise
            else:
                inputs = inputs + noise * qs
            return inputs
        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        if mode == 'ste':
            if qs is None:
                outputs_ste = torch.round(outputs) - outputs.detach() + outputs
            else:
                outputs_ste = torch.round(outputs / qs) * qs - outputs.detach() + outputs
            if means is not None:
                outputs_ste += means
            return outputs_ste
        if mode == 'dequantize':
            if qs is None:
                outputs = torch.round(outputs)
            else:
                outputs = torch.round(outputs / qs) * qs
            if means is not None:
                outputs += means
            return outputs
        assert mode == 'symbols', mode
        if qs is None:
            outputs = outputs.int()
        else:
            outputs = torch.round(outputs / qs).int()
        return outputs

    def _quantize(self, inputs: 'Tensor', mode: 'str', means: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('_quantize is deprecated. Use quantize instead.', stacklevel=2)
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(inputs: 'Tensor', means: 'Optional[Tensor]'=None, dtype: 'torch.dtype'=torch.float) ->Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @staticmethod
    def dequantize_variable(inputs: 'Tensor', means: 'Optional[Tensor]'=None, dtype: 'torch.dtype'=torch.float, qs: 'Optional[Tensor]'=None) ->Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            if qs is None:
                outputs += means
            else:
                outputs = outputs * qs + means
        elif qs is None:
            outputs = inputs.type(dtype)
        else:
            outputs = inputs.type(dtype) * qs
        return outputs

    @classmethod
    def _dequantize(cls, inputs: 'Tensor', means: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('_dequantize. Use dequantize instead.', stacklevel=2)
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() first')
        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f'Invalid CDF size {self._quantized_cdf.size()}')

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError('Uninitialized offsets. Run update() first')
        if len(self._offset.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._offset.size()}')

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError('Uninitialized CDF lengths. Run update() first')
        if len(self._cdf_length.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._cdf_length.size()}')

    def compress(self, inputs, indexes, means=None, qs=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
            qs (torch.Tensor, optional): optional quantization step size
        """
        if qs is None:
            symbols = self.quantize(inputs, 'symbols', means)
        else:
            symbols = self.quantize_variable(inputs, 'symbols', means=means, qs=qs)
        if len(inputs.size()) < 2:
            raise ValueError('Invalid `inputs` size. Expected a tensor with at least 2 dimensions.')
        if inputs.size() != indexes.size():
            raise ValueError('`inputs` and `indexes` should have the same size.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(symbols[i].reshape(-1).int().tolist(), indexes[i].reshape(-1).int().tolist(), self._quantized_cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            strings.append(rv)
        return strings

    def decompress(self, strings: 'str', indexes: 'torch.IntTensor', dtype: 'torch.dtype'=torch.float, means: 'torch.Tensor'=None, qs=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
            qs (torch.Tensor, optional): optional quantization step size
        """
        if not isinstance(strings, (tuple, list)):
            raise ValueError('Invalid `strings` parameter type.')
        if not len(strings) == indexes.size(0):
            raise ValueError('Invalid strings or indexes parameters')
        if len(indexes.size()) < 2:
            raise ValueError('Invalid `indexes` size. Expected a tensor with at least 2 dimensions.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError('Invalid means or indexes parameters')
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError('Invalid means parameters')
        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())
        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(s, indexes[i].reshape(-1).int().tolist(), cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
        if qs is None:
            outputs = self.dequantize(outputs, means, dtype)
        else:
            outputs = self.dequantize_variable(outputs, means=means, dtype=dtype, qs=qs)
        return outputs


class EntropyBottleneckVbr(EntropyModelVbr):
    """Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """
    _offset: 'Tensor'

    def __init__(self, channels: 'int', *args: Any, tail_mass: float=1e-09, init_scale: float=10, filters: Tuple[int, ...]=(3, 3, 3, 3), **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f'_matrix{i:d}', nn.Parameter(matrix))
            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f'_bias{i:d}', nn.Parameter(bias))
            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f'_factor{i:d}', nn.Parameter(factor))
        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer('target', torch.Tensor([-target, 0, target]))

    def _get_medians(self) ->Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: 'bool'=False) ->bool:
        if self._offset.numel() > 0 and not force:
            return False
        medians = self.quantiles[:, 0, 1]
        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)
        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)
        self._offset = -minima
        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]
        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def update_variable(self, force: 'bool'=False, qs=1.0) ->bool:
        if self._offset.numel() > 0 and not force:
            return False
        medians = self.quantiles[:, 0, 1]
        minima = (medians - self.quantiles[:, 0, 0]) / qs
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)
        maxima = (self.quantiles[:, 0, 2] - medians) / qs
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)
        self._offset = -minima
        pmf_start = medians - minima * qs
        pmf_length = maxima + minima + 1
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device) * qs
        samples = samples[None, :] + pmf_start[:, None, None]
        pmf, lower, upper = self._likelihood_variable(samples, stop_gradient=True, qs=qs)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) ->Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: 'Tensor', stop_gradient: 'bool') ->Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f'_matrix{i:d}')
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)
            bias = getattr(self, f'_bias{i:d}')
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self.filters):
                factor = getattr(self, f'_factor{i:d}')
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs: 'Tensor', stop_gradient: 'bool'=False) ->Tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper

    @torch.jit.unused
    def _likelihood_variable(self, inputs: 'Tensor', stop_gradient: 'bool'=False, qs: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        if qs is None:
            v0 = inputs - half
            v1 = inputs + half
        else:
            v0 = inputs - half * qs
            v1 = inputs + half * qs
        lower = self._logits_cumulative(v0, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(v1, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper

    def forward(self, x: 'Tensor', training: 'Optional[bool]'=None, qs: 'Optional[Tensor]'=None, ste: 'Optional[bool]'=False) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if not torch.jit.is_scripting():
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)
        if qs is None:
            outputs = self.quantize(values, 'noise' if training else 'dequantize', self._get_medians())
        elif ste is False:
            outputs = self.quantize_variable(values, 'noise' if training else 'dequantize', self._get_medians(), qs)
        else:
            outputs = self.quantize_variable(values, 'ste', self._get_medians(), qs)
        if not torch.jit.is_scripting():
            if qs is None:
                likelihood, _, _ = self._likelihood(outputs)
            elif ste and training:
                likelihood, _, _ = self._likelihood_variable(outputs, qs)
            else:
                likelihood, _, _ = self._likelihood_variable(outputs, qs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]
        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()
        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x, qs=None):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians, qs)

    def decompress(self, strings, size, qs=None):
        output_size = len(strings), self._quantized_cdf.size(0), *size
        indexes = self._build_indexes(output_size)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians, qs)


class _SetDefaultMixin:
    """Convenience functions for initializing classes with defaults."""

    def _setdefault(self, k, v, f):
        """Initialize attribute ``k`` with value ``v`` or ``f()``."""
        v = v or f()
        setattr(self, k, v)

    def _set_group_defaults(self, group_key, group_dict, defaults, save_direct=False):
        """Initialize attribute ``group_key`` with items from
        ``group_dict``, using defaults for missing keys.
        Ensures ``nn.Module`` attributes are properly registered.

        Args:
            - group_key:
                Name of attribute.
            - group_dict:
                Dict of items to initialize ``group_key`` with.
            - defaults:
                Dict of defaults for items not in ``group_dict``.
            - save_direct:
                If ``True``, save items directly as attributes of ``self``.
                If ``False``, save items in a ``nn.ModuleDict``.
        """
        group_dict = group_dict if group_dict is not None else {}
        for k, f in defaults.items():
            if k in group_dict:
                continue
            group_dict[k] = f()
        if save_direct:
            for k, v in group_dict.items():
                setattr(self, k, v)
        else:
            group_dict = nn.ModuleDict(group_dict)
        setattr(self, group_key, group_dict)


class LatentCodec(nn.Module, _SetDefaultMixin):

    def forward(self, y: 'Tensor', *args, **kwargs) ->Dict[str, Any]:
        raise NotImplementedError

    def compress(self, y: 'Tensor', *args, **kwargs) ->Dict[str, Any]:
        raise NotImplementedError

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Any', *args, **kwargs) ->Dict[str, Any]:
        raise NotImplementedError


class ChannelGroupsLatentCodec(LatentCodec):
    """Reconstructs groups of channels using previously decoded groups.

    Context model from [Minnen2020] and [He2022].
    Also known as a "channel-conditional" (CC) entropy model.

    See :py:class:`~compressai.models.sensetime.Elic2022Official`
    for example usage.

    [Minnen2020]: `"Channel-wise Autoregressive Entropy Models for
    Learned Image Compression" <https://arxiv.org/abs/2007.08739>`_, by
    David Minnen, and Saurabh Singh, ICIP 2020.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.
    """
    latent_codec: 'Mapping[str, LatentCodec]'
    channel_context: 'Mapping[str, nn.Module]'

    def __init__(self, latent_codec: 'Optional[Mapping[str, LatentCodec]]'=None, channel_context: 'Optional[Mapping[str, nn.Module]]'=None, *, groups: List[int], **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.groups = list(groups)
        self.groups_acc = list(accumulate(self.groups, initial=0))
        self.channel_context = nn.ModuleDict(channel_context)
        self.latent_codec = nn.ModuleDict(latent_codec)

    def __getitem__(self, key: 'str') ->LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat_ = [Tensor()] * len(self.groups)
        y_likelihoods_ = [Tensor()] * len(self.groups)
        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f'y{k}'](y_[k], params)
            y_hat_[k] = y_out_[k]['y_hat']
            y_likelihoods_[k] = y_out_[k]['likelihoods']['y']
        y_hat = torch.cat(y_hat_, dim=1)
        y_likelihoods = torch.cat(y_likelihoods_, dim=1)
        return {'likelihoods': {'y': y_likelihoods}, 'y_hat': y_hat}

    def compress(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat = torch.zeros_like(y)
        y_hat_ = y_hat.split(self.groups, dim=1)
        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f'y{k}'].compress(y_[k], params)
            y_hat_[k][:] = y_out_[k]['y_hat']
        y_strings_groups = [y_out['strings'] for y_out in y_out_]
        assert all(len(y_strings_groups[0]) == len(ss) for ss in y_strings_groups)
        return {'strings': [s for ss in y_strings_groups for s in ss], 'shape': [y_out['shape'] for y_out in y_out_], 'y_hat': y_hat}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'List[Tuple[int, ...]]', side_params: 'Tensor', **kwargs) ->Dict[str, Any]:
        n = len(strings[0])
        assert all(len(ss) == n for ss in strings)
        strings_per_group = len(strings) // len(self.groups)
        y_out_ = [{}] * len(self.groups)
        y_shape = sum(s[0] for s in shape), *shape[0][1:]
        y_hat = torch.zeros((n, *y_shape), device=side_params.device)
        y_hat_ = y_hat.split(self.groups, dim=1)
        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f'y{k}'].decompress(strings[strings_per_group * k:strings_per_group * (k + 1)], shape[k], params)
            y_hat_[k][:] = y_out_[k]['y_hat']
        return {'y_hat': y_hat}

    def merge_y(self, *args):
        return torch.cat(args, dim=1)

    def merge_params(self, *args):
        return torch.cat(args, dim=1)

    def _get_ctx_params(self, k: 'int', side_params: 'Tensor', y_hat_: 'List[Tensor]') ->Tensor:
        if k == 0:
            return side_params
        ch_ctx_params = self.channel_context[f'y{k}'](self.merge_y(*y_hat_[:k]))
        return self.merge_params(ch_ctx_params, side_params)


def quantize_ste(x: 'Tensor') ->Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


class GaussianConditionalLatentCodec(LatentCodec):
    """Gaussian conditional for compressing latent ``y`` using ``ctx_params``.

    Probability model for Gaussian of ``(scales, means)``.

    Gaussian conditonal entropy model introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: Unlike the original paper, which models only the scale
       (i.e. "width") of the Gaussian, this implementation models both
       the scale and the mean (i.e. "center") of the Gaussian.

    .. code-block:: none

                          ctx_params
                              │
                              ▼
                              │
                           ┌──┴──┐
                           │  EP │
                           └──┬──┘
                              │
               ┌───┐  y_hat   ▼
        y ──►──┤ Q ├────►────····──►── y_hat
               └───┘          GC

    """
    gaussian_conditional: 'GaussianConditional'
    entropy_parameters: 'nn.Module'

    def __init__(self, scale_table: 'Optional[Union[List, Tuple]]'=None, gaussian_conditional: 'Optional[GaussianConditional]'=None, entropy_parameters: 'Optional[nn.Module]'=None, quantizer: 'str'='noise', chunks: 'Tuple[str]'=('scales', 'means'), **kwargs):
        super().__init__()
        self.quantizer = quantizer
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(scale_table, **kwargs)
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.chunks = tuple(chunks)

    def forward(self, y: 'Tensor', ctx_params: 'Tensor') ->Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if self.quantizer == 'ste':
            y_hat = quantize_ste(y - means_hat) + means_hat
        return {'likelihoods': {'y': y_likelihoods}, 'y_hat': y_hat}

    def compress(self, y: 'Tensor', ctx_params: 'Tensor') ->Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
        return {'strings': [y_strings], 'shape': y.shape[2:4], 'y_hat': y_hat}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, int]', ctx_params: 'Tensor', **kwargs) ->Dict[str, Any]:
        y_strings, = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
        assert y_hat.shape[2:4] == shape
        return {'y_hat': y_hat}

    def _chunk(self, params: 'Tensor') ->Tuple[Tensor, Tensor]:
        scales, means = None, None
        if self.chunks == ('scales',):
            scales = params
        if self.chunks == ('means',):
            means = params
        if self.chunks == ('scales', 'means'):
            scales, means = params.chunk(2, 1)
        if self.chunks == ('means', 'scales'):
            means, scales = params.chunk(2, 1)
        return scales, means


class CheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """
    latent_codec: 'Mapping[str, LatentCodec]'
    entropy_parameters: 'nn.Module'
    context_prediction: 'CheckerboardMaskedConv2d'

    def __init__(self, latent_codec: 'Optional[Mapping[str, LatentCodec]]'=None, entropy_parameters: 'Optional[nn.Module]'=None, context_prediction: 'Optional[nn.Module]'=None, anchor_parity='even', forward_method='twopass', **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {'odd': 'even', 'even': 'odd'}[anchor_parity]
        self.forward_method = forward_method
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or nn.Identity()
        self._set_group_defaults('latent_codec', latent_codec, defaults={'y': lambda : GaussianConditionalLatentCodec(quantizer='ste')}, save_direct=True)

    def __getitem__(self, key: 'str') ->LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        if self.forward_method == 'onepass':
            return self._forward_onepass(y, side_params)
        if self.forward_method == 'twopass':
            return self._forward_twopass(y, side_params)
        if self.forward_method == 'twopass_faster':
            return self._forward_twopass_faster(y, side_params)
        raise ValueError(f'Unknown forward method: {self.forward_method}')

    def _forward_onepass(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        y_hat = self.quantize(y)
        y_ctx = self._keep_only(self.context_prediction(y_hat), 'non_anchor')
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec['y'](y, params)
        return {'likelihoods': {'y': y_out['likelihoods']['y']}, 'y_hat': y_hat}

    def _forward_twopass(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.

        """
        B, C, H, W = y.shape
        params = y.new_zeros((B, C * 2, H, W))
        y_hat_anchors = self._forward_twopass_step(y, side_params, params, self._y_ctx_zero(y), 'anchor')
        y_hat_non_anchors = self._forward_twopass_step(y, side_params, params, self.context_prediction(y_hat_anchors), 'non_anchor')
        y_hat = y_hat_anchors + y_hat_non_anchors
        y_out = self.latent_codec['y'](y, params)
        return {'likelihoods': {'y': y_out['likelihoods']['y']}, 'y_hat': y_hat}

    def _forward_twopass_step(self, y: 'Tensor', side_params: 'Tensor', params: 'Tensor', y_ctx: 'Tensor', step: 'str') ->Dict[str, Any]:
        assert step in ('anchor', 'non_anchor')
        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))
        self._copy(params, params_i, step)
        func = getattr(self.latent_codec['y'], 'entropy_parameters', lambda x: x)
        params_i = func(params_i)
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)
        _, means_i = self.latent_codec['y']._chunk(params_i)
        y_hat_i = self._keep_only(quantize_ste(y_i - means_i) + means_i, step)
        return y_hat_i

    def _forward_twopass_faster(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        func = getattr(self.latent_codec['y'], 'entropy_parameters', lambda x: x)
        params = func(params)
        params = self._keep_only(params, 'anchor')
        _, means_hat = self.latent_codec['y']._chunk(params)
        y_hat_anchors = quantize_ste(y - means_hat) + means_hat
        y_hat_anchors = self._keep_only(y_hat_anchors, 'anchor')
        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = self._keep_only(y_ctx, 'non_anchor')
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec['y'](y, params)
        y_hat = y_out['y_hat']
        self._copy(y_hat, y_hat_anchors, 'anchor')
        return {'likelihoods': {'y': y_out['likelihoods']['y']}, 'y_hat': y_hat}

    @torch.no_grad()
    def _y_ctx_zero(self, y: 'Tensor') ->Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        y_ctx_meta = self.context_prediction(y)
        return y.new_zeros(y_ctx_meta.shape)

    def compress(self, y: 'Tensor', side_params: 'Tensor') ->Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2
        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, 'all')
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec['y'].compress(y_[i], params_i)
            y_hat_[i] = y_out['y_hat']
            [y_strings_[i]] = y_out['strings']
        y_hat = self.embed(y_hat_)
        return {'strings': y_strings_, 'shape': y_hat.shape[1:], 'y_hat': y_hat}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, ...]', side_params: 'Tensor', **kwargs) ->Dict[str, Any]:
        y_strings_ = strings
        n = len(y_strings_[0])
        assert len(y_strings_) == 2
        assert all(len(x) == n for x in y_strings_)
        c, h, w = shape
        y_i_shape = h, w // 2
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, 'all')
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec['y'].decompress([y_strings_[i]], y_i_shape, params_i)
            y_hat_[i] = y_out['y_hat']
        y_hat = self.embed(y_hat_)
        return {'y_hat': y_hat}

    def unembed(self, y: 'Tensor') ->Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            ■ □ ■ □         ■ ■   □ □
            □ ■ □ ■   --->  ■ ■   □ □
            ■ □ ■ □         ■ ■   □ □
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        if self.anchor_parity == 'even':
            y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        else:
            y_[0, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 0::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 1::2]
        return y_

    def embed(self, y_: 'Tensor') ->Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            ■ ■   □ □         ■ □ ■ □
            ■ ■   □ □   --->  □ ■ □ ■
            ■ ■   □ □         ■ □ ■ □
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        if self.anchor_parity == 'even':
            y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        else:
            y[..., 0::2, 1::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 0::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[1, ..., 1::2, :]
        return y

    def _copy(self, dest: 'Tensor', src: 'Tensor', step: 'str') ->None:
        """Copy pixels in the current step."""
        assert step in ('anchor', 'non_anchor')
        parity = self.anchor_parity if step == 'anchor' else self.non_anchor_parity
        if parity == 'even':
            dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
            dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
        else:
            dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
            dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]

    def _keep_only(self, y: 'Tensor', step: 'str', inplace: 'bool'=False) ->Tensor:
        """Keep only pixels in the current step, and zero out the rest."""
        return self._mask(y, parity=self.non_anchor_parity if step == 'anchor' else self.anchor_parity, inplace=inplace)

    def _mask(self, y: 'Tensor', parity: 'str', inplace: 'bool'=False) ->Tensor:
        if not inplace:
            y = y.clone()
        if parity == 'even':
            y[..., 0::2, 0::2] = 0
            y[..., 1::2, 1::2] = 0
        elif parity == 'odd':
            y[..., 0::2, 1::2] = 0
            y[..., 1::2, 0::2] = 0
        elif parity == 'all':
            y[:] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: 'Tensor') ->Tensor:
        mode = 'noise' if self.training else 'dequantize'
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat


class EntropyBottleneckLatentCodec(LatentCodec):
    """Entropy bottleneck codec.

    Factorized prior "entropy bottleneck" introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

               ┌───┐ y_hat
        y ──►──┤ Q ├───►───····───►─── y_hat
               └───┘        EB

    """
    entropy_bottleneck: 'EntropyBottleneck'

    def __init__(self, entropy_bottleneck: 'Optional[EntropyBottleneck]'=None, **kwargs):
        super().__init__()
        self.entropy_bottleneck = entropy_bottleneck or EntropyBottleneck(**kwargs)

    def forward(self, y: 'Tensor') ->Dict[str, Any]:
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return {'likelihoods': {'y': y_likelihoods}, 'y_hat': y_hat}

    def compress(self, y: 'Tensor') ->Dict[str, Any]:
        shape = y.size()[-2:]
        y_strings = self.entropy_bottleneck.compress(y)
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        return {'strings': [y_strings], 'shape': shape, 'y_hat': y_hat}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, int]', **kwargs) ->Dict[str, Any]:
        y_strings, = strings
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        return {'y_hat': y_hat}


class GainHyperLatentCodec(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    Gain-controlled side branch for hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. note:: ``GainHyperLatentCodec`` should be used inside
       ``GainHyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

                       gain                        gain_inv
                         │                             │
                         ▼                             ▼
               ┌───┐  z  │     ┌───┐ z_hat      z_hat  │       ┌───┐
        y ──►──┤h_a├──►──×──►──┤ Q ├───►───····───►────×────►──┤h_s├──►── params
               └───┘           └───┘        EB                 └───┘

    """
    entropy_bottleneck: 'EntropyBottleneck'
    h_a: 'nn.Module'
    h_s: 'nn.Module'

    def __init__(self, entropy_bottleneck: 'Optional[EntropyBottleneck]'=None, h_a: 'Optional[nn.Module]'=None, h_s: 'Optional[nn.Module]'=None, **kwargs):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()

    def forward(self, y: 'Tensor', gain: 'Tensor', gain_inv: 'Tensor') ->Dict[str, Any]:
        z = self.h_a(y)
        z = z * gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {'likelihoods': {'z': z_likelihoods}, 'params': params}

    def compress(self, y: 'Tensor', gain: 'Tensor', gain_inv: 'Tensor') ->Dict[str, Any]:
        z = self.h_a(y)
        z = z * gain
        shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {'strings': [z_strings], 'shape': shape, 'params': params}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, int]', gain_inv: 'Tensor', **kwargs) ->Dict[str, Any]:
        z_strings, = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = z_hat * gain_inv
        params = self.h_s(z_hat)
        return {'params': params}


class GainHyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Gain-controlled hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. code-block:: none

                z_gain  z_gain_inv
                   │        │
                   ▼        ▼
                  ┌┴────────┴┐
            ┌──►──┤ lc_hyper ├──►─┐
            │     └──────────┘    │
            │                     │
            │     y_gain          ▼ params   y_gain_inv
            │        │            │              │
            │        ▼            │              ▼
            │        │         ┌──┴───┐          │
        y ──┴────►───×───►─────┤ lc_y ├────►─────×─────►── y_hat
                               └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                        z_gain                      z_gain_inv
                           │                             │
                           ▼                             ▼
                 ┌───┐  z  │ z_g ┌───┐ z_hat      z_hat  │       ┌───┐
            ┌─►──┤h_a├──►──×──►──┤ Q ├───►───····───►────×────►──┤h_s├──┐
            │    └───┘           └───┘        EB                 └───┘  │
            │                                                           │
            │                              ┌──────────────◄─────────────┘
            │                              │            params
            │                           ┌──┴──┐
            │    y_gain                 │  EP │    y_gain_inv
            │       │                   └──┬──┘        │
            │       ▼                      │           ▼
            │       │       ┌───┐          ▼           │
        y ──┴───►───×───►───┤ Q ├────►────····───►─────×─────►── y_hat
                            └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """
    latent_codec: 'Mapping[str, LatentCodec]'

    def __init__(self, latent_codec: 'Optional[Mapping[str, LatentCodec]]'=None, **kwargs):
        super().__init__()
        self._set_group_defaults('latent_codec', latent_codec, defaults={'y': GaussianConditionalLatentCodec, 'hyper': GainHyperLatentCodec}, save_direct=True)

    def __getitem__(self, key: 'str') ->LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: 'Tensor', y_gain: 'Tensor', z_gain: 'Tensor', y_gain_inv: 'Tensor', z_gain_inv: 'Tensor') ->Dict[str, Any]:
        hyper_out = self.latent_codec['hyper'](y, z_gain, z_gain_inv)
        y_out = self.latent_codec['y'](y * y_gain, hyper_out['params'])
        y_hat = y_out['y_hat'] * y_gain_inv
        return {'likelihoods': {'y': y_out['likelihoods']['y'], 'z': hyper_out['likelihoods']['z']}, 'y_hat': y_hat}

    def compress(self, y: 'Tensor', y_gain: 'Tensor', z_gain: 'Tensor', y_gain_inv: 'Tensor', z_gain_inv: 'Tensor') ->Dict[str, Any]:
        hyper_out = self.latent_codec['hyper'].compress(y, z_gain, z_gain_inv)
        y_out = self.latent_codec['y'].compress(y * y_gain, hyper_out['params'])
        y_hat = y_out['y_hat'] * y_gain_inv
        return {'strings': [*y_out['strings'], *hyper_out['strings']], 'shape': {'y': y_out['shape'], 'hyper': hyper_out['shape']}, 'y_hat': y_hat}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Dict[str, Tuple[int, ...]]', y_gain_inv: 'Tensor', z_gain_inv: 'Tensor', **kwargs) ->Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec['hyper'].decompress([z_strings], shape['hyper'], z_gain_inv)
        y_out = self.latent_codec['y'].decompress(y_strings_, shape['y'], hyper_out['params'])
        y_hat = y_out['y_hat'] * y_gain_inv
        return {'y_hat': y_hat}


class HyperLatentCodec(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    "Hyper" side-information branch introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: ``HyperLatentCodec`` should be used inside
       ``HyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

               ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        y ──►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►── params
               └───┘     └───┘        EB        └───┘

    """
    entropy_bottleneck: 'EntropyBottleneck'
    h_a: 'nn.Module'
    h_s: 'nn.Module'

    def __init__(self, entropy_bottleneck: 'Optional[EntropyBottleneck]'=None, h_a: 'Optional[nn.Module]'=None, h_s: 'Optional[nn.Module]'=None, quantizer: 'str'='noise', **kwargs):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()
        self.quantizer = quantizer

    def forward(self, y: 'Tensor') ->Dict[str, Any]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quantizer == 'ste':
            z_medians = self.entropy_bottleneck._get_medians()
            z_hat = quantize_ste(z - z_medians) + z_medians
        params = self.h_s(z_hat)
        return {'likelihoods': {'z': z_likelihoods}, 'params': params}

    def compress(self, y: 'Tensor') ->Dict[str, Any]:
        z = self.h_a(y)
        shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat)
        return {'strings': [z_strings], 'shape': shape, 'params': params}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, int]', **kwargs) ->Dict[str, Any]:
        z_strings, = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat)
        return {'params': params}


class HyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Hyperprior entropy modeling introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

                 ┌──────────┐
            ┌─►──┤ lc_hyper ├──►─┐
            │    └──────────┘    │
            │                    ▼ params
            │                    │
            │                 ┌──┴───┐
        y ──┴───────►─────────┤ lc_y ├───►── y_hat
                              └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                 ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            ┌─►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►─┐
            │    └───┘     └───┘        EB        └───┘    │
            │                                              │
            │                  ┌──────────────◄────────────┘
            │                  │            params
            │               ┌──┴──┐
            │               │  EP │
            │               └──┬──┘
            │                  │
            │   ┌───┐  y_hat   ▼
        y ──┴─►─┤ Q ├────►────····────►── y_hat
                └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """
    latent_codec: 'Mapping[str, LatentCodec]'

    def __init__(self, latent_codec: 'Optional[Mapping[str, LatentCodec]]'=None, **kwargs):
        super().__init__()
        self._set_group_defaults('latent_codec', latent_codec, defaults={'y': GaussianConditionalLatentCodec, 'hyper': HyperLatentCodec}, save_direct=True)

    def __getitem__(self, key: 'str') ->LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: 'Tensor') ->Dict[str, Any]:
        hyper_out = self.latent_codec['hyper'](y)
        y_out = self.latent_codec['y'](y, hyper_out['params'])
        return {'likelihoods': {'y': y_out['likelihoods']['y'], 'z': hyper_out['likelihoods']['z']}, 'y_hat': y_out['y_hat']}

    def compress(self, y: 'Tensor') ->Dict[str, Any]:
        hyper_out = self.latent_codec['hyper'].compress(y)
        y_out = self.latent_codec['y'].compress(y, hyper_out['params'])
        [z_strings] = hyper_out['strings']
        return {'strings': [*y_out['strings'], z_strings], 'shape': {'y': y_out['shape'], 'hyper': hyper_out['shape']}, 'y_hat': y_out['y_hat']}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Dict[str, Tuple[int, ...]]', **kwargs) ->Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec['hyper'].decompress([z_strings], shape['hyper'])
        y_out = self.latent_codec['y'].decompress(y_strings_, shape['y'], hyper_out['params'])
        return {'y_hat': y_out['y_hat']}


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str='A', **kwargs: Any):
        super().__init__(*args, **kwargs)
        if mask_type not in ('A', 'B'):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: 'Tensor') ->Tensor:
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


def _reduce_seq(xs):
    assert all(x == xs[0] for x in xs)
    return xs[0]


K = TypeVar('K')


V = TypeVar('V')


def _ld_to_dl(ld: 'List[Dict[K, V]]') ->Dict[K, List[V]]:
    dl = {}
    for d in ld:
        for k, v in d.items():
            if k not in dl:
                dl[k] = []
            dl[k].append(v)
    return dl


def default_collate(batch: 'List[Dict[K, V]]') ->Dict[K, List[V]]:
    if not isinstance(batch, list) or any(not isinstance(d, dict) for d in batch):
        raise NotImplementedError
    result = _ld_to_dl(batch)
    for k, vs in result.items():
        if all(isinstance(v, Tensor) for v in vs):
            result[k] = torch.stack(vs)
    return result


def _pad_2d(x: 'Tensor', padding: 'int') ->Tensor:
    return F.pad(x, (padding, padding, padding, padding))


def raster_scan_compress_single_stream(encoder: 'BufferedRansEncoder', y: 'Tensor', params: 'Tensor', *, gaussian_conditional: GaussianConditional, entropy_parameters: nn.Module, context_prediction: MaskedConv2d, height: int, width: int, padding: int, kernel_size: int, merge: Callable[..., Tensor]=lambda *args: torch.cat(args, dim=1)) ->Tensor:
    """Compresses y and writes to encoder bitstream.

    Returns:
        The y_hat that will be reconstructed at the decoder.
    """
    assert height == y.shape[-2]
    assert width == y.shape[-1]
    cdf = gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = gaussian_conditional.cdf_length.tolist()
    offsets = gaussian_conditional.offset.tolist()
    masked_weight = context_prediction.weight * context_prediction.mask
    y_hat = _pad_2d(y, padding)
    symbols_list = []
    indexes_list = []
    for h in range(height):
        for w in range(width):
            y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
            ctx_p = F.conv2d(y_crop, masked_weight, context_prediction.bias)
            p = params[:, :, h:h + 1, w:w + 1]
            gaussian_params = entropy_parameters(merge(p, ctx_p))
            gaussian_params = gaussian_params.squeeze(3).squeeze(2)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            indexes = gaussian_conditional.build_indexes(scales_hat)
            y_crop = y_crop[:, :, padding, padding]
            symbols = gaussian_conditional.quantize(y_crop, 'symbols', means_hat)
            y_hat_item = symbols + means_hat
            hp = h + padding
            wp = w + padding
            y_hat[:, :, hp, wp] = y_hat_item
            symbols_list.extend(symbols.squeeze().tolist())
            indexes_list.extend(indexes.squeeze().tolist())
    encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
    y_hat = _pad_2d(y_hat, -padding)
    return y_hat


def raster_scan_decompress_single_stream(decoder: 'RansDecoder', params: 'Tensor', *, gaussian_conditional: GaussianConditional, entropy_parameters: nn.Module, context_prediction: MaskedConv2d, height: int, width: int, padding: int, kernel_size: int, device, merge: Callable[..., Tensor]=lambda *args: torch.cat(args, dim=1)) ->Tensor:
    """Decodes y_hat from decoder bitstream.

    Returns:
        The reconstructed y_hat.
    """
    cdf = gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = gaussian_conditional.cdf_length.tolist()
    offsets = gaussian_conditional.offset.tolist()
    masked_weight = context_prediction.weight * context_prediction.mask
    c = context_prediction.in_channels
    shape = 1, c, height + 2 * padding, width + 2 * padding
    y_hat = torch.zeros(shape, device=device)
    for h in range(height):
        for w in range(width):
            y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
            ctx_p = F.conv2d(y_crop, masked_weight, context_prediction.bias)
            p = params[:, :, h:h + 1, w:w + 1]
            gaussian_params = entropy_parameters(merge(p, ctx_p))
            gaussian_params = gaussian_params.squeeze(3).squeeze(2)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            indexes = gaussian_conditional.build_indexes(scales_hat)
            symbols = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
            symbols = Tensor(symbols).reshape(1, -1)
            y_hat_item = gaussian_conditional.dequantize(symbols, means_hat)
            hp = h + padding
            wp = w + padding
            y_hat[:, :, hp, wp] = y_hat_item
    y_hat = _pad_2d(y_hat, -padding)
    return y_hat


class RasterScanLatentCodec(LatentCodec):
    """Autoregression in raster-scan order with local decoded context.

    PixelCNN context model introduced in
    `"Pixel Recurrent Neural Networks"
    <http://arxiv.org/abs/1601.06759>`_,
    by Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu,
    International Conference on Machine Learning (ICML), 2016.

    First applied to learned image compression in
    `"Joint Autoregressive and Hierarchical Priors for Learned Image
    Compression" <https://arxiv.org/abs/1809.02736>`_,
    by D. Minnen, J. Balle, and G.D. Toderici,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                         ctx_params
                             │
                             ▼
                             │ ┌───◄───┐
                           ┌─┴─┴─┐  ┌──┴──┐
                           │  EP │  │  CP │
                           └──┬──┘  └──┬──┘
                              │        │
                              │        ▲
               ┌───┐  y_hat   ▼        │
        y ──►──┤ Q ├────►────····───►──┴──►── y_hat
               └───┘          GC

    """
    gaussian_conditional: 'GaussianConditional'
    entropy_parameters: 'nn.Module'
    context_prediction: 'MaskedConv2d'

    def __init__(self, gaussian_conditional: 'Optional[GaussianConditional]'=None, entropy_parameters: 'Optional[nn.Module]'=None, context_prediction: 'Optional[MaskedConv2d]'=None, **kwargs):
        super().__init__()
        self.gaussian_conditional = gaussian_conditional or GaussianConditional()
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or MaskedConv2d()
        self.kernel_size = _reduce_seq(self.context_prediction.kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, y: 'Tensor', params: 'Tensor') ->Dict[str, Any]:
        y_hat = self.gaussian_conditional.quantize(y, 'noise' if self.training else 'dequantize')
        ctx_params = self.merge(params, self.context_prediction(y_hat))
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        return {'likelihoods': {'y': y_likelihoods}, 'y_hat': y_hat}

    def compress(self, y: 'Tensor', ctx_params: 'Tensor') ->Dict[str, Any]:
        n, _, y_height, y_width = y.shape
        ds = [self._compress_single(y=y[i:i + 1, :, :, :], params=ctx_params[i:i + 1, :, :, :], gaussian_conditional=self.gaussian_conditional, entropy_parameters=self.entropy_parameters, context_prediction=self.context_prediction, height=y_height, width=y_width, padding=self.padding, kernel_size=self.kernel_size, merge=self.merge) for i in range(n)]
        return {**default_collate(ds), 'shape': y.shape[2:4]}

    def _compress_single(self, **kwargs):
        encoder = BufferedRansEncoder()
        y_hat = raster_scan_compress_single_stream(encoder=encoder, **kwargs)
        y_strings = encoder.flush()
        return {'strings': [y_strings], 'y_hat': y_hat.squeeze(0)}

    def decompress(self, strings: 'List[List[bytes]]', shape: 'Tuple[int, int]', ctx_params: 'Tensor', **kwargs) ->Dict[str, Any]:
        y_strings, = strings
        y_height, y_width = shape
        ds = [self._decompress_single(y_string=y_strings[i], params=ctx_params[i:i + 1, :, :, :], gaussian_conditional=self.gaussian_conditional, entropy_parameters=self.entropy_parameters, context_prediction=self.context_prediction, height=y_height, width=y_width, padding=self.padding, kernel_size=self.kernel_size, device=ctx_params.device, merge=self.merge) for i in range(len(y_strings))]
        return default_collate(ds)

    def _decompress_single(self, y_string, **kwargs):
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        y_hat = raster_scan_decompress_single_stream(decoder=decoder, **kwargs)
        return {'y_hat': y_hat.squeeze(0)}

    @staticmethod
    def merge(*args):
        return torch.cat(args, dim=1)


class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self.func})'

    def forward(self, x):
        return self.func(x)


class NamedLayer(nn.Module):

    def __init__(self, name: 'str'):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def forward(self, x):
        return x


class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.shape})'

    def forward(self, x):
        output_shape = x.shape[0], *self.shape
        try:
            return x.reshape(output_shape)
        except RuntimeError as e:
            e.args += f'Cannot reshape input {tuple(x.shape)} to {output_shape}',
            raise e


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def __repr__(self):
        return f'{self.__class__.__name__}(dim0={self.dim0}, dim1={self.dim1})'

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1).contiguous()


class Interleave(nn.Module):

    def __init__(self, groups: 'int'):
        super().__init__()
        self.groups = groups

    def forward(self, x: 'Tensor') ->Tensor:
        g = self.groups
        n, c, *tail = x.shape
        return x.reshape(n, g, c // g, *tail).transpose(1, 2).reshape(x.shape)


class Gain(nn.Module):

    def __init__(self, shape=None, factor: 'float'=1.0):
        super().__init__()
        self.factor = factor
        self.gain = nn.Parameter(torch.ones(shape))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.factor * self.gain * x


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """
    pedestal: 'Tensor'

    def __init__(self, minimum: 'float'=0, reparam_offset: 'float'=2 ** -18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset ** 2
        self.register_buffer('pedestal', torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: 'Tensor') ->Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: 'Tensor') ->Tensor:
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out


class GDN(nn.Module):
    """Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \\frac{x[i]}{\\sqrt{\\beta[i] + \\sum_j(\\gamma[j, i] * x[j]^2)}}

    """

    def __init__(self, in_channels: 'int', inverse: 'bool'=False, beta_min: 'float'=1e-06, gamma_init: 'float'=0.1):
        super().__init__()
        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: 'Tensor') ->Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)
        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        out = x * norm
        return out


class GDN1(GDN):
    """Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \\frac{x[i]}{\\beta[i] + \\sum_j(\\gamma[j, i] * |x[j]|}

    """

    def forward(self, x: 'Tensor') ->Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)
        if not self.inverse:
            norm = 1.0 / norm
        out = x * norm
        return out


class _SpectralConvNdMixin:

    def __init__(self, dim: 'Tuple[int, ...]'):
        self.dim = dim
        self.weight_transformed = nn.Parameter(self._to_transform_domain(self.weight))
        del self._parameters['weight']

    @property
    def weight(self) ->Tensor:
        return self._from_transform_domain(self.weight_transformed)

    def _to_transform_domain(self, x: 'Tensor') ->Tensor:
        return torch.fft.rfftn(x, s=self.kernel_size, dim=self.dim, norm='ortho')

    def _from_transform_domain(self, x: 'Tensor') ->Tensor:
        return torch.fft.irfftn(x, s=self.kernel_size, dim=self.dim, norm='ortho')


class SpectralConv2d(nn.Conv2d, _SpectralConvNdMixin):
    """Spectral 2D convolution.

    Introduced in [Balle2018efficient].
    Reparameterizes the weights to be derived from weights stored in the
    frequency domain.
    In the original paper, this is referred to as "spectral Adam" or
    "Sadam" due to its effect on the Adam optimizer update rule.
    The motivation behind representing the weights in the frequency
    domain is that optimizer updates/steps may now affect all
    frequencies to an equal amount.
    This improves the gradient conditioning, thus leading to faster
    convergence and increased stability at larger learning rates.

    For comparison, see the TensorFlow Compression implementations of
    `SignalConv2D
    <https://github.com/tensorflow/compression/blob/v2.14.0/tensorflow_compression/python/layers/signal_conv.py#L61>`_
    and
    `RDFTParameter
    <https://github.com/tensorflow/compression/blob/v2.14.0/tensorflow_compression/python/layers/parameters.py#L71>`_.

    [Balle2018efficient]: `"Efficient Nonlinear Transforms for Lossy
    Image Compression" <https://arxiv.org/abs/1802.00847>`_,
    by Johannes Ballé, PCS 2018.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _SpectralConvNdMixin.__init__(self, dim=(-2, -1))


class SpectralConvTranspose2d(nn.ConvTranspose2d, _SpectralConvNdMixin):
    """Spectral 2D transposed convolution.

    Transposed version of :class:`SpectralConv2d`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _SpectralConvNdMixin.__init__(self, dim=(-2, -1))


class CheckerboardMaskedConv2d(MaskedConv2d):
    """Checkerboard masked 2D convolution; mask future "unseen" pixels.

    Checkerboard mask variant used in
    `"Checkerboard Context Model for Efficient Learned Image Compression"
    <https://arxiv.org/abs/2103.15306>`_, by Dailan He, Yaoyan Zheng,
    Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str='A', **kwargs: Any):
        super().__init__(*args, **kwargs)
        if mask_type not in ('A', 'B'):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        _, _, h, w = self.mask.size()
        self.mask[:] = 1
        self.mask[:, :, 0::2, 0::2] = 0
        self.mask[:, :, 1::2, 1::2] = 0
        self.mask[:, :, h // 2, w // 2] = mask_type == 'B'


def conv1x1(in_ch: 'int', out_ch: 'int', stride: 'int'=1) ->nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: 'int', out_ch: 'int', stride: 'int'=1) ->nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: 'int', out_ch: 'int', stride: 'int'=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        if self.skip is not None:
            identity = self.skip(x)
        out += identity
        return out


def subpel_conv3x3(in_ch: 'int', out_ch: 'int', r: 'int'=1) ->nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r))


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: 'int', out_ch: 'int', upsample: 'int'=2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: 'int', out_ch: 'int'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: 'int'):
        super().__init__()


        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(conv1x1(N, N // 2), nn.ReLU(inplace=True), conv3x3(N // 2, N // 2), nn.ReLU(inplace=True), conv1x1(N // 2, N))
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: 'Tensor') ->Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(N, N))

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class DensityEmbeddingLayer(nn.Module):
    """Density embedding for downsampling, as introduced in [He2022pcc]_.

    Applies an embedding ℝ → ℝᶜ to the local point density (scalar).
    The local point density is measured using the mean distance of the
    points within the neighborhood of a "downsampled" centroid.
    This information is useful when upsampling from the single centroid.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, dim, ngroups):
        super().__init__()
        self.embed_densities = nn.Sequential(nn.Conv1d(1, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv1d(hidden_dim, dim, 1))

    def forward(self, downsample_num):
        density_embedding = self.embed_densities(downsample_num)
        return density_embedding


def index_points(xyzs, idx):
    """Index points.

    Args:
        xyzs: (b, c, n)
        idx: (b, ...)

    Returns:
        xyzs_out: (b, c, ...)
    """
    _, c, _ = xyzs.shape
    b, *idx_dims = idx.shape
    idx_out = idx.reshape(b, 1, -1).repeat(1, c, 1)
    xyzs_out = xyzs.gather(2, idx_out).reshape(b, c, *idx_dims)
    return xyzs_out


class PointTransformerLayer(nn.Module):
    """Point Transformer layer introduced by [Zhao2021]_.

    References:

        .. [Zhao2021] `"Point Transformer"
            <https://arxiv.org/abs/2012.09164>`_, by Hengshuang Zhao,
            Li Jiang, Jiaya Jia, Philip Torr, and Vladlen Koltun,
            CVPR 2021.
    """

    def __init__(self, in_fdim, out_fdim, hidden_dim, ngroups):
        super().__init__()
        self.w_qs = nn.Conv1d(in_fdim, hidden_dim, 1)
        self.w_ks = nn.Conv1d(in_fdim, hidden_dim, 1)
        self.w_vs = nn.Conv1d(in_fdim, hidden_dim, 1)
        self.conv_delta = nn.Sequential(nn.Conv2d(3, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.conv_gamma = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.post_conv = nn.Conv1d(hidden_dim, out_fdim, 1)

    def forward(self, q_xyzs, k_xyzs, q_feats, k_feats, v_feats, knn_idx, mask):
        knn_xyzs = index_points(k_xyzs, knn_idx)
        identity = q_feats
        query = self.w_qs(q_feats)
        key = index_points(self.w_ks(k_feats), knn_idx)
        value = index_points(self.w_vs(v_feats), knn_idx)
        pos_enc = self.conv_delta(q_xyzs.unsqueeze(-1) - knn_xyzs)
        attn = self.conv_gamma(query.unsqueeze(-1) - key + pos_enc)
        attn = attn / math.sqrt(key.shape[1])
        mask_value = -torch.finfo(attn.dtype).max
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)
        result = torch.einsum('bcmk, bcmk -> bcm', attn, value + pos_enc)
        result = self.post_conv(result) + identity
        return result


class PositionEmbeddingLayer(nn.Module):
    """Position embedding for downsampling, as introduced in [He2022pcc]_.

    For each group of feature vectors (f₁, ..., fₖ) with centroid fₒ,
    represents the offsets (f₁ - fₒ, ..., fₖ - fₒ) as
    magnitude-direction vectors, then applies an MLP to each vector,
    then takes a softmax self-attention over the resulting vectors,
    and finally reduces the vectors via a sum,
    resulting in a single embedded vector for the group.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, dim, ngroups):
        super().__init__()
        self.embed_positions = nn.Sequential(nn.Conv2d(4, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, dim, 1))
        self.attention = nn.Sequential(nn.Conv2d(dim, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, dim, 1))

    def forward(self, q_xyzs, k_xyzs, knn_idx, mask):
        _, _, k = knn_idx.shape
        knn_xyzs = index_points(k_xyzs, knn_idx)
        repeated_xyzs = q_xyzs[..., None].repeat(1, 1, 1, k)
        offset_xyzs = knn_xyzs - repeated_xyzs
        direction = F.normalize(offset_xyzs, p=2, dim=1)
        distance = torch.linalg.norm(offset_xyzs, dim=1, keepdim=True)
        local_pattern = torch.cat((direction, distance), dim=1)
        position_embedding_expanded = self.embed_positions(local_pattern)
        attn = self.attention(position_embedding_expanded)
        mask_value = -torch.finfo(attn.dtype).max
        attn.masked_fill_(~mask[:, None], mask_value)
        attn = F.softmax(attn, dim=-1)
        position_embedding = (position_embedding_expanded * attn).sum(dim=-1)
        return position_embedding


def nearby_distance_sum(a_xyzs, b_xyzs, k):
    """Computes sum of nearby distances to B for each point in A.

    Partitions a point set B into non-intersecting sets
    C(a_1), ..., C(a_m) where each C(a_i) contains points that are
    nearest to a_i ∈ A.
    For each a_i ∈ A, computes the total distance from a_i to C(a_i).
    (Note that C(a_1), ..., C(a_m) may not cover all of B.)

    In more precise terms:
    For each a ∈ A, let C(a) ⊆ B denote its "collapsed point set" s.t.
    (i)   b ∈ C(a)  ⇒  min_{a' ∈ A} ||a' - b|| = ||a - b||,
    (ii)  ⋃ _{a ∈ A} C(a) ⊆ B,
    (iii) ⋂ _{a ∈ A} C(a) = ∅, and
    (iv)  |C(a)| ≤ k.
    For each a ∈ A, we then compute d(a) = ∑_{b ∈ C(a)} ||a - b||.

    Args:
        a_xyzs: (b, 3, m) Input point set A.
        b_xyzs: (b, 3, n) Input point set B.
        k: Maximum number of points in each collapsed point set C(a_i).

    Returns:
        distance: (b, m) Sum of distances from each point in A to its
            collapsed point set.
        mask: (b, m, k) Mask indicating which points in the ``knn_idx``
            belong to the collapsed point set of each point in A.
        knn_idx: (b, m, k) Indices of the points in B that are nearest
            to each point in A.
        nn_idx: (b, n, 1) Indices of the point in A that is nearest
            to each point in B.
    """
    device = a_xyzs.device
    _, _, m = a_xyzs.shape
    a_xyzs_tr = a_xyzs.permute(0, 2, 1).contiguous()
    b_xyzs_tr = b_xyzs.permute(0, 2, 1).contiguous()
    nn_idx = pointops.knnquery_heap(1, a_xyzs_tr, b_xyzs_tr)
    nn_idx_tr = nn_idx.permute(0, 2, 1).contiguous()
    knn_idx = pointops.knnquery_heap(k, b_xyzs_tr, a_xyzs_tr).long()
    torch.cuda.empty_cache()
    expect_idx = torch.arange(m, device=device)[None, :, None]
    actual_idx = index_points(nn_idx_tr, knn_idx).squeeze(1)
    mask = expect_idx == actual_idx
    knn_xyzs = index_points(b_xyzs, knn_idx)
    knn_distances = torch.linalg.norm(knn_xyzs - a_xyzs[..., None], dim=1)
    knn_distances = knn_distances * mask.float()
    distance = knn_distances.sum(dim=-1)
    return distance, mask, knn_idx, nn_idx


class DownsampleLayer(nn.Module):
    """Downsampling layer used in [He2022pcc]_.

    Downsamples positions into a smaller number of centroids.
    Each centroid is grouped with nearby points,
    and the local point density is estimated for that group.
    Then, the positions, features, and density for the group
    are embedded into a single aggregate vector from which the
    group of points may later be reconstructed.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, downsample_rate, dim, hidden_dim, k, ngroups):
        super().__init__()
        self.k = k
        self.downsample_rate = downsample_rate
        self.pre_conv = nn.Conv1d(dim, dim, 1)
        self.embed_features = PointTransformerLayer(dim, dim, hidden_dim, ngroups)
        self.embed_positions = PositionEmbeddingLayer(hidden_dim, dim, ngroups)
        self.embed_densities = DensityEmbeddingLayer(hidden_dim, dim, ngroups)
        self.post_conv = nn.Conv1d(dim * 3, dim, 1)

    def get_density(self, downsampled_xyzs, input_xyzs):
        _, _, n = input_xyzs.shape
        distance, mask, knn_idx, _ = nearby_distance_sum(downsampled_xyzs, input_xyzs, min(self.k, n))
        downsample_num = mask.sum(dim=-1).float()
        mean_distance = distance / downsample_num
        return downsample_num, mean_distance, mask, knn_idx

    def forward(self, xyzs, feats):
        sampled_xyzs, sample_idx = self.downsample_positions(xyzs, feats)
        downsample_num, mean_distance, mask, knn_idx = self.get_density(sampled_xyzs, xyzs)
        sampled_feats = self.downsample_features(sampled_xyzs, xyzs, feats, downsample_num, sample_idx, knn_idx, mask)
        return sampled_xyzs, sampled_feats, downsample_num, mean_distance

    def downsample_positions(self, xyzs, sample_num):
        _, _, n = xyzs.shape
        sample_num = round(n * self.downsample_rate)
        xyzs_tr = xyzs.permute(0, 2, 1).contiguous()
        sample_idx = pointops.furthestsampling(xyzs_tr, sample_num).long()
        sampled_xyzs = index_points(xyzs, sample_idx)
        return sampled_xyzs, sample_idx

    def downsample_features(self, sampled_xyzs, xyzs, feats, downsample_num, sample_idx, knn_idx, mask):
        identity = index_points(feats, sample_idx)
        feats = self.pre_conv(feats)
        sampled_feats = index_points(feats, sample_idx)
        embeddings = [self.embed_features(sampled_xyzs, xyzs, sampled_feats, feats, feats, knn_idx, mask), self.embed_positions(sampled_xyzs, xyzs, knn_idx, mask), self.embed_densities(downsample_num.unsqueeze(1))]
        agg_embedding = self.post_conv(torch.cat(embeddings, dim=1))
        sampled_feats_new = agg_embedding + identity
        return sampled_feats_new


class EdgeConv(nn.Module):
    """EdgeConv introduced by [Wang2019dgcnn]_.

    First, groups similar feature vectors together via k-nearest neighbors
    using the following distance metric between feature vectors fᵢ and fⱼ:
    distance[i, j] = 2fᵢᵀfⱼ - ||fᵢ||² - ||fⱼ||².

    Then, for each group of feature vectors (f₁, ..., fₖ) with centroid fₒ,
    the residual feature vectors are each concatenated with the centroid,
    then an MLP is applied to each resulting vector individually,
    i.e., (MLP(f₁ - fₒ, fₒ), ..., MLP(fₖ - fₒ, fₒ)),
    and finally the elementwise max is taken across the resulting vectors,
    resulting in a single vector fₘₐₓ for the group.

    Original code located at [DGCNN]_ under MIT License.

    References:

        .. [Wang2019dgcnn] `"Dynamic Graph CNN for Learning on Point Clouds"
            <https://arxiv.org/abs/1801.07829>`_, by Yue Wang, Yongbin
            Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein,
            Justin M. Solomon, ACM Transactions on Graphics 2019.

        .. [DGCNN] `DGCNN
            <https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py>`_
    """

    def __init__(self, in_fdim, out_fdim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(2 * in_fdim, hidden_dim, 1), nn.ReLU(), nn.Conv2d(hidden_dim, out_fdim, 1))

    def knn(self, feats, k):
        sq_norm = (feats ** 2).sum(dim=1, keepdim=True)
        pairwise_dot = torch.matmul(feats.transpose(2, 1), feats)
        pairwise_distance = 2 * pairwise_dot - sq_norm - sq_norm.transpose(2, 1)
        _, knn_idx = pairwise_distance.topk(k=k, dim=-1)
        return knn_idx

    def get_graph_features(self, feats, k):
        dim = feats.shape[1]
        if dim == 3:
            feats_tr = feats.permute(0, 2, 1).contiguous()
            knn_idx = pointops.knnquery_heap(k, feats_tr, feats_tr).long()
        else:
            knn_idx = self.knn(feats, k)
        torch.cuda.empty_cache()
        knn_feats = index_points(feats, knn_idx)
        repeated_feats = repeat(feats, 'b c n -> b c n k', k=k)
        graph_feats = torch.cat((knn_feats - repeated_feats, repeated_feats), dim=1)
        return graph_feats

    def forward(self, feats):
        _, _, n = feats.shape
        graph_feats = self.get_graph_features(feats, k=min(self.k, n))
        expanded_feats = self.conv(graph_feats)
        feats_new, _ = expanded_feats.max(dim=-1)
        return feats_new


class SubPointConv(nn.Module):
    """Sub-point convolution for upsampling, as introduced in [He2022pcc]_.

    Each feature vector (representing a "centroid" point) is sliced
    into g feature vectors, where each feature vector represents a
    point that has been upsampled from the original centroid point.
    Then, an MLP is applied to each slice individually.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, hidden_dim, k, mode, in_fdim, out_fdim, group_num):
        super().__init__()
        self.mode = mode
        self.group_num = group_num
        group_in_fdim = in_fdim // group_num
        group_out_fdim = out_fdim // group_num
        if self.mode == 'mlp':
            self.mlp = nn.Sequential(nn.Conv2d(group_in_fdim, hidden_dim, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, group_out_fdim, 1))
        elif self.mode == 'edge_conv':
            self.edge_conv = EdgeConv(in_fdim, out_fdim, hidden_dim, k)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def forward(self, feats):
        if self.mode == 'mlp':
            feats = rearrange(feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()
            expanded_feats = self.mlp(feats)
        elif self.mode == 'edge_conv':
            expanded_feats = self.edge_conv(feats)
            expanded_feats = rearrange(expanded_feats, 'b (c g) n -> b c n g', g=self.group_num).contiguous()
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return expanded_feats


class FeatsUpsampleLayer(nn.Module):
    """Feature upsampling layer used in [He2022pcc]_.

    Upsamples many features from each "centroid" feature vector.
    The feature vector associated with each centroid is upsampled
    into various candidate upsampled feature vectors.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, upsample_rate, decompress_normal=False):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.decompress_normal = decompress_normal
        out_fdim = (3 if decompress_normal else dim) * upsample_rate
        self.feats_nn = SubPointConv(hidden_dim, k, sub_point_conv_mode, dim, out_fdim, upsample_rate)

    def forward(self, feats):
        upsampled_feats = self.feats_nn(feats)
        if not self.decompress_normal:
            repeated_feats = repeat(feats, 'b c n -> b c n u', u=self.upsample_rate)
            upsampled_feats = upsampled_feats + repeated_feats
        return upsampled_feats


def icosahedron2sphere(level):
    """Samples uniformly on a sphere using a icosahedron.

    Code adapted from [IcoSphere_MATLAB]_ and [IcoSphere_Python]_,
    from paper [Xiao2009]_.

    References:

        .. [Xiao2009] `"Image-based street-side city modeling"
            <https://dl.acm.org/doi/10.1145/1618452.1618460>`_,
            by Jianxiong Xiao, Tian Fang, Peng Zhao, Maxime Lhuillier,
            and Long Quan, ACM Transactions on Graphics, 2009.

        .. [IcoSphere_MATLAB] https://github.com/jianxiongxiao/ProfXkit/blob/master/icosahedron2sphere/icosahedron2sphere.m

        .. [IcoSphere_Python] https://github.com/23michael45/PanoContextTensorflow/blob/master/PanoContextTensorflow/icosahedron2sphere.py
    """
    a = 2 / (1 + np.sqrt(5))
    M = np.array([0, a, -1, a, 1, 0, -a, 1, 0, 0, a, 1, -a, 1, 0, a, 1, 0, 0, a, 1, 0, -a, 1, -1, 0, a, 0, a, 1, 1, 0, a, 0, -a, 1, 0, a, -1, 0, -a, -1, 1, 0, -a, 0, a, -1, -1, 0, -a, 0, -a, -1, 0, -a, 1, a, -1, 0, -a, -1, 0, 0, -a, -1, -a, -1, 0, a, -1, 0, -a, 1, 0, -1, 0, a, -1, 0, -a, -a, -1, 0, -1, 0, -a, -1, 0, a, a, 1, 0, 1, 0, -a, 1, 0, a, a, -1, 0, 1, 0, a, 1, 0, -a, 0, a, 1, -1, 0, a, -a, 1, 0, 0, a, 1, a, 1, 0, 1, 0, a, 0, a, -1, -a, 1, 0, -1, 0, -a, 0, a, -1, 1, 0, -a, a, 1, 0, 0, -a, -1, -1, 0, -a, -a, -1, 0, 0, -a, -1, a, -1, 0, 1, 0, -a, 0, -a, 1, -a, -1, 0, -1, 0, a, 0, -a, 1, 1, 0, a, a, -1, 0])
    coor = M.reshape(60, 3)
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(20, 3)
    coor_norm = np.linalg.norm(coor, axis=1, keepdims=True)
    coor = list(coor / np.tile(coor_norm, (1, 3)))
    for _ in range(level):
        tris = []
        for t in range(len(tri)):
            n = len(coor)
            coor.extend([(coor[tri[t, 0]] + coor[tri[t, 1]]) / 2, (coor[tri[t, 1]] + coor[tri[t, 2]]) / 2, (coor[tri[t, 2]] + coor[tri[t, 0]]) / 2])
            tris.extend([[n, tri[t, 0], n + 2], [n, tri[t, 1], n + 1], [n + 1, tri[t, 2], n + 2], [n, n + 1, n + 2]])
        tri = np.array(tris)
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]
        coor_norm = np.linalg.norm(coor, axis=1, keepdims=True)
        coor = list(coor / np.tile(coor_norm, (1, 3)))
    return np.array(coor), np.array(tri)


class XyzsUpsampleLayer(nn.Module):
    """Position upsampling layer used in [He2022pcc]_.

    Upsamples many positions from each "centroid" feature vector.
    Each feature vector is upsampled into various offsets represented as
    magnitude-direction vectors, where each direction is determined by a
    weighted sum of various fixed hypothesized directions.
    From this, the candidate upsampled positions are simply the
    the offset vectors plus their original centroid position.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, upsample_rate):
        super().__init__()
        self.upsample_rate = upsample_rate
        hypothesis, _ = icosahedron2sphere(1)
        hypothesis = np.append(np.zeros((1, 3)), hypothesis, axis=0)
        self.hypothesis = torch.from_numpy(hypothesis).float()
        self.weight_nn = SubPointConv(hidden_dim, k, sub_point_conv_mode, dim, 43 * upsample_rate, upsample_rate)
        self.scale_nn = SubPointConv(hidden_dim, k, sub_point_conv_mode, dim, 1 * upsample_rate, upsample_rate)

    def forward(self, xyzs, feats):
        batch_size = xyzs.shape[0]
        points_num = xyzs.shape[2]
        weights = self.weight_nn(feats)
        weights = weights.unsqueeze(2)
        weights = F.softmax(weights, dim=1)
        hypothesis = repeat(self.hypothesis, 'h c -> b h c n u', b=batch_size, n=points_num, u=self.upsample_rate)
        weighted_hypothesis = weights * hypothesis
        directions = torch.sum(weighted_hypothesis, dim=1)
        directions = F.normalize(directions, p=2, dim=1)
        scales = self.scale_nn(feats)
        deltas = directions * scales
        repeated_xyzs = repeat(xyzs, 'b c n -> b c n u', u=self.upsample_rate)
        upsampled_xyzs = repeated_xyzs + deltas
        return upsampled_xyzs


class UpsampleLayer(nn.Module):
    """Upsampling layer used in [He2022pcc]_.

    Upsamples many candidate points from a smaller number of centroids.
    (Not all candidate upsampled points will be kept; some will be
    thrown away to match the predicted local point density.)

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, upsample_rate):
        super().__init__()
        self.xyzs_upsample_nn = XyzsUpsampleLayer(dim, hidden_dim, k, sub_point_conv_mode, upsample_rate)
        self.feats_upsample_nn = FeatsUpsampleLayer(dim, hidden_dim, k, sub_point_conv_mode, upsample_rate)

    def forward(self, xyzs, feats):
        upsampled_xyzs = self.xyzs_upsample_nn(xyzs, feats)
        upsampled_feats = self.feats_upsample_nn(feats)
        return upsampled_xyzs, upsampled_feats


class UpsampleNumLayer(nn.Module):
    """Predicts local point density while upsampling, as used in [He2022pcc]_.

    Extracts the number of candidate points to keep after upsampling
    from a given "centroid" feature vector.
    (Some candidate upsampled points will be thrown away to match the
    predicted local point density.)

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, upsample_rate):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.upsample_num_nn = nn.Sequential(nn.Conv1d(dim, hidden_dim, 1), nn.ReLU(), nn.Conv1d(hidden_dim, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        upsample_frac = self.upsample_num_nn(feats).squeeze(1)
        upsample_num = upsample_frac * (self.upsample_rate - 1) + 1
        return upsample_num


class RefineLayer(nn.Module):
    """Refines upsampled points, as used in [He2022pcc]_.

    After the centroids are upsampled, there may be overlapping
    point groups between nearby centroids, and other artifacts.
    Refinement should help correct various such artifacts.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, dim, hidden_dim, k, sub_point_conv_mode, decompress_normal):
        super().__init__()
        self.xyzs_refine_nn = XyzsUpsampleLayer(dim, hidden_dim, k, sub_point_conv_mode, upsample_rate=1)
        self.feats_refine_nn = FeatsUpsampleLayer(dim, hidden_dim, k, sub_point_conv_mode, upsample_rate=1, decompress_normal=decompress_normal)

    def forward(self, xyzs, feats):
        refined_xyzs = self.xyzs_refine_nn(xyzs, feats)
        refined_xyzs = rearrange(refined_xyzs, 'b c n u -> b c (n u)')
        refined_feats = self.feats_refine_nn(feats)
        refined_feats = rearrange(refined_feats, 'b c n u -> b c (n u)')
        return refined_xyzs, refined_feats


def _farthest_point_sample_yanx27(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 10000000000.0
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample(xyz, npoint, _method='pointops'):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if _method == 'pointops':
        xyz_tr = xyz.permute(0, 2, 1).contiguous()
        indices = pointops.furthestsampling(xyz_tr, npoint).long()
        return indices
    if _method == 'pytorch3d':
        sampled_points, indices = sample_farthest_points(xyz, K=npoint)
        return indices
    if _method.startswith('fpsample'):
        if _method == 'fpsample.npdu_kdtree':
            func = fpsample.fps_npdu_kdtree_sampling
        if _method == 'fpsample.bucket':
            func = lambda *args: fpsample.bucket_fps_kdline_sampling(*args, h=5)
        with ThreadPoolExecutor(max_workers=min(8, len(xyz))) as executor:
            indices = list(executor.map(lambda pc: func(pc, npoint), xyz.cpu().numpy()))
        indices = torch.from_numpy(np.stack(indices, dtype=np.int64))
        return indices
    if _method == 'yanx27':
        return _farthest_point_sample_yanx27(xyz, npoint)
    raise ValueError(f'Unknown method {_method}')


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _query_ball_point_yanx27(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def query_ball_point(radius, nsample, xyz, new_xyz, _method='pointops'):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    if _method == 'pointops':
        idx = pointops.ballquery(radius, nsample, xyz, new_xyz).long()
        return idx
    if _method == 'pytorch3d':
        dists, idx, neighbors = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=False)
        return idx
    if _method == 'yanx27':
        return _query_ball_point_yanx27(radius, nsample, xyz, new_xyz)
    raise ValueError(f'Unknown method {_method}')


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, returnfps=False):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    fps_idx = torch.zeros(B, 1, dtype=torch.long)
    new_xyz = torch.zeros(B, 1, C)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            _, D, _ = points.shape
            points = points.permute(0, 2, 1)
        else:
            D = 0
        if self.group_all:
            new_xyz, grouped_points, grouped_xyz, idx = sample_and_group_all(xyz, points, returnfps=True)
            npoint = 1
            nsample = N
        else:
            new_xyz, grouped_points, grouped_xyz, idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)
            npoint = self.npoint
            nsample = self.nsample
        assert grouped_xyz.shape == (B, npoint, nsample, C)
        assert grouped_points.shape == (B, npoint, nsample, C + D)
        assert new_xyz.shape == (B, npoint, C)
        assert idx.shape == (B, npoint)
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)
        grouped_points = grouped_points.permute(0, 3, 1, 2)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = grouped_points.permute(0, 1, 3, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return {'grouped_xyz': grouped_xyz, 'grouped_features': grouped_points, 'new_xyz': new_xyz, 'new_features': new_points, 'idx': idx}


class PointNetSetAbstractionMsg(nn.Module):

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class UpsampleBlock(nn.Module):

    def __init__(self, D, E, M, P, S, i, extra_in_ch=3, groups=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(E[i + 1] + (D[i] + extra_in_ch) * bool(M[i]), D[i], 1, groups=groups[0]), Interleave(groups=groups[0]), nn.BatchNorm1d(D[i]), nn.ReLU(inplace=True), nn.Conv1d(D[i], E[i] * S[i], 1, groups=groups[1]), Interleave(groups=groups[1]), nn.BatchNorm1d(E[i] * S[i]), nn.ReLU(inplace=True), Reshape((E[i], S[i], P[i])), Transpose(-2, -1), Reshape((E[i], P[i] * S[i])))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.block(x)


def chamfer_distance(xyzs1, xyzs2, order='b n c'):
    xyzs1_bcn = rearrange(xyzs1, f'{order} -> b c n').contiguous()
    xyzs1_bnc = rearrange(xyzs1, f'{order} -> b n c').contiguous()
    xyzs2_bcn = rearrange(xyzs2, f'{order} -> b c n').contiguous()
    xyzs2_bnc = rearrange(xyzs2, f'{order} -> b n c').contiguous()
    idx1 = pointops.knnquery_heap(1, xyzs2_bnc, xyzs1_bnc).long().squeeze(2)
    idx2 = pointops.knnquery_heap(1, xyzs1_bnc, xyzs2_bnc).long().squeeze(2)
    torch.cuda.empty_cache()
    dist1 = ((xyzs1_bcn - index_points(xyzs2_bcn, idx1)) ** 2).sum(1)
    dist2 = ((xyzs2_bcn - index_points(xyzs1_bcn, idx2)) ** 2).sum(1)
    return dist1, dist2, idx1, idx2


class ChamferPccRateDistortionLoss(nn.Module):
    """Simple loss for regular point cloud compression.

    For compression models that reconstruct the input point cloud.
    """
    LMBDA_DEFAULT = {'rec': 1.0}

    def __init__(self, lmbda=None, rate_key='bpp'):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.lmbda.setdefault(rate_key, 1.0)

    def forward(self, output, target):
        out = {**self.compute_rate_loss(output, target), **self.compute_rec_loss(output, target)}
        out['loss'] = sum(self.lmbda[k] * out[f'{k}_loss'] for k in self.lmbda.keys() if f'{k}_loss' in out)
        return out

    def compute_rate_loss(self, output, target):
        if 'likelihoods' not in output:
            return {}
        N, P, _ = target['pos'].shape
        return compute_rate_loss(output['likelihoods'], N, P)

    def compute_rec_loss(self, output, target):
        dist1, dist2, _, _ = chamfer_distance(target['pos'], output['x_hat'], order='b n c')
        loss_chamfer = dist1.mean() + dist2.mean()
        return {'rec_loss': loss_chamfer}


def compute_rate_loss(likelihoods, batch_size, bit_per_bpp):
    out_bit = {f'bit_{name}_loss': (lh.log2().sum() / -batch_size) for name, lh in likelihoods.items()}
    out_bpp = {f'bpp_{name}_loss': (out_bit[f'bit_{name}_loss'] / bit_per_bpp) for name in likelihoods.keys()}
    out = {**out_bit, **out_bpp}
    out['bit_loss'] = sum(out_bit.values())
    out['bpp_loss'] = out['bit_loss'] / bit_per_bpp
    return out


def get_chamfer_loss(gt_xyzs_, xyzs_hat_):
    num_layers = len(gt_xyzs_)
    chamfer_loss_ = []
    nearest_gt_idx_ = []
    for i in range(num_layers):
        xyzs1 = gt_xyzs_[i]
        xyzs2 = xyzs_hat_[num_layers - i - 1]
        dist1, dist2, _, idx2 = chamfer_distance(xyzs1, xyzs2, order='b c n')
        chamfer_loss_.append(dist1.mean() + dist2.mean())
        nearest_gt_idx_.append(idx2.long())
    return chamfer_loss_, nearest_gt_idx_


def get_density_loss(gt_dnums_, gt_mdis_, unums_hat_, mdis_hat_, nearest_gt_idx_):
    num_layers = len(gt_dnums_)
    l1_loss = nn.L1Loss(reduction='mean')
    mean_distance_loss_ = []
    upsample_num_loss_ = []
    for i in range(num_layers):
        if i == num_layers - 1:
            mdis_i = gt_mdis_[i]
            dnum_i = gt_dnums_[i]
        else:
            idx = nearest_gt_idx_[i + 1]
            mdis_i = index_points(gt_mdis_[i].unsqueeze(1), idx).squeeze(1)
            dnum_i = index_points(gt_dnums_[i].unsqueeze(1), idx).squeeze(1)
        mean_distance_loss_.append(l1_loss(mdis_hat_[num_layers - i - 1], mdis_i))
        upsample_num_loss_.append(l1_loss(unums_hat_[num_layers - i - 1], dnum_i))
    return sum(mean_distance_loss_), sum(upsample_num_loss_)


def get_latent_xyzs_loss(gt_latent_xyzs, latent_xyzs_hat):
    return F.mse_loss(gt_latent_xyzs, latent_xyzs_hat)


def get_normal_loss(gt_normals, pred_normals, nearest_gt_idx):
    nearest_normal = index_points(gt_normals, nearest_gt_idx)
    return F.mse_loss(pred_normals, nearest_normal)


def get_pts_num_loss(gt_xyzs_, unums_hat_):
    num_layers = len(gt_xyzs_)
    b, _, _ = gt_xyzs_[0].shape
    gt_num_points_ = [x.shape[2] for x in gt_xyzs_]
    return sum(torch.abs(unums_hat_[num_layers - i - 1].sum() - gt_num_points_[i] * b) for i in range(num_layers))


class RateDistortionLoss_hrtzxf2022(nn.Module):
    """Loss introduced in [He2022pcc]_ for "hrtzxf2022-pcc-rec" model.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """
    LMBDA_DEFAULT = {'bpp': 1.0, 'chamfer': 10000.0, 'chamfer_layers': (1.0, 0.1, 0.1), 'latent_xyzs': 100.0, 'mean_distance': 50.0, 'normal': 100.0, 'pts_num': 0.005, 'upsample_num': 1.0}

    def __init__(self, lmbda=None, compress_normal=False, latent_xyzs_codec_mode='learned'):
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.compress_normal = compress_normal
        self.latent_xyzs_codec_mode = latent_xyzs_codec_mode

    def forward(self, output, target):
        device = target['pos'].device
        B, P, _ = target['pos'].shape
        out = {}
        chamfer_loss_, nearest_gt_idx_ = get_chamfer_loss(output['gt_xyz_'], output['xyz_hat_'])
        out['chamfer_loss'] = sum(self.lmbda['chamfer_layers'][i] * chamfer_loss_[i] for i in range(len(chamfer_loss_)))
        out['rec_loss'] = chamfer_loss_[0]
        out['mean_distance_loss'], out['upsample_num_loss'] = get_density_loss(output['gt_downsample_num_'], output['gt_mean_distance_'], output['upsample_num_hat_'], output['mean_distance_hat_'], nearest_gt_idx_)
        out['pts_num_loss'] = get_pts_num_loss(output['gt_xyz_'], output['upsample_num_hat_'])
        if self.latent_xyzs_codec_mode == 'learned':
            out['latent_xyzs_loss'] = get_latent_xyzs_loss(output['gt_latent_xyz'], output['latent_xyz_hat'])
        elif self.latent_xyzs_codec_mode == 'float16':
            out['latent_xyzs_loss'] = torch.tensor([0.0], device=device)
        else:
            raise ValueError(f'Unknown latent_xyzs_codec_mode: {self.latent_xyzs_codec_mode}')
        if self.compress_normal:
            out['normal_loss'] = get_normal_loss(output['gt_normal'], output['feat_hat'].tanh(), nearest_gt_idx_[0])
        else:
            out['normal_loss'] = torch.tensor([0.0], device=device)
        if 'likelihoods' in output:
            out.update(compute_rate_loss(output['likelihoods'], B, P))
        out['loss'] = sum(self.lmbda[k] * out[f'{k}_loss'] for k in self.lmbda.keys() if f'{k}_loss' in out)
        return out


def collect_likelihoods_list(likelihoods_list, num_pixels: 'int'):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0
    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)
                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp
                bpp_info_dict[f'bpp_loss.{label}'] += bpp.sum()
                bpp_info_dict[f'bpp_loss.{label}.{i}.{field}'] = bpp.sum()
            bpp_info_dict[f'bpp_loss.{label}.{i}'] = label_bpp.sum()
        bpp_info_dict[f'bpp_loss.{i}'] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_details: 'bool'=False, bitdepth: 'int'=8):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2 ** bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) for frame_likelihoods in likelihoods_list for likelihoods in frame_likelihoods)

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f'len(x)={len(x)} != len(target)={len(target)})')
        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError('number of channels mismatches while computing distortion')
        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)
        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)
        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) ->bool:
        return isinstance(x, torch.Tensor) and x.ndimension() == 4 or isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)

    @classmethod
    def _check_tensors_list(cls, lst):
        if not isinstance(lst, (tuple, list)) or len(lst) < 1 or any(not cls._check_tensor(x) for x in lst):
            raise ValueError('Expected a list of 4D torch.Tensor (or tuples of) as input')

    def forward(self, output, target):
        assert isinstance(target, type(output['x_hat']))
        assert len(output['x_hat']) == len(target)
        self._check_tensors_list(target)
        self._check_tensors_list(output['x_hat'])
        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output['x_hat'], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)
            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)
            if self.return_details:
                out[f'frame{i}.mse_loss'] = distortion
        out['mse_loss'] = torch.stack(distortions).mean()
        scaled_distortions = sum(scaled_distortions) / num_frames
        assert isinstance(output['likelihoods'], list)
        likelihoods_list = output.pop('likelihoods')
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)
        lambdas = torch.full_like(bpp_loss, self.lmbda)
        bpp_loss = bpp_loss.mean()
        out['loss'] = (lambdas * scaled_distortions).mean() + bpp_loss
        out['distortion'] = scaled_distortions.mean()
        out['bpp_loss'] = bpp_loss
        return out


SCALES_LEVELS = 64


SCALES_MAX = 256


SCALES_MIN = 0.11


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


KEY_MAP = {'_bias': 'biases', '_matrix': 'matrices', '_factor': 'factors'}


def remap_old_keys(module_name, state_dict):

    def remap_subkey(s: 'str') ->str:
        for k, v in KEY_MAP.items():
            if s.startswith(k):
                return '.'.join((v, s.split(k)[1]))
        return s
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(module_name):
            k = '.'.join((module_name, remap_subkey(k.split(f'{module_name}.')[1])))
        new_state_dict[k] = v
    return new_state_dict


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(module, buffer_name, state_dict_key, state_dict, policy='resize_if_empty', dtype=torch.int):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)
    if policy in ('resize_if_empty', 'resize'):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == 'resize' or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == 'register':
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(module, module_name, buffer_names, state_dict, policy='resize_if_empty', dtype=torch.int):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        _update_registered_buffer(module, buffer_name, f'{module_name}.{buffer_name}', state_dict, policy, dtype)


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with any number of
    EntropyBottleneck or GaussianConditional modules.
    """

    def __init__(self, entropy_bottleneck_channels=None, init_weights=None):
        super().__init__()
        if entropy_bottleneck_channels is not None:
            warnings.warn('The entropy_bottleneck_channels parameter is deprecated. Create an entropy_bottleneck in your model directly instead:\n\nclass YourModel(CompressionModel):\n    def __init__(self):\n        super().__init__()\n        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)\n', DeprecationWarning, stacklevel=2)
            self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        if init_weights is not None:
            warnings.warn('The init_weights parameter was removed as it was never functional.', DeprecationWarning, stacklevel=2)

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue
            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(module, name, ['_quantized_cdf', '_offset', '_cdf_length'], state_dict)
                state_dict = remap_old_keys(name, state_dict)
            if isinstance(module, GaussianConditional):
                update_registered_buffers(module, name, ['_quantized_cdf', '_offset', '_cdf_length', 'scale_table'], state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False, update_quantiles: 'bool'=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force, update_quantiles=update_quantiles)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) ->Tensor:
        """Returns the total auxiliary loss over all ``EntropyBottleneck``\\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)


class SimpleVAECompressionModel(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """
    g_a: 'nn.Module'
    g_s: 'nn.Module'
    latent_codec: 'LatentCodec'

    def __getitem__(self, key: 'str') ->LatentCodec:
        return self.latent_codec[key]

    def forward(self, x):
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out['y_hat']
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': y_out['likelihoods']}

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out['y_hat']
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)


class FactorizedPrior(CompressionModel):
    """Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N), conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, 3))
        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) ->int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {'strings': [y_strings], 'shape': y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class FactorizedPriorReLU(FactorizedPrior):
    """Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(conv(3, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), deconv(N, 3))


class ScaleHyperprior(CompressionModel):
    """Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N), conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, 3))
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), conv(N, M, stride=1, kernel_size=3), nn.ReLU(inplace=True))
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) ->int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class MeanScaleHyperprior(ScaleHyperprior):
    """Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.LeakyReLU(inplace=True), conv(N, N), nn.LeakyReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, M), nn.LeakyReLU(inplace=True), deconv(M, M * 3 // 2), nn.LeakyReLU(inplace=True), conv(M * 3 // 2, M * 2, stride=1, kernel_size=3))

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    """Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                   params ▼
                         └─┬─┘                                          │
                     y_hat ▼                  ┌─────┐                   │
                           ├──────────►───────┤  CP ├────────►──────────┤
                           │                  └─────┘                   │
                           ▼                                            ▼
                           │                                            │
                           ·                  ┌─────┐                   │
                        GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                           ·     scales_hat   └─────┘
                           │      means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional
        EP = Entropy parameters network
        CP = Context prediction (masked convolution)

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(conv(3, N, kernel_size=5, stride=2), GDN(N), conv(N, N, kernel_size=5, stride=2), GDN(N), conv(N, N, kernel_size=5, stride=2), GDN(N), conv(N, M, kernel_size=5, stride=2))
        self.g_s = nn.Sequential(deconv(M, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, 3, kernel_size=5, stride=2))
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.LeakyReLU(inplace=True), conv(N, N, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), conv(N, N, stride=2, kernel_size=5))
        self.h_s = nn.Sequential(deconv(N, M, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), deconv(M, M * 3 // 2, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), conv(M * 3 // 2, M * 2, stride=1, kernel_size=3))
        self.entropy_parameters = nn.Sequential(nn.Conv2d(M * 12 // 3, M * 10 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(M * 10 // 3, M * 8 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(M * 8 // 3, M * 6 // 3, 1))
        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) ->int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, 'noise' if self.training else 'dequantize')
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).', stacklevel=2)
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding)
            y_strings.append(string)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, bias=self.context_prediction.bias)
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, 'symbols', means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat
                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).', stacklevel=2)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = torch.zeros((z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding), device=z_hat.device)
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(y_string, y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding)
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, self.context_prediction.weight, bias=self.context_prediction.bias)
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)
                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp:hp + 1, wp:wp + 1] = rv


def _select_xyzs_and_feats_single(candidate_xyzs, candidate_feats, upsample_num):
    batch_size, _, points_num, max_upsample_num = candidate_xyzs.shape
    assert batch_size == 1
    upsample_num = upsample_num.round().long().squeeze(0).view(-1, 1)
    mask = torch.arange(max_upsample_num).view(1, -1).repeat(points_num, 1)
    mask = (mask < upsample_num).view(-1)
    [idx] = mask.nonzero(as_tuple=True)
    idx = idx.unsqueeze(0)
    xyzs = index_points(candidate_xyzs.view(*candidate_xyzs.shape[:2], -1), idx)
    feats = index_points(candidate_feats.view(*candidate_feats.shape[:2], -1), idx)
    return xyzs, feats


def cycle_after(x, end):
    """Cycle tensor after a given index.

    Example:

        .. code-block:: python

            >>> x = torch.tensor([[5, 0, 7, 6, 2], [3, 1, 4, 8, 9]])
            >>> end = torch.tensor([2, 3])
            >>> idx, _, _ = cycle_after(x, end)
            >>> idx
            tensor([[5, 0, 5, 0, 5], [3, 1, 4, 3, 1]])
    """
    *dims, n = x.shape
    assert end.shape == tuple(dims)
    idx = torch.arange(n, device=x.device).repeat(*dims, 1)
    mask = idx >= end.unsqueeze(-1)
    idx[mask] %= end.unsqueeze(-1).repeat([1] * len(dims) + [n])[mask]
    x = x.gather(-1, idx)
    return x, idx, ~mask


def randperm(shape, device=None, dim=-1):
    """Random permutation, like `torch.randperm`, but with a shape."""
    if dim != -1:
        raise NotImplementedError
    idx = torch.rand(shape, device=device).argsort(dim=dim)
    return idx


def resample_points(xyzs, feats, num_points):
    """Resample points to a target number.

    Args:
        xyzs: (b, 3, n)
        feats: (b, c, n)

    Returns:
        new_xyzs: (b, 3, num_points)
        new_feats: (b, c, num_points)
    """
    b, _, n = xyzs.shape
    device = xyzs.device
    assert b == 1
    if n == num_points:
        return xyzs, feats
    if n > num_points:
        xyzs_tr = xyzs.permute(0, 2, 1).contiguous()
        idx = pointops.furthestsampling(xyzs_tr, num_points).long()
    elif n < num_points:
        idx_repeated = torch.arange(n, device=device).repeat(num_points // n)
        idx_random = torch.multinomial(torch.ones(n, device=device), num_points % n) if num_points % n > 0 else torch.arange(0, device=device)
        idx = torch.cat((idx_repeated, idx_random))
        perm = torch.randperm(len(idx), device=device)
        idx = idx[perm]
    idx = idx.reshape(1, -1)
    new_xyzs = index_points(xyzs, idx)
    new_feats = index_points(feats, idx)
    return new_xyzs, new_feats


def select_xyzs_and_feats(candidate_xyzs, candidate_feats, upsample_num, upsample_rate=None, method='batch_loop'):
    """Selects subset of points to match predicted local point cloud densities.

    Args:
        candidate_xyzs: (b, 3, n, s)
        candidate_feats: (b, c, n, s)
        upsample_num: (b, n)
        upsample_rate: Maximum number of points per group.
        method: "batch_loop" or "batch_noloop".

    Returns:
        xyzs: (b, 3, m)
        feats: (b, c, m)
    """
    device = candidate_xyzs.device
    b, c, n, s = candidate_feats.shape
    if b == 1 and upsample_rate is None:
        xyzs, feats = _select_xyzs_and_feats_single(candidate_xyzs, candidate_feats, upsample_num)
        return xyzs, feats
    if method == 'batch_loop':
        max_points = ceil(n * upsample_rate)
        xyzs = []
        feats = []
        for i in range(b):
            xyzs_i, feats_i = _select_xyzs_and_feats_single(candidate_xyzs[[i]], candidate_feats[[i]], upsample_num[[i]])
            xyzs_i, feats_i = resample_points(xyzs_i, feats_i, max_points)
            xyzs.append(xyzs_i)
            feats.append(feats_i)
        xyzs = torch.cat(xyzs)
        feats = torch.cat(feats)
    if method == 'batch_noloop':
        upsample_num = upsample_num.round().long().clip(1, s)
        idx = torch.arange(s, device=device).repeat(b, n, 1)
        mask = idx < upsample_num.unsqueeze(-1)
        idx = randperm((b, n, s), device=device, dim=-1)
        idx += torch.arange(n, device=device).view(1, -1, 1) * s
        idx = idx.view(b, -1)
        mask = mask.view(b, n * s)
        perm = mask.argsort(dim=-1, descending=True)
        idx = idx.gather(-1, perm)
        max_points = mask.sum(dim=-1).max().item() if upsample_rate is None else ceil(n * upsample_rate)
        idx = idx[..., :max_points]
        idx, _, _ = cycle_after(idx, mask.sum(dim=-1))
        idx = idx.gather(-1, randperm(idx.shape, device=device, dim=-1))
        xyzs = index_points(candidate_xyzs.view(b, 3, -1), idx)
        feats = index_points(candidate_feats.view(b, c, -1), idx)
    return xyzs, feats


class Decoder(nn.Module):

    def __init__(self, downsample_rate, candidate_upsample_rate, dim, hidden_dim, k, sub_point_conv_mode, compress_normal):
        super().__init__()
        self.k = k
        self.compress_normal = compress_normal
        self.num_layers = len(downsample_rate)
        self.downsample_rate = downsample_rate
        self.upsample_layers = nn.ModuleList([UpsampleLayer(dim, hidden_dim, k, sub_point_conv_mode, candidate_upsample_rate[i]) for i in range(self.num_layers)])
        self.upsample_num_layers = nn.ModuleList([UpsampleNumLayer(dim, hidden_dim, candidate_upsample_rate[i]) for i in range(self.num_layers)])
        self.refine_layers = nn.ModuleList([RefineLayer(dim, hidden_dim, k, sub_point_conv_mode, compress_normal and i == self.num_layers - 1) for i in range(self.num_layers)])

    def forward(self, xyzs, feats):
        latent_xyzs = xyzs.clone()
        xyzs_hat_ = []
        unums_hat_ = []
        for i, (upsample_nn, upsample_num_nn, refine_nn) in enumerate(zip(self.upsample_layers, self.upsample_num_layers, self.refine_layers)):
            candidate_xyzs, candidate_feats = upsample_nn(xyzs, feats)
            upsample_num = upsample_num_nn(feats)
            xyzs, feats = select_xyzs_and_feats(candidate_xyzs, candidate_feats, upsample_num, upsample_rate=1 / self.downsample_rate[self.num_layers - i - 1])
            xyzs, feats = refine_nn(xyzs, feats)
            xyzs_hat_.append(xyzs)
            unums_hat_.append(upsample_num)
        mdis_hat_ = self.get_pred_mdis([latent_xyzs, *xyzs_hat_], unums_hat_)
        return xyzs_hat_, unums_hat_, mdis_hat_, feats

    def get_pred_mdis(self, xyzs_hat_, unums_hat_):
        mdis_hat_ = []
        for prev_xyzs, curr_xyzs, curr_unums in zip(xyzs_hat_[:-1], xyzs_hat_[1:], unums_hat_):
            distance, _, _, _ = nearby_distance_sum(prev_xyzs, curr_xyzs, self.k)
            curr_mdis = distance / curr_unums
            mdis_hat_.append(curr_mdis)
        return mdis_hat_


class Encoder(nn.Module):

    def __init__(self, downsample_rate, dim, hidden_dim, k, ngroups):
        super().__init__()
        downsample_layers = [DownsampleLayer(downsample_rate[i], dim, hidden_dim, k, ngroups) for i in range(len(downsample_rate))]
        self.downsample_layers = nn.ModuleList(downsample_layers)

    def forward(self, xyzs, feats):
        gt_xyzs_ = []
        gt_dnums_ = []
        gt_mdis_ = []
        for downsample_layer in self.downsample_layers:
            gt_xyzs_.append(xyzs)
            xyzs, feats, downsample_num, mean_distance = downsample_layer(xyzs, feats)
            gt_dnums_.append(downsample_num)
            gt_mdis_.append(mean_distance)
        latent_xyzs = xyzs
        latent_feats = feats
        return gt_xyzs_, gt_dnums_, gt_mdis_, latent_xyzs, latent_feats


class XyzsLatentCodec(nn.Module):

    def __init__(self, dim, hidden_dim, k, ngroups, mode='learned', conv_mode='mlp'):
        super().__init__()
        self.mode = mode
        if mode == 'learned':
            if conv_mode == 'edge_conv':
                self.analysis = EdgeConv(3, dim, hidden_dim, k)
                self.synthesis = EdgeConv(dim, 3, hidden_dim, k)
            elif conv_mode == 'mlp':
                self.analysis = nn.Sequential(nn.Conv1d(3, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv1d(hidden_dim, dim, 1))
                self.synthesis = nn.Sequential(nn.Conv1d(dim, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(inplace=True), nn.Conv1d(hidden_dim, 3, 1))
            else:
                raise ValueError(f'Unknown conv_mode: {conv_mode}')
            self.entropy_bottleneck = EntropyBottleneck(dim)
        else:
            self.placeholder = nn.Parameter(torch.empty(0))

    def forward(self, latent_xyzs):
        if self.mode == 'learned':
            z = self.analysis(latent_xyzs)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == 'float16':
            z_likelihoods = latent_xyzs.new_full(latent_xyzs.shape, 2 ** -16)
            latent_xyzs_hat = latent_xyzs.float()
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return {'likelihoods': {'y': z_likelihoods}, 'y_hat': latent_xyzs_hat}

    def compress(self, latent_xyzs):
        if self.mode == 'learned':
            z = self.analysis(latent_xyzs)
            shape = z.shape[2:]
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == 'float16':
            z = latent_xyzs
            shape = z.shape[2:]
            z_hat = latent_xyzs
            z_strings = [np.ascontiguousarray(x, dtype='>f2').tobytes() for x in z_hat.cpu().numpy()]
            latent_xyzs_hat = z_hat.float()
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return {'strings': [z_strings], 'shape': shape, 'y_hat': latent_xyzs_hat}

    def decompress(self, strings, shape):
        [z_strings] = strings
        if self.mode == 'learned':
            z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
            latent_xyzs_hat = self.synthesis(z_hat)
        elif self.mode == 'float16':
            z_hat = [np.frombuffer(s, dtype='>f2').reshape(shape) for s in z_strings]
            z_hat = torch.from_numpy(np.stack(z_hat))
            latent_xyzs_hat = z_hat.float()
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return {'y_hat': latent_xyzs_hat}


class DensityPreservingReconstructionPccModel(CompressionModel):
    """Density-preserving deep point cloud compression.

    Model introduced by [He2022pcc]_.

    References:

        .. [He2022pcc] `"Density-preserving Deep Point Cloud Compression"
            <https://arxiv.org/abs/2204.12684>`_, by Yun He, Xinlin Ren,
            Danhang Tang, Yinda Zhang, Xiangyang Xue, and Yanwei Fu,
            CVPR 2022.
    """

    def __init__(self, downsample_rate=(1 / 3, 1 / 3, 1 / 3), candidate_upsample_rate=(8, 8, 8), in_dim=3, feat_dim=8, hidden_dim=64, k=16, ngroups=1, sub_point_conv_mode='mlp', compress_normal=False, latent_xyzs_codec=None, **kwargs):
        super().__init__()
        self.compress_normal = compress_normal
        self.pre_conv = nn.Sequential(nn.Conv1d(in_dim, hidden_dim, 1), nn.GroupNorm(ngroups, hidden_dim), nn.ReLU(), nn.Conv1d(hidden_dim, feat_dim, 1))
        self.encoder = Encoder(downsample_rate, feat_dim, hidden_dim, k, ngroups)
        self.decoder = Decoder(downsample_rate, candidate_upsample_rate, feat_dim, hidden_dim, k, sub_point_conv_mode, compress_normal)
        self.latent_codec = nn.ModuleDict({'feat': EntropyBottleneckLatentCodec(channels=feat_dim), 'xyz': XyzsLatentCodec(feat_dim, hidden_dim, k, ngroups, **latent_xyzs_codec or {})})

    def _prepare_input(self, input):
        input_data = [input['pos']]
        if self.compress_normal:
            input_data.append(input['normal'])
        input_data = torch.cat(input_data, dim=1).permute(0, 2, 1).contiguous()
        xyzs = input_data[:, :3].contiguous()
        gt_normals = input_data[:, 3:3 + 3 * self.compress_normal].contiguous()
        feats = input_data
        return xyzs, gt_normals, feats

    def forward(self, input):
        xyzs, gt_normals, feats = self._prepare_input(input)
        feats = self.pre_conv(feats)
        gt_xyzs_, gt_dnums_, gt_mdis_, latent_xyzs, latent_feats = self.encoder(xyzs, feats)
        gt_latent_xyzs = latent_xyzs
        latent_feats = latent_feats.unsqueeze(-1)
        latent_feats_out = self.latent_codec['feat'](latent_feats)
        latent_feats_hat = latent_feats_out['y_hat'].squeeze(-1)
        latent_xyzs_out = self.latent_codec['xyz'](latent_xyzs)
        latent_xyzs_hat = latent_xyzs_out['y_hat']
        xyzs_hat_, unums_hat_, mdis_hat_, feats_hat = self.decoder(latent_xyzs_hat, latent_feats_hat)
        xyzs_hat = xyzs_hat_[-1].permute(0, 2, 1).contiguous()
        return {'x_hat': xyzs_hat, 'xyz_hat_': xyzs_hat_, 'latent_xyz_hat': latent_xyzs_hat, 'feat_hat': feats_hat, 'upsample_num_hat_': unums_hat_, 'mean_distance_hat_': mdis_hat_, 'gt_xyz_': gt_xyzs_, 'gt_latent_xyz': gt_latent_xyzs, 'gt_normal': gt_normals, 'gt_downsample_num_': gt_dnums_, 'gt_mean_distance_': gt_mdis_, 'likelihoods': {'latent_feat': latent_feats_out['likelihoods']['y'], 'latent_xyz': latent_xyzs_out['likelihoods']['y']}}

    def compress(self, input):
        xyzs, _, feats = self._prepare_input(input)
        feats = self.pre_conv(feats)
        _, _, _, latent_xyzs, latent_feats = self.encoder(xyzs, feats)
        latent_feats = latent_feats.unsqueeze(-1)
        latent_feats_out = self.latent_codec['feat'].compress(latent_feats)
        latent_xyzs = latent_xyzs
        latent_xyzs_out = self.latent_codec['xyz'].compress(latent_xyzs)
        return {'strings': [latent_feats_out['strings'], latent_xyzs_out['strings']], 'shape': [latent_feats_out['shape'], latent_xyzs_out['shape']]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        latent_feats_out = self.latent_codec['feat'].decompress(strings[0], shape[0])
        latent_feats_hat = latent_feats_out['y_hat'].squeeze(-1)
        latent_xyzs_out = self.latent_codec['xyz'].decompress(strings[1], shape[1])
        latent_xyzs_hat = latent_xyzs_out['y_hat']
        xyzs_hat_, _, _, feats_hat = self.decoder(latent_xyzs_hat, latent_feats_hat)
        xyzs_hat = xyzs_hat_[-1].permute(0, 2, 1).contiguous()
        return {'x_hat': xyzs_hat, 'feat_hat': feats_hat}


GAIN = 10.0


class PointNet2SsgReconstructionPccModel(CompressionModel):
    """PointNet++-based PCC reconstruction model.

    Model based on PointNet++ [Qi2017PointNetPlusPlus]_, and modified
    for compression by [Ulhaq2024]_.
    Uses single-scale grouping (SSG) for point set abstraction.

    References:

        .. [Qi2017PointNetPlusPlus] `"PointNet++: Deep Hierarchical
            Feature Learning on Point Sets in a Metric Space"
            <https://arxiv.org/abs/1706.02413>`_, by Charles R. Qi,
            Li Yi, Hao Su, and Leonidas J. Guibas, NIPS 2017.

        .. [Ulhaq2024] `"Scalable Human-Machine Point Cloud Compression"
            <TODO>`_,
            by Mateen Ulhaq and Ivan V. Bajić, PCS 2024.
    """

    def __init__(self, num_points=1024, num_classes=40, D=(0, 128, 192, 256), P=(1024, 256, 64, 1), S=(None, 4, 4, 64), R=(None, 0.2, 0.4, None), E=(3, 64, 32, 16, 0), M=(0, 0, 64, 64), normal_channel=False):
        """
        Args:
            num_points: Number of input points. [unused]
            num_classes: Number of object classes. [unused]
            D: Number of input feature channels.
            P: Number of output points.
            S: Number of samples per centroid.
            R: Radius of the ball to query points within.
            E: Number of output feature channels after each upsample.
            M: Number of latent channels for the bottleneck.
            normal_channel: Whether the input includes normals.
        """
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.D = D
        self.P = P
        self.S = S
        self.R = R
        self.E = E
        self.M = M
        self.normal_channel = bool(normal_channel)
        assert P[0] == P[1] * S[1]
        assert P[1] == P[2] * S[2]
        assert P[2] == P[3] * S[3]
        self.levels = 4
        self.down = nn.ModuleDict({'_1': PointNetSetAbstraction(npoint=P[1], radius=R[1], nsample=S[1], in_channel=D[0] + 3, mlp=[D[1] // 2, D[1] // 2, D[1]], group_all=False), '_2': PointNetSetAbstraction(npoint=P[2], radius=R[2], nsample=S[2], in_channel=D[1] + 3, mlp=[D[1], D[1], D[2]], group_all=False), '_3': PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=D[2] + 3, mlp=[D[2], D[3], D[3]], group_all=True)})
        i_final = self.levels - 1
        groups_h_final = 1 if D[i_final] * M[i_final] <= 2 ** 16 else 4
        self.h_a = nn.ModuleDict({**{f'_{i}': nn.Sequential(Reshape((D[i] + 3, P[i + 1] * S[i + 1])), nn.Conv1d(D[i] + 3, M[i], 1), Gain((M[i], 1), factor=GAIN)) for i in range(self.levels - 1) if M[i] > 0}, f'_{i_final}': nn.Sequential(Reshape((D[i_final], 1)), nn.Conv1d(D[i_final], M[i_final], 1, groups=groups_h_final), Interleave(groups=groups_h_final), Gain((M[i_final], 1), factor=GAIN))})
        self.h_s = nn.ModuleDict({**{f'_{i}': nn.Sequential(Gain((M[i], 1), factor=1 / GAIN), nn.Conv1d(M[i], D[i] + 3, 1)) for i in range(self.levels - 1) if M[i] > 0}, f'_{i_final}': nn.Sequential(Gain((M[i_final], 1), factor=1 / GAIN), nn.Conv1d(M[i_final], D[i_final], 1, groups=groups_h_final), Interleave(groups=groups_h_final))})
        self.up = nn.ModuleDict({'_0': nn.Sequential(nn.Conv1d(E[1] + D[0] + 3 * bool(M[0]), E[1], 1), nn.ReLU(inplace=True), nn.Conv1d(E[1], E[0], 1), Reshape((E[0], P[0])), Transpose(-2, -1)), '_1': UpsampleBlock(D, E, M, P, S, i=1, extra_in_ch=3, groups=(1, 4)), '_2': UpsampleBlock(D, E, M, P, S, i=2, extra_in_ch=3, groups=(1, 4)), '_3': UpsampleBlock(D, E, M, P, S, i=3, extra_in_ch=0, groups=(1, 4))})
        self.latent_codec = nn.ModuleDict({f'_{i}': EntropyBottleneckLatentCodec(channels=M[i], tail_mass=0.0001) for i in range(self.levels) if M[i] > 0})

    def forward(self, input):
        xyz, norm = self._get_inputs(input)
        y_out_, u_, uu_ = self._compress(xyz, norm, mode='forward')
        x_hat, y_hat_, v_ = self._decompress(y_out_, mode='forward')
        return {'x_hat': x_hat, 'likelihoods': {f'y_{i}': y_out_[i]['likelihoods']['y'] for i in range(self.levels) if 'likelihoods' in y_out_[i]}, 'debug_outputs': {**{f'u_{i}': v for i, v in u_.items() if v is not None}, **{f'uu_{i}': v for i, v in uu_.items()}, **{f'y_hat_{i}': v for i, v in y_hat_.items()}, **{f'v_{i}': v for i, v in v_.items() if v.numel() > 0}}}

    def compress(self, input):
        xyz, norm = self._get_inputs(input)
        y_out_, _, _ = self._compress(xyz, norm, mode='compress')
        return {'strings': [ss for level in range(self.levels) for ss in y_out_[level]['strings']], 'shape': {f'y_{i}': y_out_[i]['shape'] for i in range(self.levels)}}

    def decompress(self, strings, shape):
        y_inputs_ = {i: {'strings': [strings[i]], 'shape': shape[f'y_{i}']} for i in range(self.levels)}
        x_hat, _, _ = self._decompress(y_inputs_, mode='decompress')
        return {'x_hat': x_hat}

    def _get_inputs(self, input):
        points = input['pos'].transpose(-2, -1)
        if self.normal_channel:
            xyz = points[:, :3, :]
            norm = points[:, 3:, :]
        else:
            xyz = points
            norm = None
        return xyz, norm

    def _compress(self, xyz, norm, *, mode):
        lc_func = {'forward': lambda lc: lc, 'compress': lambda lc: lc.compress}[mode]
        B, _, _ = xyz.shape
        xyz_ = {(0): xyz}
        u_ = {(0): norm}
        uu_ = {}
        y_ = {}
        y_out_ = {}
        for i in range(1, self.levels):
            down_out_i = self.down[f'_{i}'](xyz_[i - 1], u_[i - 1])
            xyz_[i] = down_out_i['new_xyz']
            u_[i] = down_out_i['new_features']
            uu_[i - 1] = down_out_i['grouped_features']
        uu_[self.levels - 1] = u_[self.levels - 1][:, :, None, :]
        for i in reversed(range(0, self.levels)):
            if self.M[i] == 0:
                y_out_[i] = {'strings': [[b''] * B], 'shape': ()}
                continue
            y_[i] = self.h_a[f'_{i}'](uu_[i])
            y_out_[i] = lc_func(self.latent_codec[f'_{i}'])(y_[i][..., None])
        return y_out_, u_, uu_

    def _decompress(self, y_inputs_, *, mode):
        y_hat_ = {}
        y_out_ = {}
        uu_hat_ = {}
        v_ = {}
        for i in reversed(range(0, self.levels)):
            if self.M[i] == 0:
                continue
            if mode == 'forward':
                y_out_[i] = y_inputs_[i]
            elif mode == 'decompress':
                y_out_[i] = self.latent_codec[f'_{i}'].decompress(y_inputs_[i]['strings'], shape=y_inputs_[i]['shape'])
            y_hat_[i] = y_out_[i]['y_hat'].squeeze(-1)
            uu_hat_[i] = self.h_s[f'_{i}'](y_hat_[i])
        B, _, *tail = uu_hat_[self.levels - 1].shape
        v_[self.levels] = uu_hat_[self.levels - 1].new_zeros((B, 0, *tail))
        for i in reversed(range(0, self.levels)):
            v_[i] = self.up[f'_{i}'](v_[i + 1] if self.M[i] == 0 else torch.cat([v_[i + 1], uu_hat_[i]], dim=1))
        x_hat = v_[0]
        return x_hat, y_hat_, v_


class Cheng2020AnchorCheckerboard(SimpleVAECompressionModel):
    """Cheng2020 anchor model with checkerboard context model.

    Base transform model from [Cheng2020]. Context model from [He2021].

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(**kwargs)
        self.g_a = nn.Sequential(ResidualBlockWithStride(3, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), conv3x3(N, N, stride=2))
        self.g_s = nn.Sequential(ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), subpel_conv3x3(N, 3, 2))
        h_a = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2))
        h_s = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), subpel_conv3x3(N, N, 2), nn.LeakyReLU(inplace=True), conv3x3(N, N * 3 // 2), nn.LeakyReLU(inplace=True), subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2), nn.LeakyReLU(inplace=True), conv3x3(N * 3 // 2, N * 2))
        self.latent_codec = HyperpriorLatentCodec(latent_codec={'y': CheckerboardLatentCodec(latent_codec={'y': GaussianConditionalLatentCodec(quantizer='ste')}, entropy_parameters=nn.Sequential(nn.Conv2d(N * 12 // 3, N * 10 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(N * 10 // 3, N * 8 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(N * 8 // 3, N * 6 // 3, 1)), context_prediction=CheckerboardMaskedConv2d(N, 2 * N, kernel_size=5, stride=1, padding=2)), 'hyper': HyperLatentCodec(entropy_bottleneck=EntropyBottleneck(N), h_a=h_a, h_s=h_s, quantizer='ste')})

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.conv1.weight'].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: 'int', out_ch: 'int'):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: 'Tensor') ->Tensor:
        identity = self.skip(x)
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity


def ramp(a, b, steps=None, method='linear', **kwargs):
    if method == 'linear':
        return torch.linspace(a, b, steps, **kwargs)
    if method == 'log':
        return torch.logspace(math.log10(a), math.log10(b), steps, **kwargs)
    raise ValueError(f'Unknown ramp method: {method}')


def sequential_channel_ramp(in_ch: 'int', out_ch: 'int', *, min_ch: int=0, num_layers: int=3, interp: str='linear', make_layer=None, make_act=None, skip_last_act: bool=True, **layer_kwargs) ->nn.Module:
    """Interleave layers of gradually ramping channels with nonlinearities."""
    channels = ramp(in_ch, out_ch, num_layers + 1, method=interp).floor().int()
    channels[1:-1] = channels[1:-1].clip(min=min_ch)
    channels = channels.tolist()
    layers = [module for ch_in, ch_out in zip(channels[:-1], channels[1:]) for module in [make_layer(ch_in, ch_out, **layer_kwargs), make_act()]]
    if skip_last_act:
        layers = layers[:-1]
    return nn.Sequential(*layers)


class Elic2022Official(SimpleVAECompressionModel):
    """ELIC 2022; uneven channel groups with checkerboard spatial context.

    Context model from [He2022].
    Based on modified attention model architecture from [Cheng2020].

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        self.groups = list(groups)
        assert sum(self.groups) == M
        self.g_a = nn.Sequential(conv(3, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), conv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), AttentionBlock(N), conv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), conv(N, M, kernel_size=5, stride=2), AttentionBlock(M))
        self.g_s = nn.Sequential(AttentionBlock(M), deconv(M, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, N, kernel_size=5, stride=2), AttentionBlock(N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, 3, kernel_size=5, stride=2))
        h_a = nn.Sequential(conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True), conv(N, N, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(N, N, kernel_size=5, stride=2))
        h_s = nn.Sequential(deconv(N, N, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(N, N * 3 // 2, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1))
        channel_context = {f'y{k}': sequential_channel_ramp(sum(self.groups[:k]), self.groups[k] * 2, min_ch=N, num_layers=3, make_layer=nn.Conv2d, make_act=lambda : nn.ReLU(inplace=True), kernel_size=5, stride=1, padding=2) for k in range(1, len(self.groups))}
        spatial_context = [CheckerboardMaskedConv2d(self.groups[k], self.groups[k] * 2, kernel_size=5, stride=1, padding=2) for k in range(len(self.groups))]
        param_aggregation = [sequential_channel_ramp(self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2, self.groups[k] * 2, min_ch=N * 2, num_layers=3, make_layer=nn.Conv2d, make_act=lambda : nn.ReLU(inplace=True), kernel_size=1, stride=1, padding=0) for k in range(len(self.groups))]
        scctx_latent_codec = {f'y{k}': CheckerboardLatentCodec(latent_codec={'y': GaussianConditionalLatentCodec(quantizer='ste')}, context_prediction=spatial_context[k], entropy_parameters=param_aggregation[k]) for k in range(len(self.groups))}
        self.latent_codec = HyperpriorLatentCodec(latent_codec={'y': ChannelGroupsLatentCodec(groups=self.groups, channel_context=channel_context, latent_codec=scctx_latent_codec), 'hyper': HyperLatentCodec(entropy_bottleneck=EntropyBottleneck(N), h_a=h_a, h_s=h_s, quantizer='ste')})

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Elic2022Chandelier(SimpleVAECompressionModel):
    """ELIC 2022; simplified context model using only first and most recent groups.

    Context model from [He2022], with simplifications and parameters
    from the [Chandelier2023] implementation.
    Based on modified attention model architecture from [Cheng2020].

    .. note::

        This implementation contains some differences compared to the
        original [He2022] paper. For instance, the implemented context
        model only uses the first and the most recently decoded channel
        groups to predict the current channel group. In contrast, the
        original paper uses all previously decoded channel groups.
        Also, the last layer of h_s is now a conv rather than a deconv.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, groups=None, **kwargs):
        super().__init__(**kwargs)
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        self.groups = list(groups)
        assert sum(self.groups) == M
        self.g_a = nn.Sequential(conv(3, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), conv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), AttentionBlock(N), conv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), conv(N, M, kernel_size=5, stride=2), AttentionBlock(M))
        self.g_s = nn.Sequential(AttentionBlock(M), deconv(M, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, N, kernel_size=5, stride=2), AttentionBlock(N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, N, kernel_size=5, stride=2), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), ResidualBottleneckBlock(N, N), deconv(N, 3, kernel_size=5, stride=2))
        h_a = nn.Sequential(conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True), conv(N, N, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(N, N, kernel_size=5, stride=2))
        h_s = nn.Sequential(deconv(N, N, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(N, N * 3 // 2, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(N * 3 // 2, M * 2, kernel_size=3, stride=1))
        channel_context = {f'y{k}': nn.Sequential(conv(self.groups[0] + (k > 1) * self.groups[k - 1], 224, kernel_size=5, stride=1), nn.ReLU(inplace=True), conv(224, 128, kernel_size=5, stride=1), nn.ReLU(inplace=True), conv(128, self.groups[k] * 2, kernel_size=5, stride=1)) for k in range(1, len(self.groups))}
        spatial_context = [CheckerboardMaskedConv2d(self.groups[k], self.groups[k] * 2, kernel_size=5, stride=1, padding=2) for k in range(len(self.groups))]
        param_aggregation = [nn.Sequential(conv1x1(self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M * 2, M * 2), nn.ReLU(inplace=True), conv1x1(M * 2, 512), nn.ReLU(inplace=True), conv1x1(512, self.groups[k] * 2)) for k in range(len(self.groups))]
        scctx_latent_codec = {f'y{k}': CheckerboardLatentCodec(latent_codec={'y': GaussianConditionalLatentCodec(quantizer='ste', chunks=('means', 'scales'))}, context_prediction=spatial_context[k], entropy_parameters=param_aggregation[k]) for k in range(len(self.groups))}
        self.latent_codec = HyperpriorLatentCodec(latent_codec={'y': ChannelGroupsLatentCodec(groups=self.groups, channel_context=channel_context, latent_codec=scctx_latent_codec), 'hyper': HyperLatentCodec(entropy_bottleneck=EntropyBottleneck(N), h_a=h_a, h_s=h_s, quantizer='ste')})
        self._monkey_patch()

    def _monkey_patch(self):
        """Monkey-patch to use only first group and most recent group."""

        def merge_y(self: 'ChannelGroupsLatentCodec', *args):
            if len(args) == 0:
                return Tensor()
            if len(args) == 1:
                return args[0]
            if len(args) < len(self.groups):
                return torch.cat([args[0], args[-1]], dim=1)
            return torch.cat(args, dim=1)
        chan_groups_latent_codec = self.latent_codec['y']
        obj = chan_groups_latent_codec
        obj.merge_y = types.MethodType(merge_y, obj)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class ScaleHyperpriorVbr(ScaleHyperprior):
    """Variable bitrate (vbr) version of bmshj2018-hyperprior (see compressai/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.
    """

    def __init__(self, N, M, vr_entbttlnck=False, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]
        self.levels = len(self.lmbda)
        self.Gain = torch.nn.Parameter(torch.tensor([0.1, 0.13944, 0.19293, 0.26874, 0.37268, 0.51801, 0.71957, 1.0]), requires_grad=True)
        Nds = 12
        self.QuantABCD = nn.Sequential(nn.Linear(1 + 1, Nds), nn.ReLU(), nn.Linear(Nds, Nds), nn.ReLU(), nn.Linear(Nds, 1))
        self.no_quantoffset = False
        self.vr_entbttlnck = vr_entbttlnck
        if self.vr_entbttlnck:
            self.entropy_bottleneck = EntropyBottleneckVbr(N)
            Ndsz = 10
            self.gayn2zqstep = nn.Sequential(nn.Linear(1, Ndsz), nn.ReLU(), nn.Linear(Ndsz, Ndsz), nn.ReLU(), nn.Linear(Ndsz, 1), nn.Softplus())
            self.lower_bound_zqstep = LowerBound(0.5)

    def _raise_stage_error(self, stage):
        raise ValueError(f'Invalid stage (stage={stage}) parameter for this model.')

    def _get_scale(self, stage, s, inputscale):
        s = max(0, min(s, len(self.Gain) - 1))
        if self.training:
            if stage > 1:
                scale = self.Gain[s].detach()
            else:
                scale = self.Gain[s].detach()
        elif inputscale == 0:
            scale = self.Gain[s].detach()
        else:
            scale = inputscale
        return scale

    def forward(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        """stage: 1 -> non-vbr (old) code path; vbr modules not used; operates like corresponding google model. use for initial training.
        2 -> vbr code path; vbr modules now used. use for post training with e.g. MOO.
        """
        scale = self._get_scale(stage, s, inputscale)
        rescale = 1.0 / scale.clone().detach()
        if stage == 1:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
            x_hat = self.g_s(y_hat)
        elif stage == 2:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            if not self.vr_entbttlnck:
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                z_offset = self.entropy_bottleneck._get_medians()
                z_tmp = z - z_offset
                z_hat = quantize_ste(z_tmp) + z_offset
            else:
                z_qstep = self.gayn2zqstep(1.0 / scale.clone().view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                z_hat, z_likelihoods = self.entropy_bottleneck(z, qs=z_qstep[0], training=None, ste=False)
            scales_hat = self.h_s(z_hat)
            if self.no_quantoffset:
                y_hat = quantize_ste(y * scale, 'ste') * rescale
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            else:
                y_ch_means = 0
                y_zm = y - y_ch_means
                y_zm_sc = y_zm * scale
                signs = torch.sign(y_zm_sc).detach()
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=4)
                q_offsets[q_abs < 0.0001] = 0
                y_hat = signs * (q_abs + q_offsets)
                y_hat = y_hat * rescale + y_ch_means
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        else:
            self._raise_stage_error(self, stage)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict, vr_entbttlnck=False):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M, vr_entbttlnck)
        if 'QuantOffset' in state_dict.keys():
            del state_dict['QuantOffset']
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False, scale=None):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        if isinstance(self.entropy_bottleneck, EntropyBottleneckVbr):
            sc = scale
            if sc is None:
                rv = self.entropy_bottleneck.update(force=force)
            else:
                z_qstep = self.gayn2zqstep(1.0 / sc.view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                rv = self.entropy_bottleneck.update_variable(force=force, qs=z_qstep)
        elif isinstance(self.entropy_bottleneck, EntropyBottleneck):
            rv = self.entropy_bottleneck.update(force=force)
        updated |= rv
        return updated

    def compress(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        if stage == 1 or stage == 2 and not self.vr_entbttlnck:
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        elif stage == 2:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_strings = self.entropy_bottleneck.compress(z, qs=z_qstep[0])
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], qs=z_qstep[0])
        else:
            self._raise_stage_error(self, stage)
        scales_hat = self.h_s(z_hat)
        if stage == 1:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes)
        elif stage == 2:
            indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
            y_strings = self.gaussian_conditional.compress(y * scale, indexes)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape, stage: 'int'=2, s: 'int'=1, inputscale=0):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        rescale = torch.tensor(1.0) / scale
        if stage == 1 or stage == 2 and not self.vr_entbttlnck:
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        elif stage == 2:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape, qs=z_qstep[0])
        else:
            self._raise_stage_error(self, stage)
        scales_hat = self.h_s(z_hat)
        if stage == 1:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        elif stage == 2:
            indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
            if self.no_quantoffset:
                y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * rescale
            else:
                q_val = self.gaussian_conditional.decompress(strings[0], indexes)
                q_abs, signs = q_val.abs(), torch.sign(q_val)
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=4)
                q_offsets[q_abs < 0.0001] = 0
                y_hat = signs * (q_abs + q_offsets)
                y_ch_means = 0
                y_hat = y_hat * rescale + y_ch_means
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class MeanScaleHyperpriorVbr(ScaleHyperpriorVbr, MeanScaleHyperprior):
    """Variable bitrate (vbr) version of mbt2018-mean (see compressai/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.
    """

    def __init__(self, N=192, M=320, vr_entbttlnck=False, **kwargs):
        super().__init__(N, M, vr_entbttlnck=vr_entbttlnck, **kwargs)

    def forward(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        """stage: 1 -> non-vbr (old) code path; vbr modules not used; operates like corresponding google model. use for initial training.
        2 -> vbr code path; vbr modules now used. use for post training with e.g. MOO.
        """
        scale = self._get_scale(stage, s, inputscale)
        rescale = 1.0 / scale.clone().detach()
        if stage == 1:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            x_hat = self.g_s(y_hat)
        elif stage == 2:
            y = self.g_a(x)
            z = self.h_a(y)
            if not self.vr_entbttlnck:
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                z_offset = self.entropy_bottleneck._get_medians()
                z_tmp = z - z_offset
                z_hat = quantize_ste(z_tmp) + z_offset
            else:
                z_qstep = self.gayn2zqstep(1.0 / scale.clone().view(1))
                z_qstep = self.lower_bound_zqstep(z_qstep)
                z_hat, z_likelihoods = self.entropy_bottleneck(z, qs=z_qstep[0], training=None, ste=False)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            if self.no_quantoffset:
                y_hat = self.quantizer.quantize((y - means_hat) * scale, 'ste') * rescale + means_hat
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
            else:
                y_zm = y - means_hat
                y_zm_sc = y_zm * scale
                signs = torch.sign(y_zm_sc).detach()
                q_abs = quantize_ste(torch.abs(y_zm_sc))
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=4)
                q_offsets[q_abs < 0.0001] = 0
                y_hat = signs * (q_abs + q_offsets)
                y_hat = y_hat * rescale + means_hat
                _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale, means=means_hat * scale)
            x_hat = self.g_s(y_hat)
        else:
            self._raise_stage_error(self, stage)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    def compress(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        y = self.g_a(x)
        z = self.h_a(y)
        if stage == 1 or stage == 2 and not self.vr_entbttlnck:
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        elif stage == 2:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_strings = self.entropy_bottleneck.compress(z, qs=z_qstep[0])
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], qs=z_qstep[0])
        else:
            self._raise_stage_error(self, stage)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        if stage == 1:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        elif stage == 2:
            indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
            y_strings = self.gaussian_conditional.compress(y * scale, indexes, means=means_hat * scale)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape, stage: 'int'=2, s: 'int'=1, inputscale=0):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        rescale = torch.tensor(1.0) / scale
        if stage == 1 or stage == 2 and not self.vr_entbttlnck:
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        elif stage == 2:
            z_qstep = self.gayn2zqstep(1.0 / scale.view(1))
            z_qstep = self.lower_bound_zqstep(z_qstep)
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape, qs=z_qstep[0])
        else:
            self._raise_stage_error(self, stage)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        if stage == 1:
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        elif stage == 2:
            indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
            if self.no_quantoffset:
                y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat * scale) * rescale
            else:
                q_val = self.gaussian_conditional.decompress(strings[0], indexes)
                q_abs, signs = q_val.abs(), torch.sign(q_val)
                q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=4)
                q_offsets[q_abs < 0.0001] = 0
                y_hat = signs * (q_abs + q_offsets)
                y_hat = y_hat * rescale + means_hat
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class JointAutoregressiveHierarchicalPriorsVbr(ScaleHyperpriorVbr, JointAutoregressiveHierarchicalPriors):
    """Variable bitrate (vbr) version of mbt2018 (see compressai/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.
    """

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, vr_entbttlnck=False, **kwargs)
        self.ste_recursive = True
        self.scl2ctx = True
        self.scale_to_context = nn.Linear(1, 2 * M)

    def forward(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        """stage: 1 -> non-vbr (old) code path; vbr modules not used; operates like corresponding google model. use for initial training.
        2 -> vbr code path; vbr modules now used. use for post training with e.g. MOO.
        """
        scale = self._get_scale(stage, s, inputscale)
        rescale = 1.0 / scale.clone().detach()
        if stage == 1:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)
            y_hat = self.gaussian_conditional.quantize(y, 'noise' if self.training else 'dequantize')
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            x_hat = self.g_s(y_hat)
        elif stage == 2:
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = quantize_ste(z_tmp) + z_offset
            params = self.h_s(z_hat)
            if self.ste_recursive:
                kernel_size = 5
                padding = (kernel_size - 1) // 2
                y_hat = F.pad(y, (padding, padding, padding, padding))
                y_hat, y_likelihoods = self._stequantization(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)
            else:
                raise ValueError('ste_recurseive=False is not supported.')
            x_hat = self.g_s(y_hat)
        else:
            self._raise_stage_error(self, stage)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    def _stequantization(self, y_hat, params, height, width, kernel_size, padding, scale, rescale):
        y_likelihoods = torch.zeros([y_hat.size(0), y_hat.size(1), height, width])
        if self.scl2ctx:
            ctx_scl = self.scale_to_context.forward(scale.view(1, 1)).view(1, -1, 1, 1)
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size].clone()
                ctx_p = F.conv2d(y_crop, masked_weight, bias=self.context_prediction.bias)
                if self.scl2ctx:
                    p = params[:, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p + ctx_scl), dim=1))
                else:
                    p = params[:, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_crop = y_crop[:, :, padding, padding]
                _, y_likelihoods[:, :, h:h + 1, w:w + 1] = self.gaussian_conditional(((y_crop - means_hat) * scale).unsqueeze(2).unsqueeze(3), (scales_hat * scale).unsqueeze(2).unsqueeze(3))
                if self.no_quantoffset:
                    y_q = self.quantizer.quantize((y_crop - means_hat.detach()) * scale, 'ste') * rescale + means_hat.detach()
                else:
                    y_zm = y_crop - means_hat
                    y_zm_sc = y_zm * scale
                    signs = torch.sign(y_zm_sc).detach()
                    q_abs = quantize_ste(torch.abs(y_zm_sc))
                    q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                    stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=2), scale.detach().expand(q_stdev.unsqueeze(dim=2).shape)), dim=2)
                    q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=2)
                    q_offsets[q_abs < 0.0001] = 0
                    y_q = signs * (q_abs + q_offsets)
                    y_q = y_q * rescale + means_hat
                y_hat[:, :, h + padding, w + padding] = y_q
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat, y_likelihoods

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, stage: 'int'=2, s: 'int'=1, inputscale=0):
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).', stacklevel=2)
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        rescale = torch.tensor(1.0) / scale
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding, scale, rescale, stage)
            y_strings.append(string)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding, scale, rescale, stage):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        if self.scl2ctx and stage == 2:
            ctx_scl = self.scale_to_context.forward(scale.view(1, 1)).view(1, -1, 1, 1)
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, bias=self.context_prediction.bias)
                if self.scl2ctx and stage == 2:
                    ctx_p = ctx_p + ctx_scl
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
                y_crop = y_crop[:, :, padding, padding]
                if stage == 1:
                    y_q = self.gaussian_conditional.quantize(y_crop, 'symbols', means_hat)
                    y_hat[:, :, h + padding, w + padding] = y_q + means_hat
                elif stage == 2:
                    if self.no_quantoffset or self.no_quantoffset is False and self.ste_recursive is False:
                        y_q = self.gaussian_conditional.quantize((y_crop - means_hat) * scale, 'symbols')
                        y_hat[:, :, h + padding, w + padding] = y_q * rescale + means_hat
                    else:
                        y_zm = y_crop - means_hat.detach()
                        y_zm_sc = y_zm * scale
                        signs = torch.sign(y_zm_sc).detach()
                        q_abs = quantize_ste(torch.abs(y_zm_sc))
                        q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                        stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=2), scale.detach().expand(q_stdev.unsqueeze(dim=2).shape)), dim=2)
                        q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=2)
                        q_offsets[q_abs < 0.0001] = 0
                        y_q = (signs * (q_abs + 0)).int()
                        y_hat[:, :, h + padding, w + padding] = signs * (q_abs + q_offsets) * rescale + means_hat
                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()
        return string

    def decompress(self, strings, shape, stage: 'int'=2, s: 'int'=1, inputscale=0):
        assert isinstance(strings, list) and len(strings) == 2
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).', stacklevel=2)
        if inputscale != 0:
            scale = inputscale
        else:
            assert s in range(0, self.levels), f's should in range(0, {self.levels}), but get s:{s}'
            scale = torch.abs(self.Gain[s])
        rescale = torch.tensor(1.0) / scale
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = torch.zeros((z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding), device=z_hat.device)
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(y_string, y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding, scale, rescale, stage)
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding, scale, rescale, stage):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        if stage == 2:
            if self.no_quantoffset is False and self.ste_recursive is False:
                y_rec = torch.zeros_like(y_hat)
        if self.scl2ctx and stage == 2:
            ctx_scl = self.scale_to_context.forward(scale.view(1, 1)).view(1, -1, 1, 1)
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, self.context_prediction.weight, bias=self.context_prediction.bias)
                if self.scl2ctx and stage == 2:
                    ctx_p = ctx_p + ctx_scl
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                if stage == 1:
                    rv = self.gaussian_conditional.dequantize(rv, means_hat)
                    hp = h + padding
                    wp = w + padding
                    y_hat[:, :, hp:hp + 1, wp:wp + 1] = rv
                elif stage == 2:
                    if self.no_quantoffset:
                        rv = self.gaussian_conditional.dequantize(rv) * rescale + means_hat
                        hp = h + padding
                        wp = w + padding
                        y_hat[:, :, hp:hp + 1, wp:wp + 1] = rv
                    else:
                        q_val = self.gaussian_conditional.dequantize(rv)
                        q_abs, signs = q_val.abs(), torch.sign(q_val)
                        q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
                        stdev_and_gain = torch.cat((q_stdev.unsqueeze(dim=4), scale.detach().expand(q_stdev.unsqueeze(dim=4).shape)), dim=4)
                        q_offsets = -1 * self.QuantABCD.forward(stdev_and_gain).squeeze(dim=4)
                        q_offsets[q_abs < 0.0001] = 0
                        rv_out = signs * (q_abs + q_offsets) * rescale + means_hat
                        hp = h + padding
                        wp = w + padding
                        if self.ste_recursive is False:
                            y_hat[:, :, hp:hp + 1, wp:wp + 1] = self.gaussian_conditional.dequantize(rv) * rescale + means_hat
                            y_rec[:, :, hp:hp + 1, wp:wp + 1] = rv_out
                        else:
                            y_hat[:, :, hp:hp + 1, wp:wp + 1] = rv_out
        if stage == 2:
            if self.no_quantoffset is False and self.ste_recursive is False:
                y_hat = y_rec


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2 ** bit_depth - 1
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_sub = torch.exp(-ctx.alpha ** ctx.beta * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta) * grad_output.clone()
        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]
        return grad_input, None, None


def gaussian_kernel1d(kernel_size: 'int', sigma: 'float', device: 'torch.device', dtype: 'torch.dtype'):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(kernel_size: 'int', sigma: 'float', device: 'torch.device', dtype: 'torch.dtype'):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError('Missing kernel_size or sigma parameters')
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)
    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='replicate')
    x = torch.nn.functional.conv2d(x, kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)), groups=x.size(1))
    return x


def meshgrid2d(N: 'int', C: 'int', H: 'int', W: 'int', device: 'torch.device'):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)


class ScaleSpaceFlow(CompressionModel):
    """Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(self, num_levels: 'int'=5, sigma0: 'float'=1.5, scale_field_shift: 'float'=1.0):
        super().__init__()


        class Encoder(nn.Sequential):

            def __init__(self, in_planes: 'int', mid_planes: 'int'=128, out_planes: 'int'=192):
                super().__init__(conv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, out_planes, kernel_size=5, stride=2))


        class Decoder(nn.Sequential):

            def __init__(self, out_planes: 'int', in_planes: 'int'=192, mid_planes: 'int'=128):
                super().__init__(deconv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, out_planes, kernel_size=5, stride=2))


        class HyperEncoder(nn.Sequential):

            def __init__(self, in_planes: 'int'=192, mid_planes: 'int'=192, out_planes: 'int'=192):
                super().__init__(conv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2))


        class HyperDecoder(nn.Sequential):

            def __init__(self, in_planes: 'int'=192, mid_planes: 'int'=192, out_planes: 'int'=192):
                super().__init__(deconv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, out_planes, kernel_size=5, stride=2))


        class HyperDecoderWithQReLU(nn.Module):

            def __init__(self, in_planes: 'int'=192, mid_planes: 'int'=192, out_planes: 'int'=192):
                super().__init__()

                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)
                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))
                return x


        class Hyperprior(CompressionModel):

            def __init__(self, planes: 'int'=192, mid_planes: 'int'=192):
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {'y': y_likelihoods, 'z': z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)
                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, 'dequantize', means)
                return y_hat, {'strings': [y_string, z_string], 'shape': z.size()[-2:]}

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)
                return y_hat
        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        self.img_hyperprior = Hyperprior()
        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3, in_planes=384)
        self.res_hyperprior = Hyperprior()
        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        self.motion_hyperprior = Hyperprior()
        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f'Invalid number of frames: {len(frames)}.')
        reconstructions = []
        frames_likelihoods = []
        x_hat, likelihoods = self.forward_keyframe(frames[0])
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()
        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, x_ref)
            reconstructions.append(x_ref)
            frames_likelihoods.append(likelihoods)
        return {'x_hat': reconstructions, 'likelihoods': frames_likelihoods}

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, likelihoods = self.img_hyperprior(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, {'keyframe': likelihoods}

    def encode_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, out_keyframe = self.img_hyperprior.compress(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, out_keyframe

    def decode_keyframe(self, strings, shape):
        y_hat = self.img_hyperprior.decompress(strings, shape)
        x_hat = self.img_decoder(y_hat)
        return x_hat

    def forward_inter(self, x_cur, x_ref):
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec, {'motion': motion_likelihoods, 'residual': res_likelihoods}

    def encode_inter(self, x_cur, x_ref):
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion)
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, out_res = self.res_hyperprior.compress(y_res)
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec, {'strings': {'motion': out_motion['strings'], 'residual': out_res['strings']}, 'shape': {'motion': out_motion['shape'], 'residual': out_res['shape']}}

    def decode_inter(self, x_ref, strings, shapes):
        key = 'motion'
        y_motion_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        key = 'residual'
        y_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key])
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: 'float', num_levels: 'int'):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(interp, scale_factor=2, mode='bilinear', align_corners=False)
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: 'str'='border'):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(f'Invalid number of dimensions for volume {volume.ndimension()}')
        N, C, _, H, W = volume.size()
        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)
        out = F.grid_sample(volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False)
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)
        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f'Invalid number of frames: {len(frames)}.')
        frame_strings = []
        shape_infos = []
        x_ref, out_keyframe = self.encode_keyframe(frames[0])
        frame_strings.append(out_keyframe['strings'])
        shape_infos.append(out_keyframe['shape'])
        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)
            frame_strings.append(out_interframe['strings'])
            shape_infos.append(out_interframe['shape'])
        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f'Invalid number of frames: {len(strings)}.')
        assert len(strings) == len(shapes), f'Number of information should match {len(strings)} != {len(shapes)}.'
        dec_frames = []
        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)
        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)
        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(ResidualBlockWithStride(3, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), conv3x3(N, N, stride=2))
        self.h_a = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2))
        self.h_s = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), subpel_conv3x3(N, N, 2), nn.LeakyReLU(inplace=True), conv3x3(N, N * 3 // 2), nn.LeakyReLU(inplace=True), subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2), nn.LeakyReLU(inplace=True), conv3x3(N * 3 // 2, N * 2))
        self.g_s = nn.Sequential(ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), subpel_conv3x3(N, 3, 2))

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.conv1.weight'].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = nn.Sequential(ResidualBlockWithStride(3, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), AttentionBlock(N), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), conv3x3(N, N, stride=2), AttentionBlock(N))
        self.g_s = nn.Sequential(AttentionBlock(N), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), AttentionBlock(N), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), subpel_conv3x3(N, 3, 2))


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class DummyCompressionModel(CompressionModel):

    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)


class Foo(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionBlock,
     lambda: ([], {'N': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CheckerboardMaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CustomDataParallel,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (DensityEmbeddingLayer,
     lambda: ([], {'hidden_dim': 4, 'dim': 4, 'ngroups': 1}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (GDN,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GDN1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Interleave,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Lambda,
     lambda: ([], {'func': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LowerBound,
     lambda: ([], {'bound': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NamedLayer,
     lambda: ([], {'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonNegativeParametrizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PointNetSetAbstraction,
     lambda: ([], {'npoint': 4, 'radius': 4, 'nsample': 4, 'in_channel': 4, 'mlp': [4, 4], 'group_all': 4}),
     lambda: ([torch.rand([4, 1, 4]), torch.rand([4, 3, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlockUpsample,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlockWithStride,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBottleneckBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpectralConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpectralConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (UpsampleNumLayer,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'upsample_rate': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

