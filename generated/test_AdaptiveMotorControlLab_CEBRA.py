
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


import abc


import torch


import types


from typing import List


from typing import Literal


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import numpy.typing as npt


import collections


import copy


import warnings


import scipy.linalg


from typing import IO


import pandas as pd


import scipy.io


from numpy.random import Generator


from numpy.random import PCG64


from sklearn.decomposition import PCA


import sklearn.model_selection


import sklearn.neighbors


import scipy.stats


import torch.distributions


from torch import nn


import scipy.interpolate


from typing import Iterable


import matplotlib.axes


import sklearn.utils.validation as sklearn_utils_validation


from functools import wraps


from collections.abc import Iterable


import matplotlib.cm


import matplotlib.colors


import matplotlib.figure


import matplotlib.pyplot as plt


import sklearn.utils.validation


import itertools


from typing import Callable


from typing import Dict


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


from typing import Generator


import sklearn


import sklearn.base


import sklearn.linear_model


import sklearn.decomposition


import math


import torch.nn.functional as F


import collections.abc as collections_abc


import sklearn.utils.estimator_checks


import sklearn.metrics


import functools


import matplotlib


from sklearn.exceptions import NotFittedError


import torch.nn as nn


import time


import logging


class PoissonNeuronTransform(nn.Module):
    """Transform spike rates into expected spike counts.

    This is an implementation for transforming arrays or tensors containing spike
    rates into expected spike counts.

    Args:
        num_neurons: The number of neurons to simulate. Needs to match the
            dimensions of the array passed to :py:meth:`__call__`.
        refractory_period: The neuron's absolute refractory period, in seconds.
            The absolute refactory period is the lower bound for the inter-spike
            interval for each neuron.

    References:
        https://neuronaldynamics.epfl.ch/online/Ch7.S3.html
    """

    def __init__(self, num_neurons: 'int', refractory_period: 'float'=0.0):
        super().__init__()
        if refractory_period < 0:
            raise ValueError(f'Refractory period needs to be non-negative, but got {refractory_period}')
        if num_neurons <= 0:
            raise ValueError(f'num_neurons needs to be positive, but got {num_neurons}')
        self.refractory_period = refractory_period
        self.num_neurons = num_neurons

    def __call__(self, spike_rates: 'torch.Tensor') ->torch.Tensor:
        """Sample spike counts from spike rates

        Args:
            spike_rates: The non-negative spike rates for each neuron, in a
                tensor with shape ``neurons x trials x timesteps``. The number
                of neurons needs to match :py:attr:`num_neurons`.

        Returns:
            A tensor of same shape as the input array, containing a sample
            of spike counts.
        """
        n_neurons, n_trials, n_timesteps = spike_rates.shape
        assert n_neurons == self.num_neurons
        time_interval = 1
        for n_sigmas in [4, 8, 12, 16, 24, 48, 96]:
            num_spikes = int(spike_rates.max() * (time_interval * (1 + n_sigmas)))
            delta_distribution = torch.distributions.Exponential(rate=spike_rates)
            deltas = delta_distribution.sample((num_spikes,)) + self.refractory_period
            spike_times = torch.cumsum(deltas, dim=0)
            if (spike_times[-2] > time_interval).all():
                break
        else:
            raise ValueError()
        return (spike_times < time_interval).sum(dim=0)


class ContrastiveLoss(nn.Module):
    """Base class for contrastive losses.

    Note:
        - Added in 0.0.2.
    """

    def forward(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the contrastive loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.
        """
        raise NotImplementedError()


@torch.jit.script
def infonce(pos_dist: 'torch.Tensor', neg_dist: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE implementation

    See :py:class:`BaseInfoNCE` for reference.

    Note:
        - The behavior of this function changed beginning in CEBRA 0.3.0.
        The InfoNCE implementation is numerically stabilized.
    """
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    c_mean = c.mean()
    align_corrected = align - c_mean
    uniform_corrected = uniform + c_mean
    return align + uniform, align_corrected, uniform_corrected


class BaseInfoNCE(ContrastiveLoss):
    """Base class for all InfoNCE losses.

    Given a similarity measure :math:`\\phi` which will be implemented by the subclasses
    of this class, the generalized InfoNCE loss is computed as

    .. math::

        \\sum_{i=1}^n - \\phi(x_i, y^{+}_i) + \\log \\sum_{j=1}^{n} e^{\\phi(x_i, y^{-}_{ij})}

    where :math:`n` is the batch size, :math:`x` are the reference samples (``ref``),
    :math:`y^{+}` are the positive samples (``pos``) and :math:`y^{-}` are the negative
    samples (``neg``).

    """

    def _distance(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor]:
        """The similarity measure.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        Returns:
            The distance between reference samples and positive samples of shape `(n,)`, and
            the distances between reference samples and negative samples of shape `(n, n)`.

        """
        raise NotImplementedError()

    def forward(self, ref, pos, neg) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        See Also:
            :py:class:`BaseInfoNCE`.
        """
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return infonce(pos_dist, neg_dist)


class FixedInfoNCE(BaseInfoNCE):
    """InfoNCE base loss with a fixed temperature.

    Attributes:
        temperature:
            The softmax temperature
    """

    def __init__(self, temperature: 'float'=1.0):
        super().__init__()
        self.temperature = temperature


class LearnableInfoNCE(BaseInfoNCE):
    """InfoNCE base loss with a learnable temperature.

    Attributes:
        temperature:
            The current value of the learnable temperature parameter.
        min_temperature:
            The minimum temperature to use. Increase the minimum temperature
            if you encounter numerical issues during optimization.
    """

    def __init__(self, temperature: 'float'=1.0, min_temperature: 'Optional[float]'=None):
        super().__init__()
        if min_temperature is None:
            self.max_inverse_temperature = math.inf
        else:
            self.max_inverse_temperature = 1.0 / min_temperature
        log_inverse_temperature = torch.tensor(math.log(1.0 / float(temperature)))
        self.log_inverse_temperature = nn.Parameter(log_inverse_temperature)
        self.min_temperature = min_temperature

    @torch.jit.export
    def _prepare_inverse_temperature(self) ->torch.Tensor:
        """Compute the current inverse temperature."""
        inverse_temperature = torch.exp(self.log_inverse_temperature)
        inverse_temperature = torch.clamp(inverse_temperature, max=self.max_inverse_temperature)
        return inverse_temperature

    @property
    def temperature(self) ->float:
        with torch.no_grad():
            return 1.0 / self._prepare_inverse_temperature().item()


@torch.jit.script
def dot_similarity(ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum('ni,ni->n', ref, pos)
    neg_dist = torch.einsum('ni,mi->nm', ref, neg)
    return pos_dist, neg_dist


class FixedCosineInfoNCE(FixedInfoNCE):
    """Cosine similarity function with fixed temperature.

    The similarity metric is given as

    .. math ::

        \\phi(x, y) =  x^\\top y  / \\tau

    with fixed temperature :math:`\\tau > 0`.

    Note that this loss function should typically only be used with normalized.
    This class itself does *not* perform any checks. Ensure that :math:`x` and
    :math:`y` are normalized.
    """

    @torch.jit.export
    def _distance(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


@torch.jit.script
def euclidean_similarity(ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Negative L2 distance between the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    ref_sq = torch.einsum('ni->n', ref ** 2)
    pos_sq = torch.einsum('ni->n', pos ** 2)
    neg_sq = torch.einsum('ni->n', neg ** 2)
    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)
    return pos_dist, neg_dist


class FixedEuclideanInfoNCE(FixedInfoNCE):
    """L2 similarity function with fixed temperature.

    The similarity metric is given as

    .. math ::

        \\phi(x, y) =  - \\| x - y \\| / \\tau

    with fixed temperature :math:`\\tau > 0`.
    """

    @torch.jit.export
    def _distance(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


class LearnableCosineInfoNCE(LearnableInfoNCE):
    """Cosine similarity function with a learnable temperature.

    Like :py:class:`FixedCosineInfoNCE`, but with a learnable temperature
    parameter :math:`\\tau`.
    """

    @torch.jit.export
    def _distance(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = dot_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


class LearnableEuclideanInfoNCE(LearnableInfoNCE):
    """L2 similarity function with fixed temperature.

    Like :py:class:`FixedEuclideanInfoNCE`, but with a learnable temperature
    parameter :math:`\\tau`.
    """

    @torch.jit.export
    def _distance(self, ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = euclidean_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


class NCE(ContrastiveLoss):
    """Noise contrastive estimation (Gutman & Hyvarinen, 2012)

    Attributes:
        temperature (float): The softmax temperature
        negative_weight (float): Relative weight of the negative samples
        reduce (str): How to reduce the negative samples. Can be
            ``sum`` or ``mean``.
    """

    def __init__(self, temperature=1.0, negative_weight=1.0, reduce='mean'):
        super().__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight
        assert reduce in ['mean', 'sum']
        self._reduce = getattr(torch, reduce)

    def forward(self, ref, pos, neg):
        """Compute the NCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        See Also:
            :py:class:`NCE`.
        """
        pos_dist = torch.einsum('ni,ni->n', ref, pos) / self.temperature
        neg_dist = torch.einsum('ni,mi->nm', ref, neg) / self.temperature
        align = F.logsigmoid(pos_dist)
        uniform = self._reduce(F.logsigmoid(-neg_dist), dim=1)
        return align + self.negative_weight * uniform, align, uniform


class _Skip(nn.Module):
    """Add a skip connection to a list of modules

    Args:
        *modules (torch.nn.Module): Modules to add to the bottleneck
        crop (tuple of ints): Number of timesteps to crop around the
            shortcut of the module to match the output with the bottleneck
            layers. This can be typically inferred from the strides/sizes
            of any conv layers within the bottleneck.
    """

    def __init__(self, *modules, crop=(1, 1)):
        super().__init__()
        self.module = nn.Sequential(*modules)
        self.crop = slice(crop[0], -crop[1] if isinstance(crop[1], int) and crop[1] > 0 else None)

    def forward(self, inp: 'torch.Tensor') ->torch.Tensor:
        """Compute forward pass through the skip connection.

        Implements the operation ``self.module(inp[..., self.crop]) + skip``.

        Args:
            inp: 3D input tensor

        Returns:
            3D output tensor of same dimension as `inp`.
        """
        skip = self.module(inp)
        return inp[..., self.crop] + skip


class Squeeze(nn.Module):
    """Squeeze 3rd dimension of input tensor, pass through otherwise."""

    def forward(self, inp: 'torch.Tensor') ->torch.Tensor:
        """Squeeze 3rd dimension of input tensor, pass through otherwise.

        Args:
            inp: 1-3D input tensor

        Returns:
            If the third dimension of the input tensor can be squeezed,
            return the resulting 2D output tensor. If input is 2D or less,
            return the input.
        """
        if inp.dim() > 2:
            return inp.squeeze(2)
        return inp


class _Norm(nn.Module):
    """Normalize the input tensor across its first dimension.

    TODO:
        * Move this class to ``cebra.models.layers``.
    """

    def forward(self, inp):
        """Normalize the input tensor across its first dimension."""
        return inp / torch.norm(inp, dim=1, keepdim=True)


class _MeanAndConv(nn.Module):

    def __init__(self, inp, output, kernel, *, stride):
        super().__init__()
        self.downsample = stride
        self.layer = nn.Conv1d(inp, output, kernel, stride=stride)

    def forward(self, inp: 'torch.Tensor') ->torch.Tensor:
        connect = self.layer(inp)
        downsampled = F.interpolate(inp, scale_factor=1 / self.downsample)
        return torch.cat([connect, downsampled[..., :connect.size(-1)]], dim=1)


class HasFeatureEncoder:
    """Networks with an explicitly defined feature encoder."""

    @property
    def feature_encoder(self) ->nn.Module:
        return self.net


class ConvolutionalModelMixin:
    """Mixin for models that support operating on a time-series.

    The input for convolutional models should be ``batch, dim, time``
    and the convolution will be applied across the last dimension.
    """
    pass


class ResampleModelMixin:
    """Mixin for models that re-sample the signal over time."""

    @property
    def resample_factor(self) ->float:
        """The factor by which the signal is downsampled."""
        return NotImplementedError()


class MultiobjectiveModel(nn.Module):
    """Wrapper around contrastive learning models to all training with multiple objectives

    Multi-objective training splits the last layer's feature representation into multiple
    chunks, which are then used for individual training objectives.

    Args:
        module: The module to wrap
        dimensions: A tuple of dimension values to extract from the model's feature embedding.
        renormalize: If True, the individual feature slices will be re-normalized before
            getting returned---this option only makes sense in conjunction with a loss based
            on the cosine distance or dot product.
        output_mode: A mode as defined in ``MultiobjectiveModel.Mode``. Overlapping means that
            when ``dimensions`` are set to `(x0, x1, ...)``, features will be extracted from
            ``0:x0, 0:x1, ...``. When mode is set to separate, features are extracted from
            ``x0:x1, x1:x2, ...``.
        append_last_dimension: Defaults to True, and will allow to omit the last dimension in
            the ``dimensions`` argument (which should be equal to the output dimension) of the
            given model.

    TODO:
        - Update nn.Module type annotation for ``module`` to cebra.models.Model
    """


    class Mode:
        """Mode for slicing and potentially normalizing the output embedding.

        The options are:

        - ``OVERLAPPING``: When ``dimensions`` are set to `(x0, x1, ...)``, features will be
          extracted from ``0:x0, 0:x1, ...``.
        - ``SEPARATE``: Features are extracted from ``x0:x1, x1:x2, ...``

        """
        OVERLAPPING = 'overlapping'
        SEPARATE = 'separate'
        _ALL = {OVERLAPPING, SEPARATE}

        def is_valid(self, mode):
            """Check if a given string representation is valid.

            Args:
                mode: String representation of the mode.

            Returns:
                ``True`` for a valid representation, ``False`` otherwise.
            """
            return mode in _ALL

    def __init__(self, module: 'nn.Module', dimensions: 'Tuple[int]', renormalize: 'bool'=False, output_mode: 'str'='overlapping', append_last_dimension: 'bool'=False):
        super().__init__()
        if not isinstance(module, cebra.models.Model):
            raise ValueError(f'Can only wrap models that are subclassing the cebra.models.Model abstract base class. Got a model of type {type(module)}.')
        self.module = module
        self.renormalize = renormalize
        self.output_mode = output_mode
        self._norm = _Norm()
        self._compute_slices(dimensions, append_last_dimension)

    @property
    def get_offset(self):
        """See :py:meth:`cebra.models.model.Model.get_offset`."""
        return self.module.get_offset

    @property
    def num_output(self):
        """See :py:attr:`cebra.models.model.Model.num_output`."""
        return self.module.num_output

    def _compute_slices(self, dimensions, append_last_dimension):

        def _valid_dimensions(dimensions):
            return max(dimensions) == self.num_output
        if append_last_dimension:
            if _valid_dimensions(dimensions):
                raise ValueError(f'append_last_dimension should only be used if extra values are available. Last requested dimensionality is already {dimensions[-1]}.')
            dimensions += self.num_output,
        if not _valid_dimensions(dimensions):
            raise ValueError(f'Max of given dimensions needs to match the number of outputs in the encoder network. Got {dimensions} and expected a maximum value of {self.num_output}.')
        if self.output_mode == self.Mode.OVERLAPPING:
            self.feature_ranges = tuple(slice(0, dimension) for dimension in dimensions)
        elif self.output_mode == self.Mode.SEPARATE:
            from_dimension = (0,) + dimensions
            self.feature_ranges = tuple(slice(i, j) for i, j in zip(from_dimension, dimensions))
        else:
            raise ValueError(f"Unknown mode: '{self.output_mode}', use one of {self.Mode._ALL}.")

    def forward(self, inputs):
        """Compute multiple embeddings for a single signal input.

        Args:
            inputs: The input tensor

        Returns:
            A tuple of tensors which are sliced according to `self.feature_ranges`
            if `renormalize` is set to true, each of the tensors will be normalized
            across the first (feature) dimension.

        TODO:
            - Cover this function with unit tests
        """
        output = self.module(inputs)
        outputs = (output[:, slice_features] for slice_features in self.feature_ranges)
        if self.renormalize:
            outputs = (self._norm(output) for output in outputs)
        return tuple(outputs)


class _Squeeze(nn.Module):

    def forward(self, inp):
        return inp.squeeze(2)


class PointwiseLinear(nn.Module):
    """Pointwise linear layer, mapping (d,i) -> (d,j) features."""

    def __init__(self, num_parallel, num_inputs, num_outputs):
        super().__init__()

        def uniform(a, b, size):
            r = torch.rand(size)
            return r * (b - a) + a
        weight = uniform(-num_inputs ** 0.5, num_inputs ** 0.5, size=(num_parallel, num_inputs, num_outputs))
        bias = uniform(-num_inputs ** 0.5, num_inputs ** 0.5, size=(1, 1, num_outputs))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, inputs):
        return torch.einsum('ndi,dij->ndj', inputs, self.weight) + self.bias


class PointwiseProjector(nn.Module):
    """Projector, applied pointwise to feature output"""

    def __init__(self, num_inputs, num_units):
        super().__init__()
        self.net = nn.Sequential(PointwiseLinear(num_inputs, 1, num_units), cebra_models_layers._Skip(PointwiseLinear(num_inputs, num_units, num_units), nn.GELU()), cebra_models_layers._Skip(PointwiseLinear(num_inputs, num_units, num_units), nn.GELU()), PointwiseLinear(num_inputs, num_units, 1))
        self.norm = cebra_models_layers._Norm()

    def forward(self, inputs):
        return self.norm(self.net(inputs[:, :, None]).squeeze(2))


class FeatureExtractor(nn.Sequential):

    def __init__(self, num_neurons, num_units, num_output):
        super().__init__(nn.Conv1d(num_neurons, num_units, 2), nn.GELU(), cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()), cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()), cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()), nn.Conv1d(num_units, num_output, 3), _Squeeze())


@torch.jit.script
def ref_dot_similarity(ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor', temperature: 'float'):
    pos_dist = torch.einsum('ni,ni->n', ref, pos) / temperature
    neg_dist = torch.einsum('ni,mi->nm', ref, neg) / temperature
    return pos_dist, neg_dist


@torch.jit.script
def ref_infonce(pos_dist: 'torch.Tensor', neg_dist: 'torch.Tensor'):
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform


class ReferenceInfoNCE(nn.Module):
    """The InfoNCE loss.
    Attributes:
        temperature (float): The softmax temperature
    """

    def __init__(self, temperature: 'float'=1.0):
        super().__init__()
        self.temperature = temperature

    def _distance(self, ref, pos, neg):
        return ref_dot_similarity(ref, pos, neg, self.temperature)

    def forward(self, ref, pos, neg):
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return ref_infonce(pos_dist, neg_dist)


@torch.jit.script
def ref_euclidean_similarity(ref: 'torch.Tensor', pos: 'torch.Tensor', neg: 'torch.Tensor', temperature: 'float'):
    ref_sq = torch.einsum('ni->n', ref ** 2) / temperature
    pos_sq = torch.einsum('ni->n', pos ** 2) / temperature
    neg_sq = torch.einsum('ni->n', neg ** 2) / temperature
    pos_cosine, neg_cosine = ref_dot_similarity(ref, pos, neg, temperature)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)
    return pos_dist, neg_dist


class ReferenceInfoMSE(ReferenceInfoNCE):
    """A variant of the InfoNCE loss using a MSE error.
    Attributes:
        temperature (float): The softmax temperature
    """

    def _distance(self, ref, pos, neg):
        return ref_euclidean_similarity(ref, pos, neg, self.temperature)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FixedCosineInfoNCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (FixedEuclideanInfoNCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (LearnableCosineInfoNCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (LearnableEuclideanInfoNCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (NCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (PointwiseLinear,
     lambda: ([], {'num_parallel': 4, 'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PoissonNeuronTransform,
     lambda: ([], {'num_neurons': 4}),
     lambda: ([], {'spike_rates': torch.rand([4, 4, 4])})),
    (ReferenceInfoMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ReferenceInfoNCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_MeanAndConv,
     lambda: ([], {'inp': 4, 'output': 4, 'kernel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (_Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_Skip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 0])], {})),
    (_Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

