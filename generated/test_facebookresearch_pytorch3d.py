
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


import math


import torch


import numpy as np


from torch import nn


from torch import optim


import warnings


from typing import Optional


import torch.optim


import inspect


from collections import defaultdict


from typing import Any


from typing import Dict


from typing import List


from typing import Tuple


import time


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import random


from torch.utils.data.dataset import Dataset


import collections


from typing import Sequence


from typing import Union


import torch.nn.functional as F


from torch.nn import init


from torch.nn import Parameter


from enum import Enum


from typing import Iterator


from torch.utils.data import BatchSampler


from torch.utils.data import ConcatDataset


from torch.utils.data import RandomSampler


from torch.utils.data import Sampler


from typing import ClassVar


from typing import Iterable


from typing import Type


from abc import ABC


from abc import abstractmethod


from typing import Generic


from typing import Mapping


from typing import TypeVar


from collections import Counter


from torch.utils.data.sampler import Sampler


import pandas as pd


import functools


from typing import cast


import copy


from collections import OrderedDict


from typing import TYPE_CHECKING


import torch.nn.functional as Fu


import torchvision


from typing import Callable


from collections.abc import Mapping


import torch.nn as nn


from logging import Logger


import torchvision.utils


from torch.nn import functional as F


from math import pi


import itertools


from functools import partial


from typing import get_args


from typing import get_origin


from collections import deque


from enum import IntEnum


from typing import BinaryIO


from collections import namedtuple


from typing import Deque


from typing import ContextManager


from typing import IO


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from typing import NamedTuple


from random import randint


from torch.nn.functional import interpolate


from itertools import zip_longest


from itertools import tee


from math import cos


from math import sin


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CUDAExtension


from itertools import product


from torch.autograd import Variable


import typing


from numbers import Real


from typing import Generator


from math import radians


from torch.nn.functional import normalize


from torch.distributions import MultivariateNormal


import re


def axis_angle_to_quaternion(axis_angle: 'torch.Tensor') ->torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-06
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - angles[small_angles] * angles[small_angles] / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def quaternion_to_matrix(quaternions: 'torch.Tensor') ->torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r), two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r), two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: 'torch.Tensor') ->torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rotation_6d_to_matrix(d6: 'torch.Tensor') ->torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


height = 1000


n_points = 10


width = 1000


class SceneModel(nn.Module):
    """A simple model to demonstrate use in Modules."""

    def __init__(self):
        super(SceneModel, self).__init__()
        self.gamma = 1.0
        torch.manual_seed(1)
        vert_pos = torch.rand((1, n_points, 3), dtype=torch.float32) * 10.0
        vert_pos[:, :, 2] += 25.0
        vert_pos[:, :, :2] -= 5.0
        self.register_parameter('vert_pos', nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter('vert_col', nn.Parameter(torch.zeros(1, n_points, 3, dtype=torch.float32), requires_grad=True))
        self.register_parameter('vert_rad', nn.Parameter(torch.ones(1, n_points, dtype=torch.float32) * 0.001, requires_grad=False))
        self.register_parameter('vert_opy', nn.Parameter(torch.ones(1, n_points, dtype=torch.float32), requires_grad=False))
        self.register_buffer('cam_params', torch.tensor([[np.sin(angle) * 35.0, 0.0, 30.0 - np.cos(angle) * 35.0, 0.0, -angle, 0.0, 5.0, 2.0] for angle in [-1.5, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.5]], dtype=torch.float32))
        self.renderer = Renderer(width, height, n_points)

    def forward(self, cam=None):
        if cam is None:
            cam = self.cam_params
            n_views = 8
        else:
            n_views = 1
        return self.renderer.forward(self.vert_pos.expand(n_views, -1, -1), self.vert_col.expand(n_views, -1, -1), self.vert_rad.expand(n_views, -1), cam, self.gamma, 45.0, return_forward_info=True)


class Node(torch.nn.Module):

    def __init__(self, children=(), params=(), param_groups=None):
        super().__init__()
        for i, child in enumerate(children):
            self.add_module('m' + str(i), child)
        for i, param in enumerate(params):
            setattr(self, 'p' + str(i), param)
        if param_groups is not None:
            self.param_groups = param_groups

    def __str__(self):
        return 'modules:\n' + str(self._modules) + '\nparameters\n' + str(self._parameters)


class HarmonicEmbedding(torch.nn.Module):

    def __init__(self, n_harmonic_functions: 'int'=6, omega_0: 'float'=1.0, logspace: 'bool'=True, append_input: 'bool'=True) ->None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (n_harmonic_functions - 1), n_harmonic_functions, dtype=torch.float32)
        self.register_buffer('_frequencies', frequencies * omega_0, persistent=False)
        self.register_buffer('_zero_half_pi', torch.tensor([0.0, 0.5 * torch.pi]), persistent=False)
        self.append_input = append_input

    def forward(self, x: 'torch.Tensor', diag_cov: 'Optional[torch.Tensor]'=None, **kwargs) ->torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """
        embed = x[..., None] * self._frequencies
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)
            embed = embed * exp_var[..., None, :, :]
        embed = embed.reshape(*x.shape[:-1], -1)
        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(input_dims: 'int', n_harmonic_functions: 'int', append_input: 'bool') ->int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: 'int'=3) ->int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(input_dims, len(self._frequencies), self.append_input)


class LinearWithRepeat(torch.nn.Module):
    """
    if x has shape (..., k, n1)
    and y has shape (..., n2)
    then
    LinearWithRepeat(n1 + n2, out_features).forward((x,y))
    is equivalent to
    Linear(n1 + n2, out_features).forward(
        torch.cat([x, y.unsqueeze(-2).expand(..., k, n2)], dim=-1)
    )

    Or visually:
    Given the following, for each ray,

                feature   ->

    ray         xxxxxxxx
    position    xxxxxxxx
      |         xxxxxxxx
      v         xxxxxxxx


    and
                            yyyyyyyy

    where the y's do not depend on the position
    but only on the ray,
    we want to evaluate a Linear layer on both
    types of data at every position.

    It's as if we constructed

                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy
                xxxxxxxxyyyyyyyy

    and sent that through the Linear.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, device=None, dtype=None) ->None:
        """
        Copied from torch.nn.Linear.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """
        Copied from torch.nn.Linear.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: 'Tuple[torch.Tensor, torch.Tensor]') ->torch.Tensor:
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


def _is_actually_dataclass(some_class) ->bool:
    return '__dataclass_fields__' in some_class.__dict__

