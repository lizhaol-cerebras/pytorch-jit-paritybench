
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


import torch


from typing import Optional


import torch.nn as nn


import numpy as np


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


from abc import ABC


from abc import abstractmethod


from typing import List


from typing import Dict


from typing import Any


from typing import Tuple


import types


from functools import partial


import inspect


import enum


import copy


import typing


from typing import get_type_hints


from typing import Type


from typing import Callable


from functools import lru_cache


from typing import Union


from typing import TYPE_CHECKING


import torch.nn.functional as F


from typing import Set


from typing import Iterator


from torch.utils.data import Dataset


import logging as log


import re


from torch.multiprocessing import Pool


from copy import deepcopy


import random


import math


import collections


from torch.utils.data._utils.collate import default_convert


from torch.utils.data._utils.collate import default_collate_err_msg_format


from typing import DefaultDict


from collections import defaultdict


from scipy.stats import ortho_group


import time


import torchvision


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage import gaussian_filter


import abc


from typing import Iterable


import queue


from typing import Literal


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.utils.data import DataLoader


import pandas as pd


class SigDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, activation, bias):
        """Initialize the SigDecoder.
        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            hidden_dim (int): Hidden dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.hidden_layer = nn.Linear(self.input_dim, hidden_dim, bias=bias)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim, bias=bias)

    def forward_feature(self, x):
        """A specialized forward function for the MLP, to obtain 3 hidden channels, post sigmoid activation.
        after

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., 3]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        return x_h[..., :3]

    def forward(self, x):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., output_dim]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        x_h[..., 3:] = self.activation(x_h[..., 3:])
        out = self.output_layer(x_h)
        return out


class WispModule(nn.Module, ABC):
    """ A general interface for all Wisp building blocks, such as neural fields, grids and tracers.
        WispModules should:
        1. Provide their name & dictionary of public properties. That makes them compatible with systems like
        logging & gui.
        2. WispModules extend torch's nn.Module out of convenience.
        Modules are not required however, to implement a forward() function.
    """

    def __init__(self):
        super().__init__()

    def name(self) ->str:
        """
        Returns:
            (str) A WispModule should be given a meaningful, human readable name.
            By default, the class name is used.
        """
        return type(self).__name__

    @abstractmethod
    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        raise NotImplementedError('Wisp modules should implement the `public_properties` method')


class FullSort(nn.Module):
    """The "FullSort" activation function from https://arxiv.org/abs/1811.05381.
    """

    def forward(self, x):
        """Sorts the feature dimension.
        
        Args:
            x (torch.FloatTensor): Some tensor of shape [..., feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [..., feature_size]
        """
        return torch.sort(x, dim=-1)[0]


class MinMax(nn.Module):
    """The "MinMax" activation function from https://arxiv.org/abs/1811.05381.
    """

    def forward(self, x):
        """Partially sorts the feature dimension.
        
        The feature dimension needs to be a multiple of 2.

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [batch, feature_size]
        """
        N, M = x.shape
        x = x.reshape(N, M // 2, 2)
        return torch.cat([x.min(-1, keepdim=True)[0], x.max(-1, keepdim=True)[0]], dim=-1).reshape(N, M)


class Identity(nn.Module):
    """Identity function. Occasionally useful.
    """

    def forward(self, x):
        """Returns the input. :)

        Args:
            x (Any): Anything

        Returns:
            (Any): The input!
        """
        return x


class BasicDecoder(WispModule):
    """Super basic but super useful MLP class.
    """

    def __init__(self, input_dim, output_dim, activation, bias, layer=nn.Linear, num_layers=1, hidden_dim=128, skip=[]):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim + self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]
        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = self.activation(l(h))
                h = torch.cat([x, h], dim=-1)
            else:
                h = self.activation(l(h))
        out = self.lout(h)
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)

    def name(self) ->str:
        """ A human readable name for the given wisp module. """
        return 'BasicDecoder'

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {'Input Dim': self.input_dim, 'Hidden Dim': self.hidden_dim, 'Outpt Dim': self.output_dim, 'Num. Layers': self.num_layers, 'Layer Type': self.layer.__name__, 'Activation': self.activation.__name__, 'Bias': self.bias, 'Skip Connections': self.skip}


class PositionalEmbedder(WispModule):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers.
    """

    def __init__(self, num_freq, max_freq_log2, log_sampling=True, include_input=True, input_dim=3):
        """Initialize the module.

        Args:
            num_freq (int): The number of frequency bands to sample. 
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()
        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        if self.log_sampling:
            self.bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freq)
        else:
            self.bands = torch.linspace(1, 2.0 ** max_freq_log2, steps=num_freq)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords):
        """Embeds the coordinates.

        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        winded = (coords[:, None] * self.bands[None, :, None]).reshape(N, coords.shape[1] * self.num_freq)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) ->str:
        """ A human readable name for the given wisp module. """
        return 'Positional Encoding'

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {'Output Dim': self.out_dim, 'Num. Frequencies': self.num_freq, 'Max Frequency': f'2^{self.max_freq_log2}', 'Include Input': self.include_input}


@dataclass
class ASQueryResults:
    """ A data holder for keeping the results of a single acceleration structure query() call.
    A query receives a set of input coordinates and returns the cell indices of the acceleration structure where
    the query coordinates fall.
    """
    pidx: 'torch.LongTensor'
    """ Holds the query results.
    - If the query is invoked with `with_parents=False`, this field is a tensor of shape [num_coords],
      containing indices of cells of the acceleration structure, where the query coordinates match.
    - If the query is invoked with `with_parents=True`, this field is a tensor of shape [num_coords, level+1],
      containing indices of the cells of the acceleration structure + the full parent hierarchy of each 
      cell query result.
    """


@dataclass
class ASRaytraceResults:
    """ A data holder for keeping the results of a single acceleration structure raytrace() call.
    A raytrace operation returns all intersections of the ray set with the acceleration structure cells.
    Ray/cell intersections are also referred to in Kaolin & Wisp as "nuggets".
    """
    ridx: 'torch.LongTensor'
    """ A tensor containing the ray index of the ray that intersected each cell [num_nuggets]. 
    (can be used to index into rays.origins and rays.dirs)
    """
    pidx: 'torch.LongTensor'
    """ Point indices into the cells of the acceleration structure, where the ray intersected a cell [num_nuggets] """
    depth: 'torch.FloatTensor'
    """ Depths of each nugget, representing:
      - The first intersection point of the ray with the cell (entry), and 
      - Optionally also a second intersection point of the ray with the cell (exit).
      A tensor of [num_intersections, 1 or 2]. 
    """


class MultiTable(nn.Module):
    """Class that holds multiresolution grid tables.
    """

    def __init__(self, resolutions: 'Tuple[int, ...]', coord_dim: 'int', feature_dim: 'int', std: 'float'=0.01, max_feats: 'Optional[int]'=None):
        """
        Args:
            resolutions (List[int, ...]): The resolutions in the multiresolution hierarchy.
            coord_dim (int): The coordinate dimension for the grid.
            feature_dim (int): The feature dimension for the grid.
            std (float): The standard deviation for the features.
            max_feats (Optional[int]): The max number of features (when in use for hash grids, for example)
        """
        super().__init__()
        self.num_lods = len(resolutions)
        self.max_feats = max_feats
        self.register_buffer('begin_idxes', torch.zeros(self.num_lods + 1, dtype=torch.int64))
        self.register_buffer('num_feats', torch.zeros(self.num_lods, dtype=torch.int64))
        self.coord_dim = coord_dim
        self.feature_dim = feature_dim
        self.resolutions = torch.zeros([self.num_lods, 1], dtype=torch.int64)
        for i in range(len(resolutions)):
            self.resolutions[i] = resolutions[i]
        num_so_far = 0
        for i in range(self.num_lods):
            resolution = self.resolutions[i]
            num_feats_level = resolution[0] ** self.coord_dim
            if self.max_feats:
                num_feats_level = min(self.max_feats, num_feats_level)
            self.begin_idxes[i] = num_so_far
            self.num_feats[i] = num_feats_level
            num_so_far += num_feats_level
        self.begin_idxes[self.num_lods] = num_so_far
        self.total_feats = sum(self.num_feats)
        self.feats = nn.Parameter(torch.randn(self.total_feats, self.feature_dim) * std)

    def get_level(self, idx):
        """Gets the features for a specific level.

        Args:
            idx (int): The level of the multiresolution grid to get.
        """
        return self.feats[self.begin_idxes[idx]:self.begin_idxes[idx + 1]]


@torch.jit.script
def fast_filter_method(mask_idx: 'torch.Tensor', depth: 'torch.Tensor', deltas: 'torch.Tensor', samples: 'torch.Tensor', num_samples: 'int', num_rays: 'int', device: 'torch.device') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    depth_samples = depth[mask_idx[:, 0], mask_idx[:, 1]][:, None]
    deltas = deltas[mask_idx[:, 0], mask_idx[:, 1]].reshape(-1, 1)
    samples = samples[mask_idx[:, 0], mask_idx[:, 1], :]
    ridx = torch.arange(0, num_rays, device=device)
    ridx = ridx[..., None].repeat(1, num_samples)[mask_idx[:, 0], mask_idx[:, 1]]
    return depth_samples, deltas, samples, ridx


class TriplanarFeatureVolume(WispModule):
    """Triplanar feature volume represents a single triplane, e.g. a single LOD in a TriplanarGrid. """

    def __init__(self, fdim, fsize, std, bias):
        """Initializes the feature triplane.

        Args:
            fdim (int): The feature dimension.
            fsize (int): The height and width of the texture map.
            std (float): The standard deviation for the Gaussian initialization.
            bias (float): The mean for the Gaussian initialization.
        """
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fmx = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.fmy = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.fmz = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.padding_mode = 'reflection'

    def forward(self, x):
        """Interpolates from the feature volume.

        Args:
            x (torch.FloatTensor): Coordinates of shape [batch, num_samples, 3] or [batch, 3].

        Returns:
            (torch.FloatTensor): Features of shape [batch, num_samples, fdim] or [batch, fdim].
        """
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3)
            samplex = F.grid_sample(self.fmx, sample_coords[..., [1, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            sampley = F.grid_sample(self.fmy, sample_coords[..., [0, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            samplez = F.grid_sample(self.fmz, sample_coords[..., [0, 1]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            sample = torch.stack([samplex, sampley, samplez], dim=1).permute(0, 3, 1, 2)
        else:
            sample_coords = x.reshape(1, N, 1, 3)
            samplex = F.grid_sample(self.fmx, sample_coords[..., [1, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            sampley = F.grid_sample(self.fmy, sample_coords[..., [0, 2]], align_corners=True, padding_modes=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            samplez = F.grid_sample(self.fmz, sample_coords[..., [0, 1]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            sample = torch.stack([samplex, sampley, samplez], dim=1)
        return sample

    def name(self) ->str:
        """ A human readable name for the given wisp module. """
        return 'TriplanarFeatureVolume'

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {'Resolution': f'3x{self.fsize}x{self.fsize}'}


def normalize_frobenius(x):
    """Normalizes the matrix according to the Frobenius norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    norm = torch.sqrt((torch.abs(x) ** 2).sum())
    return x / norm


class FrobeniusLinear(nn.Module):
    """A standard Linear layer which applies a Frobenius normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_frobenius(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def normalize_L_1(x):
    """Normalizes the matrix according to the L1 norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    abscolsum = torch.sum(torch.abs(x), dim=0)
    abscolsum = torch.min(torch.stack([1.0 / abscolsum, torch.ones_like(abscolsum)], dim=0), dim=0)[0]
    return x * abscolsum[None, :]


class L_1_Linear(nn.Module):
    """A standard Linear layer which applies a L1 normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_1(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def normalize_L_inf(x):
    """Normalizes the matrix according to the Linf norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    absrowsum = torch.sum(torch.abs(x), axis=1)
    absrowsum = torch.min(torch.stack([1.0 / absrowsum, torch.ones_like(absrowsum)], dim=0), dim=0)[0]
    return x * absrowsum[:, None]


class L_inf_Linear(nn.Module):
    """A standard Linear layer which applies a Linf normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_inf(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


class BaseNeuralField(WispModule):
    """The base class for all Neural Fields within Wisp.
    Neural Fields are defined as modules which take coordinates as input and output signals of some form.
    The term "Neural" is loosely used here to imply these modules are generally subject for optimization.

    The domain of neural fields in Wisp is flexible, and left up for the user to decide when implementing the subclass.
    Popular neural fields from the literature, such as Neural Radiance Fields (Mildenhall et al. 2020),
    and Neural Signed Distance Functions (SDFs) can be implemented by creating and registering
    the required forward functions (for i.e. rgb, density, sdf values).

    BaseNeuralField subclasses  usually consist of several optional components:
    - A feature grid (BLASGrid), sometimes also known as 'hybrid representations'.
      These are responsible for querying and interpolating features, often in the context of some 3D volume
      (but not limited to).
      Feature grids often employ some acceleration structure (i.e. OctreeAS),
      which can be used to accelerate spatial queries or raytracing ops,
      hence the term "BLAS" (Bottom Level Acceleration Structure).
    - A decoder (i.e. BasicDecoder) which can feeds on features (or coordinates / pos embeddings) and coverts
      them to output signals.
    - Other components such as positional embedders may be employed.

    BaseNeuralFields are generally meant to be compatible with BaseTracers, thus forming a complete pipeline of
    render-able neural primitives.
    """

    def __init__(self):
        super().__init__()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([channel for channels in self._forward_functions.values() for channel in channels])

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.
        By default, the device is queried from the first registered torch nn.parameter.
        Override this property to explicitly specify the device.

        Returns:
            (torch.device): The expected device for inputs to this neural field.
        """
        return next(self.parameters()).device

    def _register_forward_function(self, fn, channels):
        """Registers a forward function.

        Args:
            fn (function): Function to register.
            channels (list of str): Channel output names.
        """
        if isinstance(channels, str):
            channels = [channels]
        self._forward_functions[fn] = set(channels)

    @abstractmethod
    def register_forward_functions(self):
        """Register forward functions with the channels that they output.

        This function should be overrided and call `self._register_forward_function` to
        tell the class which functions output what output channels. The function can be called
        multiple times to register multiple functions.

        Example:

        ```
        self._register_forward_function(self.rgba, ["density", "rgb"])
        self._register_forward_function(self.sdf, ["sdf"])
        ```
        """
        pass

    def get_forward_function(self, channel):
        """Will return the function that will return the channel.

        Args:
            channel (str): The name of the channel to return.

        Returns:
            (function): Function that will return the function. Will return None if the channel is not supported.
        """
        if channel not in self.get_supported_channels():
            raise Exception(f'Channel {channel} is not supported in {self.__class__.__name__}')
        for fn in self._forward_functions:
            output_channels = self._forward_functions[fn]
            if channel in output_channels:
                return lambda *args, **kwargs: fn(*args, **kwargs)[channel]

    def get_supported_channels(self):
        """Returns the channels that are supported by this class.

        Returns:
            (set): Set of channel strings.
        """
        return self.supported_channels

    def prune(self):
        """Prunes the neural field components (i.e. grid, or blas) based on current state.
        Neural fields may override this function to allow trainers to periodically invoke this logic.

        For example:
            A NeRF may use a hash grid with an octree acceleration structure. Since features and occupancy are tracked
            by separate structures, calling this function may update the blas (occupancy structure)
            with which cells should be marked as empty, according to decoded features -> density.
        """
        pass

    def forward(self, channels=None, **kwargs):
        """Queries the neural field with channels.

        Args:
            channels (str or list of str or set of str): Requested channels. See return value for details.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (list or dict or torch.Tensor):
                If channels is a string, will return a tensor of the request channel.
                If channels is a list, will return a list of channels.
                If channels is a set, will return a dictionary of channels.
                If channels is None, will return a dictionary of all channels.
        """
        if not (isinstance(channels, str) or isinstance(channels, list) or isinstance(channels, set) or channels is None):
            raise Exception(f'Channels type invalid, got {type(channels)}.Make sure your arguments for the nef are provided as keyword arguments.')
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        unsupported_channels = requested_channels - self.get_supported_channels()
        if unsupported_channels:
            raise Exception(f'Channels {unsupported_channels} are not supported in {self.__class__.__name__}')
        filtered_forward_functions = []
        for fn in self._forward_functions:
            output_channels = self._forward_functions[fn]
            supported_channels = output_channels & requested_channels
            num_supported_channels = len(supported_channels)
            if num_supported_channels != 0:
                filtered_forward_functions.append((num_supported_channels, fn))
        filtered_forward_functions = sorted(filtered_forward_functions, key=lambda x: x[0], reverse=True)
        return_dict = {}
        for _, fn in filtered_forward_functions:
            torch.cuda.nvtx.range_push(f'{fn.__name__}')
            output_channels = self._forward_functions[fn]
            supported_channels = output_channels & requested_channels
            requested_channels = requested_channels - supported_channels
            if len(supported_channels) != 0:
                argspec = inspect.getfullargspec(fn)
                if argspec.defaults is None:
                    required_len = 0
                else:
                    required_len = len(argspec.defaults)
                required_args = argspec.args[:-required_len][1:]
                optional_args = argspec.args[-required_len:]
                input_args = {}
                for _arg in required_args:
                    if _arg not in kwargs:
                        raise Exception(f'Argument {_arg} not found as input to in {self.__class__.__name__}.{fn.__name__}()')
                    input_args[_arg] = kwargs[_arg]
                for _arg in optional_args:
                    if _arg in kwargs:
                        input_args[_arg] = kwargs[_arg]
                output = fn(**input_args)
                for channel in supported_channels:
                    return_dict[channel] = output[channel]
            torch.cuda.nvtx.range_pop()
        if isinstance(channels, str):
            if channels in return_dict:
                return return_dict[channels]
            else:
                return None
        elif isinstance(channels, list):
            return [return_dict[channel] for channel in channels]
        else:
            return return_dict

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return dict()


def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'none':
        return Identity()
    elif activation_type == 'fullsort':
        return FullSort()
    elif activation_type == 'minmax':
        return MinMax()
    elif activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'celu':
        return F.celu
    elif activation_type == 'selu':
        return F.selu
    elif activation_type == 'leaky_relu':
        return F.leaky_relu
    elif activation_type == 'gelu':
        return F.gelu
    else:
        assert False and 'activation type does not exist'


def spectral_norm_(*args, **kwargs):
    """Initializes a spectral norm layer.
    """
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


def get_layer_class(layer_type):
    """Convenience function to return the layer class name from text.

    Args:
        layer_type (str): Text name for the layer.

    Retunrs:
        (nn.Module): The layer to be used for the decoder.
    """
    if layer_type == 'none' or layer_type == 'linear':
        return nn.Linear
    elif layer_type == 'spectral_norm':
        return spectral_norm_
    elif layer_type == 'frobenius_norm':
        return FrobeniusLinear
    elif layer_type == 'l_1_norm':
        return L_1_Linear
    elif layer_type == 'l_inf_norm':
        return L_inf_Linear
    else:
        assert False and 'layer type does not exist'


def get_positional_embedder(frequencies, input_dim=3, include_input=True):
    """Utility function to get a positional encoding embedding.

    Args:
        frequencies (int): The number of frequencies used to define the PE:
            [2^0, 2^1, 2^2, ... 2^(frequencies - 1)].
        input_dim (int): The input coordinate dimension.
        include_input (bool): If true, will concatenate the input coords.

    Returns:
        (nn.Module, int):
        - The embedding module
        - The output dimension of the embedding.
    """
    encoder = PositionalEmbedder(frequencies, frequencies - 1, input_dim=input_dim, include_input=include_input)
    return encoder, encoder.out_dim


class ImageNeuralField(BaseNeuralField):
    """Model for encoding images.
    """

    def __init__(self, grid: 'BLASGrid', activation_type: 'str'='relu', layer_type: 'str'='none', hidden_dim: 'int'=128, num_layers: 'int'=1):
        super().__init__()
        self.grid = grid
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if self.grid.multiscale_type == 'cat':
            self.feature_dim = self.grid.feature_dim * len(self.grid.resolutions)
        else:
            self.feature_dim = self.grid.feature_dim
        self.embedder, self.embed_dim = get_positional_embedder(frequencies=3, include_input=True)
        self.embed_dim = 14
        self.input_dim = self.feature_dim + self.embed_dim
        self.decoder = BasicDecoder(self.input_dim, 3, get_activation_class(self.activation_type), True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers, hidden_dim=self.hidden_dim, skip=[])

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgb, ['rgb'])

    def rgb(self, coords, lod=None):
        """Compute color for some locations

        Inputs:
            coords            : packed float tensor of shape [batch, 3]
            lod               : int of lod
        Outputs:
            float tensor of shape [batch, 3]
        """
        if lod is None:
            lod = len(self.grid.resolutions) - 1
        batch, _ = coords.shape
        feats = self.grid.interpolate(coords, lod).reshape(-1, self.feature_dim)
        embedded_pos = self.embedder(coords).view(batch, self.embed_dim)
        fpos = torch.cat([feats, embedded_pos], dim=-1)
        rgb = torch.sigmoid(self.decoder(fpos))
        return rgb


def sample_unif_sphere(n):
    """Sample uniformly random points on a sphere.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """
    u = np.random.rand(2, n)
    z = 1 - 2 * u[0, :]
    r = np.sqrt(1.0 - z * z)
    phi = 2 * np.pi * u[1, :]
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).transpose()
    return xyz


class NeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self, grid: 'BLASGrid', pos_embedder: 'str'='none', view_embedder: 'str'='none', pos_multires: 'int'=10, view_multires: 'int'=4, position_input: 'bool'=False, activation_type: 'str'='relu', layer_type: 'str'='linear', hidden_dim: 'int'=128, num_layers: 'int'=1, bias: 'bool'=False, prune_density_decay: 'Optional[float]'=0.01 * 512 / np.sqrt(3), prune_min_density: 'Optional[float]'=0.6):
        """
        Creates a new NeRF instance, which maps 3D input coordinates + view directions to RGB + density.

        This neural field consists of:
         * A feature grid (backed by an acceleration structure to boost raymarching speed)
         * Color & density decoders
         * Optional: positional embedders for input position coords & view directions, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation
          * Pruning scheme for HashGrids

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, TriplanarGrid, CodebookGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the density decoder.
                 - 'identity': The sample coordinates are fed as is into the density decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the density decoder.
            view_embedder (str): Type of positional embedder to use for view directions.
                Options:
                 - 'none': No positional input is fed into the color decoder.
                 - 'identity': The view directions are fed as is into the color decoder.
                 - 'positional': The view directions are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the color decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            view_multires (int): Number of frequencies used for 'positional' embedding of view_embedder.
                 Used only if view_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of both decoders.
            num_layers (int): Number of hidden layers in both decoders.
            bias (bool): Whether to use bias in the decoders.
            prune_density_decay (Optional[float]): Decay rate of density per "prune step",
                 using the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
            prune_min_density (Optional[float]): Minimal density allowed for "cells" before they get pruned during a "prune step".
                 Used within the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
        """
        super().__init__()
        self.grid = grid
        self.pos_embedder_type = pos_embedder
        self.view_embedder_type = view_embedder
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires, include_input=position_input)
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires, include_input=True)
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.decoder_density, self.decoder_color = self.init_decoders(activation_type, layer_type, num_layers, hidden_dim)
        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density
        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none' and not include_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or embedder_type == 'none' and include_input:
            embedder, embed_dim = torch.nn.Identity(), 3
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, include_input=include_input)
        elif embedder_type == 'tcnn':
            embedder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Composite', 'nested': [{'n_dims_to_encode': 3, 'otype': 'SphericalHarmonics', 'degree': 4}]})
            embed_dim = 16
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder_density = BasicDecoder(input_dim=self.density_net_input_dim(), output_dim=16, activation=get_activation_class(activation_type), bias=self.bias, layer=get_layer_class(layer_type), num_layers=num_layers, hidden_dim=hidden_dim, skip=[])
        if decoder_density.lout.bias is not None:
            decoder_density.lout.bias.data[0] = 1.0
        decoder_color = BasicDecoder(input_dim=self.color_net_input_dim(), output_dim=3, activation=get_activation_class(activation_type), bias=self.bias, layer=get_layer_class(layer_type), num_layers=num_layers + 1, hidden_dim=hidden_dim, skip=[])
        return decoder_density, decoder_color

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.prune_density_decay is None or self.prune_min_density is None:
            return
        if self.grid is not None:
            if isinstance(self.grid, (HashGrid, TriplanarGrid)):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density
                self.grid.occupancy = self.grid.occupancy
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points
                res = 2.0 ** self.grid.blas.max_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0]))
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels='density')
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]
                mask = self.grid.occupancy > min_density
                _points = points[mask]
                if _points.shape[0] == 0:
                    return
                if hasattr(self.grid.blas.__class__, 'from_quantized_points'):
                    self.grid.blas = self.grid.blas.__class__.from_quantized_points(_points, self.grid.blas.max_level)
                else:
                    raise Exception(f'The BLAS {self.grid.blas.__class__.__name__} does not support initialization from_quantized_points, which is required for pruning.')
            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid.__class__.__name__}')

    def register_forward_functions(self):
        """Registers the forward function to call per requested channel.
        """
        self._register_forward_function(self.rgba, ['density', 'rgb'])

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape
        feats = self.grid.interpolate(coords, lod_idx).reshape(batch, self.effective_feature_dim())
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(batch, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)
        density_feats = self.decoder_density(feats)
        if self.view_embedder is not None:
            if self.view_embedder_type == 'tcnn':
                ray_d = (ray_d + 1.0) / 2.0
            embedded_dir = self.view_embedder(ray_d).view(batch, self.view_embed_dim)
            fdir = torch.cat([density_feats, embedded_dir], dim=-1)
        else:
            fdir = density_feats
        colors = torch.sigmoid(self.decoder_color(fdir[..., 1:]))
        density = torch.relu(density_feats[..., 0:1])
        return dict(rgb=colors, density=density)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def density_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 15 + self.view_embed_dim

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {'Grid': self.grid, 'Pos. Embedding': self.pos_embedder, 'View Embedding': self.view_embedder, 'Decoder (density)': self.decoder_density, 'Decoder (color)': self.decoder_color}
        if self.prune_density_decay is not None:
            properties['Pruning Density Decay'] = self.prune_density_decay
        if self.prune_min_density is not None:
            properties['Pruning Min Density'] = self.prune_min_density
        return properties


class NeuralSDF(BaseNeuralField):
    """Model for encoding neural signed distance functions (implicit surfaces).
    This field implementation uses feature grids for faster and more efficient queries.
    For example, the usage of Octree follows the idea from Takikawa et al. 2021 (Neural Geometric Level of Detail).
    """

    def __init__(self, grid: 'BLASGrid', pos_embedder: 'str'='positional', pos_multires: 'int'=4, position_input: 'bool'=True, activation_type: 'str'='relu', layer_type: 'str'='none', hidden_dim: 'int'=128, num_layers: 'int'=1):
        """
        Creates a new neural field of an implicit furface, which maps 3D input coordinates to SDF values.

        This neural field consists of:
         * A feature grid (backed by an acceleration structure to boost raymarching speed)
         * SDF decoder
         * Optional: positional embedders for input position coords, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, TriplanarGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the SDF decoder.
                 - 'identity': The sample coordinates are fed as is into the SDF decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the SDF decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of SDF decoder.
            num_layers (int): Number of hidden layers in SDF decoder.
        """
        super().__init__()
        self.grid = grid
        self.pos_multires = pos_multires
        self.position_input = position_input
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires, position_input)
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder = self.init_decoder(activation_type, layer_type, num_layers, hidden_dim)
        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, position_input=True):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none' and not position_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or embedder_type == 'none' and position_input:
            embedder, embed_dim = torch.nn.Identity(), 3
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, position_input=position_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralSDF: {embedder_type}')
        return embedder, embed_dim

    def init_decoder(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder = BasicDecoder(input_dim=self.decoder_input_dim(), output_dim=1, activation=get_activation_class(activation_type), bias=True, layer=get_layer_class(layer_type), num_layers=num_layers, hidden_dim=hidden_dim, skip=[])
        return decoder

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.sdf, ['sdf'])

    def sdf(self, coords, lod_idx=None):
        """Computes the Signed Distance Function for input samples.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            (torch.FloatTensor):
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        if shape[0] == 0:
            return dict(sdf=torch.zeros_like(coords)[..., 0:1])
        if lod_idx is None:
            lod_idx = self.grid.num_lods - 1
        if len(shape) == 2:
            coords = coords[:, None]
        num_samples = coords.shape[1]
        feats = self.grid.interpolate(coords, lod_idx)
        if self.pos_embedder is not None:
            feats = torch.cat([self.pos_embedder(coords.view(-1, 3)).view(-1, num_samples, self.pos_embed_dim), feats], dim=-1)
        sdf = self.decoder(feats)
        if len(shape) == 2:
            sdf = sdf[:, 0]
        return dict(sdf=sdf)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def decoder_input_dim(self):
        input_dim = self.effective_feature_dim()
        if self.position_input:
            input_dim += self.pos_embed_dim
        return input_dim

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {'Grid': self.grid, 'Pos. Embedding': self.pos_embedder, 'Decoder (sdf)': self.decoder}
        return properties


class NeuralSDFTex(BaseNeuralField):
    """Model for encoding neural signed distance functions + plenoptic color, e.g., implicit surfaces with albedo.
    """

    def __init__(self, grid: 'BLASGrid'=None, embedder_type: 'str'='none', pos_multires: 'int'=10, activation_type: 'str'='relu', layer_type: 'str'='none', hidden_dim: 'int'=128, num_layers: 'int'=1):
        super().__init__()
        self.grid = grid
        self.embedder_type = embedder_type
        self.pos_multires = pos_multires
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(embedder_type, pos_multires)
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.position_input = embedder_type != 'none'
        self.decoder, self.effective_feature_dim, self.input_dim = self.init_decoder(activation_type, layer_type, num_layers, hidden_dim, self.position_input, self.pos_embed_dim)
        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, pos_multires):
        """Creates positional embedding functions for the position and view direction.
        """
        is_active = embedder_type == 'positional'
        pos_embedder, pos_embed_dim = get_positional_embedder(frequencies=pos_multires, active=is_active)
        log.info(f'Position Embed Dim: {pos_embed_dim}')
        return pos_embedder, pos_embed_dim

    def init_decoder(self, activation_type, layer_type, num_layers, hidden_dim, position_input, pos_embed_dim):
        """Initializes the decoder object.
        """
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        input_dim = effective_feature_dim
        if position_input:
            input_dim += pos_embed_dim
        decoder = BasicDecoder(input_dim=input_dim, output_dim=4, activation=get_activation_class(activation_type), bias=True, layer=get_layer_class(layer_type), num_layers=num_layers, hidden_dim=hidden_dim, skip=[])
        return decoder, effective_feature_dim, input_dim

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgbsdf, ['rgb', 'sdf'])

    def rgbsdf(self, coords, lod_idx=None):
        """Computes the RGB + SDF for some samples.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            {"rgb": torch.FloatTensor, "sdf": torch.FloatTensor}:
            - RGB of shape [batch, num_samples, 3]
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        if shape[0] == 0:
            return dict(rgb=torch.zeros_like(coords)[..., :3], sdf=torch.zeros_like(coords)[..., 0:1])
        if lod_idx is None:
            lod_idx = self.num_lods - 1
        if len(shape) == 2:
            coords = coords[:, None]
        feats = self.grid.interpolate(coords, lod_idx)
        if self.position_input:
            feats = torch.cat([self.pos_embedder(coords), feats], dim=-1)
        rgbsdf = self.decoder(feats)
        if len(shape) == 2:
            rgbsdf = rgbsdf[:, 0]
        return dict(rgb=torch.sigmoid(rgbsdf[..., :3]), sdf=rgbsdf[..., 3:4])


class SPCField(BaseNeuralField):
    """ A field based on Structured Point Clouds (SPC) from kaolin.
    SPC is a hierarchical compressed data structure, which can be interpreted in various ways:
    * Quantized point cloud, where each sparse point is quantized to some (possibly very dense) grid.
      Each point is associated with some feature(s).
    * An Octree, where each cell center is represented by a quantized point.
    Throughout wisp, SPCs are used to implement efficient octrees or grid structures.
    This field class allows wisp to render SPCs directly with their feature content (hence no embedders or decoders
    are assumed).

    When rendered, SPCs behave like octrees which allow for efficient tracing.
    Feature samples per ray may be collected from each intersected "cell" of the structured point cloud.
    """

    def __init__(self, spc_octree: 'torch.ByteTensor', features_dict: 'Dict[str, torch.tensor]'=None, device: 'torch.device'='cuda'):
        """Creates a new Structured Point Cloud (SPC), represented as a Wisp Field.

        In wisp, SPCs are considered neural fields, since their features may be optimized.
        See `examples/spc_browser` for an elaborate description of SPCs.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
            features_dict (dict):
                A dictionary holding the features information of the SPC.
                Keys are assumed to be a subset of ('colors', 'normals').
                Values are torch feature tensors containing information per point, of shape
                :math:`(\\text{num_points}, \\text{feat_dim})`.
                Where `num_points` is the number of occupied cells in the SPC.
                See `kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc` for conversion of point
                cloud information to such features.
            device (torch.device):
                Torch device on which the features and topology of the SPC field will be stored.
        """
        super().__init__()
        self.spc_octree = spc_octree
        self.features_dict = features_dict if features_dict is not None else dict()
        self.spc_device = device
        self.grid = None
        self.colors = None
        self.normals = None
        self.init_grid(spc_octree)

    def init_grid(self, spc_octree: 'torch.ByteTensor'):
        """ Uses the OctreeAS / OctreeGrid mechanism to quickly parse the SPC object into a Wisp Neural Field.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
        """
        spc_features = self.features_dict
        if 'colors' in self.features_dict:
            colors = spc_features['colors']
            colors = colors.reshape(-1, 4) / 255.0
            self.colors = colors
        if 'normals' in self.features_dict:
            normals = spc_features['normals']
            normals = normals.reshape(-1, 3)
            self.normals = normals
        if self.colors is None:
            if self.normals is not None:
                colors = 0.5 * (normals + 1.0)
            else:
                lengths = torch.tensor([len(spc_octree)], dtype=torch.int32)
                level, pyramids, exsum = kaolin_ops_spc.scan_octrees(spc_octree, lengths)
                point_hierarchies = kaolin_ops_spc.generate_points(spc_octree, pyramids, exsum)
                colors = point_hierarchies[pyramids[0, 1, level]:]
                colors = colors / np.power(2, level)
            self.colors = colors
        _, pyramid, _ = wisp_spc_ops.octree_to_spc(spc_octree)
        self.grid = OctreeGrid(blas=OctreeAS(spc_octree), feature_dim=3, num_lods=0)

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.

        Returns:
            (torch.device): The expected device used for this Structured Point Cloud.
        """
        return self.spc_device

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ['rgb'])

    def rgba(self, ridx_hit=None):
        """Compute color for the provided ray hits.

        Args:
            ridx_hit (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     used to indicate index of first hit voxel.

        Returns:
            {"rgb": torch.FloatTensor}:
                - RGB tensor of shape [batch, 1, 3]
        """
        level = self.grid.blas.max_level
        offset = self.grid.blas.pyramid[1, level]
        ridx_hit = ridx_hit - offset
        colors = self.colors[ridx_hit, :3].unsqueeze(1)
        return dict(rgb=colors)

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {'Grid': self.grid}
        return properties


class Pipeline(nn.Module):
    """Base class for implementing neural field pipelines.

    Pipelines consist of several components:

        - Neural fields (``self.nef``) which take coordinates as input and outputs signals.
          These usually consist of several optional components:

            - A feature grid (``self.nef.grid``)
              Sometimes also known as 'hybrid representations'.
            - An acceleration structure (``self.nef.grid.blas``) which can be used to accelerate spatial queries.
            - A decoder (``self.net.decoder``) which can take the features (or coordinates, or embeddings) and covert it to signals.

        - A forward map (``self.tracer``) which is a function which will invoke the pipeline in
          some outer loop. Usually this consists of renderers which will output a RenderBuffer object.
    
    The 'Pipeline' classes are responsible for holding and orchestrating these components.
    """

    def __init__(self, nef: 'BaseNeuralField', tracer: 'BaseTracer'=None):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__()
        self.nef: 'BaseNeuralField' = nef
        self.tracer: 'BaseTracer' = tracer

    def forward(self, *args, **kwargs):
        """The forward function will use the tracer (the forward model) if one is available. 
        
        Otherwise, it'll execute the neural field.
        """
        if self.tracer is not None:
            return self.tracer(self.nef, *args, **kwargs)
        else:
            return self.nef(*args, **kwargs)


class RasterizationPipeline(nn.Module):
    """ Wrapper class for implementing neural / non-neural Rasterization pipelines.
    RasterizationPipeline is a thin wrapper around existing rasterizers, which simply hints wisp
    the wrapped object is a rasterizer which relies on camera input, rather than rays.
    """

    def __init__(self, rasterizer):
        """Initialize the Pipeline.

        Args:
            rasterizer: A general model of a rasterizer.
                No assumptions are made on the rasterizer object. The only requirement is
                for this object to be callable.
                Rasterizers are encouraged to return a Renderbuffer object, but are not required to do so.
        """
        super().__init__()
        self.rasterizer = rasterizer

    def forward(self, *args, **kwargs):
        """The forward function will invoke the underlying rasterizer (the forward model).
        Rasterizer is any general callable interface.
        """
        return self.rasterizer(*args, **kwargs)


PI = torch.pi if hasattr(torch, 'pi') else np.pi


__RB_VARIANTS__ = dict()


def normalize(V: 'torch.Tensor', F: 'torch.Tensor', mode: 'str'):
    """Normalizes a mesh.

    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
        mode (str): Different methods of normalization.

    Returns:
        (torch.FloatTensor, torch.LongTensor):
        - Normalized Vertices
        - Faces
    """
    if mode == 'sphere':
        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.0
        V = V - V_center
        max_dist = torch.sqrt(torch.max(torch.sum(V ** 2, dim=-1)))
        V_scale = 1.0 / max_dist
        V *= V_scale
        return V, F
    elif mode == 'aabb':
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min
        max_dist = torch.max(V)
        V *= 1.0 / max_dist
        V = V * 2.0 - 1.0
        return V, F
    elif mode == 'planar':
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min
        x_max = torch.max(V[..., 0])
        z_max = torch.max(V[..., 2])
        V[..., 0] *= 1.0 / x_max
        V[..., 2] *= 1.0 / z_max
        max_dist = torch.max(V)
        V[..., 1] *= 1.0 / max_dist
        V = V * 2.0 - 1.0
        y_min = torch.min(V[..., 1])
        V[..., 1] -= y_min
        return V, F
    elif mode == 'none':
        return V, F


def blend_alpha_composite_over(c1: 'torch.Tensor', c2: 'torch.Tensor', alpha1: 'torch.Tensor', alpha2: 'torch.Tensor'):
    """ An alpha compositing op where a front pixel is alpha blended with the background pixel
    (in a usual painter's algorithm manner).
    Useful for blending channels such as RGB.
    See: https://en.wikipedia.org/wiki/Alpha_compositing

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): alpha channel tensor, corresponding to first channel, in the shape of c1.
        alpha2 (torch.Tensor): alpha channel tensor, corresponding to second channel, in the shape of c1.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    alpha_out = alpha1 + alpha2 * (1.0 - alpha1)
    c_out = torch.where(condition=alpha_out > 0, input=(c1 * alpha1 + c2 * alpha2 * (1.0 - alpha1)) / alpha_out, other=torch.zeros_like(c1))
    return c_out


soft_blue = 0.721, 0.9, 1.0


class BaseTracer(WispModule, ABC):
    """Base class for all tracers within Wisp.
    Tracers drive the mapping process which takes an input "Neural Field", and outputs a RenderBuffer of pixels.
    Different tracers may employ different algorithms for querying points, tracing and marching rays through the
    neural field.
    A common paradigm for tracers to employ is as follows:
    1. Take input in the form of rays
    2. Generate samples by tracing / marching rays, or querying coordinates over the neural field.
       Possibly make use of the neural field spatial structure for high performance.
    3. Invoke neural field's methods to decode sample features into actual channel values, such as color, density,
       signed distance, and so forth.
    4. Aggregate the sample values to decide on the final pixel value.
       The exact output may depend on the requested channel type, blending mode or other parameters.
    Wisp tracers are therefore flexible, and designed to be compatible with specific neural fields,
    depending on the forward functions they support and internal grid structures they use.
    Tracers are generally expected to be differentiable (e.g. they're part of the training loop),
    though non-differentiable tracers are also allowed.
    """

    def __init__(self, bg_color: 'Tuple[float, float, float]'=(0.0, 0.0, 0.0)):
        """Initializes the tracer class and sets the default arguments for trace.
        This should be overrided and called if you want to pass custom defaults into the renderer.
        If overridden, it should keep the arguments to `self.trace` in `self.` class variables.
        Then, if these variables exist and no function arguments are passed into forward,
        it will override them as the default.

        Args:
            bg_color (Tuple[float, float, float]): The clear background color used by default for the color channel.
        """
        super().__init__()
        self.bg_color = bg_color

    @abstractmethod
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Implement the function to return the supported channels, e.g.       
        return set(["depth", "rgb"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Implement the function to return the required channels, e.g.
        return set(["rgb", "density"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def trace(self, nef, rays, channels, extra_channels, *args, **kwargs):
        """Apply the forward map on the nef. 

        Tracers are required to implement this function, which commonly follows these paradigm:
        1. Take input in the form of rays
        2. Generate samples by tracing / marching rays, or querying coordinates over the neural field.
           Possibly make use of the neural field spatial structure for high performance.
        3. Invoke neural field's methods to decode sample features into actual channel values, such as color, density,
           signed distance, and so forth.
        4. Aggregate the sample values to decide on the final pixel value.
           The exact output may depend on the requested channel type, blending mode or other parameters.
        
        Args:
            nef (nn.Module): A neural field that uses a grid class.
            rays (Rays): Pack of rays to trace through the neural field.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): Requested extra channels, which are not first class channels supported by
                the tracer but will still be able to handle with some fallback options.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        pass

    def forward(self, nef, rays: 'Rays', channels=None, **kwargs):
        """Queries the tracer with channels.

        Args:
            nef (BaseNeuralField): Neural field to be traced. The nef will be queried for decoded sample values.
            rays (Rays): Pack of rays to trace through the neural field.
            channels (str or list of str or set of str): Requested channel names.
            This list should include at least all channels in tracer.get_supported_channels(),
            and may include extra channels in addition.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        nef_channels = nef.get_supported_channels()
        unsupported_inputs = self.get_required_nef_channels() - nef_channels
        if unsupported_inputs:
            raise Exception(f'The neural field class {type(nef)} does not output the required channels {unsupported_inputs}.')
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        extra_channels = requested_channels - self.get_supported_channels()
        unsupported_outputs = extra_channels - nef_channels
        if unsupported_outputs:
            raise Exception(f'Channels {unsupported_outputs} are not supported in the tracer {type(self)} or neural field {type(nef)}.')
        if extra_channels is None:
            requested_extra_channels = set()
        elif isinstance(extra_channels, str):
            requested_extra_channels = set([extra_channels])
        else:
            requested_extra_channels = set(extra_channels)
        required_args = dict(inspect.signature(BaseTracer.trace).parameters)
        required_args.pop('self', None)
        required_args.pop('args', None)
        required_args.pop('kwargs', None)
        optional_args = dict(inspect.signature(self.trace).parameters)
        optional_args.pop('self', None)
        optional_args.pop('args', None)
        optional_args.pop('kwargs', None)
        for _arg in required_args:
            optional_args.pop(_arg)
        input_args = {}
        for _arg in optional_args:
            if _arg in kwargs:
                input_args[_arg] = kwargs[_arg]
            else:
                default_arg = getattr(self, _arg, None)
                if default_arg is not None:
                    input_args[_arg] = default_arg
        with torch.cuda.nvtx.range('Tracer.trace'):
            rb = self.trace(nef, rays, requested_channels, requested_extra_channels, **input_args)
        return rb

    def public_properties(self) ->Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return dict()


gold = 1.0, 0.804, 0.0


lime_green = 0.519, 0.819, 0.0


purple = 0.667, 0.0, 0.429


soft_red = 1.0, 0.0, 0.085


_REGISTERED_RENDERABLE_NEURAL_FIELDS = defaultdict(dict)


def register_neural_field_type(neural_field_type: 'Type[BaseNeuralField]', tracer_type: 'Type[BaseTracer]', renderer_type: 'Type[BottomLevelRenderer]'):
    """ Register new types of neural fields with their associated bottom level renderers using this function.
        This allows the interactive renderer to display this neural field type on the canvas.
    """
    field_name = neural_field_type.__name__
    tracer_name = tracer_type.__name__
    _REGISTERED_RENDERABLE_NEURAL_FIELDS[field_name][tracer_name] = renderer_type


def field_renderer(field_type: 'Type[BaseNeuralField]', tracer_type: 'Type[BaseTracer]'):
    """ A decorator that registers a neural field type with a renderer.
        By registering the renderer type, the interactive renderer knows what type of renderer to create
        when dealing with this type of field.
        Essentially, this allows displaying custom types of objects on the canvas.
    """

    def _register_renderer_fn(renderer_class: 'Type[BottomLevelRenderer]'):
        register_neural_field_type(field_type, tracer_type, renderer_class)
        return renderer_class
    return _register_renderer_fn


class PackedRFTracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is differentiable, and can be employed within training loops.

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """

    def __init__(self, raymarch_type: 'str'='ray', num_steps: 'int'=1024, step_size: 'float'=1.0, bg_color: 'Tuple[float, float, float]'=(1.0, 1.0, 1.0)):
        """Set the default trace() arguments.

        Args:
            raymarch_type (str): Sample generation strategy to use for raymarch.
                'voxel' - intersects the rays with the acceleration structure cells.
                    Then among the intersected cells, each cell is sampled `num_steps` times.
                'ray' - samples `num_steps` along each ray, and then filters out samples which falls outside of occupied
                    cells of the acceleration structure.
            num_steps (int): The number of steps to use for the sampling. The meaning of this parameter changes
                depending on `raymarch_type`:
                'voxel' - each acceleration structure cell which intersects a ray is sampled `num_steps` times.
                'ray' - number of samples generated per ray, before culling away samples which don't fall
                    within occupied cells.
                The exact number of samples generated, therefore, depends on this parameter but also the occupancy
                status of the acceleration structure.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (Tuple[float, float, float]): The background color to use.
        """
        super().__init__(bg_color=bg_color)
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)
        self.prev_num_samples = None

    def get_prev_num_samples(self):
        """Returns the number of ray samples that were executed.
        
        Returns None if the tracer has never ran.

        Returns:
            (int): The number of ray samples.
        """
        return self.prev_num_samples

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'rgb', 'density'}

    def trace(self, nef, rays, channels, extra_channels, lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white'):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (Tuple[float, float, float]): The background color to use.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        assert nef.grid is not None and 'this tracer requires a grid'
        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1
        raymarch_results = nef.grid.raymarch(rays, level=nef.grid.active_lods[lod_idx], num_samples=num_steps, raymarch_type=raymarch_type)
        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        depths = raymarch_results.depth_samples
        self.prev_num_samples = samples.shape[0]
        pack_info = raymarch_results.pack_info
        boundary = raymarch_results.boundary
        hit_ray_d = rays.dirs.index_select(0, ridx)
        num_samples = samples.shape[0]
        color, density = nef(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=['rgb', 'density'])
        density = density.reshape(num_samples, 1)
        extra_outputs = {}
        self.bg_color = self.bg_color
        if 'depth' in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else:
            depth = None
        rgb = torch.zeros(N, 3, device=rays.origins.device) + self.bg_color
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)
        ridx_hit = ridx[boundary]
        tau = density * deltas
        del density, deltas
        ray_colors, transmittance = spc_render.exponential_integration(color, tau, boundary, exclusive=True)
        if 'depth' in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(num_samples, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth
        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[..., 0] > 0.0
        rgb[ridx_hit] = self.bg_color * (1.0 - alpha) + ray_colors
        for channel in extra_channels:
            feats = nef(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=channel)
            num_channels = feats.shape[-1]
            ray_feats, transmittance = spc_render.exponential_integration(feats.view(num_samples, num_channels), tau, boundary, exclusive=True)
            composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats
        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, **extra_outputs)


def find_depth_bound(query, nug_depth, info, curr_idxes=None):
    """Associate query points to the closest depth bound in-order.
    
    TODO: Document the input.
    """
    if curr_idxes is None:
        curr_idxes = torch.nonzero(info).contiguous()
    return _C.render.find_depth_bound_cuda(query.contiguous(), curr_idxes.contiguous(), nug_depth.contiguous())


def finitediff_gradient(x, f, eps=0.005):
    """Compute 3D gradient using finite difference.

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
    eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
    eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)
    grad = torch.cat([f(x + eps_x) - f(x - eps_x), f(x + eps_y) - f(x - eps_y), f(x + eps_z) - f(x - eps_z)], dim=-1)
    grad = grad / (eps * 2.0)
    return grad


class PackedSDFTracer(BaseTracer):
    """Tracer class for sparse SDFs.

    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - SDF: Signed Distance Function
    PackedSDFTracer is non-differentiable, and follows the sphere-tracer implementation of
    Neural Geometric Level of Detail (Takikawa et al. 2021).

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """

    def __init__(self, num_steps=1024, step_size=0.8, min_dis=0.0003):
        """Set the default trace() arguments.
        Args:
            num_steps (int): Max number of steps used by Sphere Trace if query did not converge
            step_size (float): Scale factor for step size used to advance the Sphere Tracer.
        """
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.min_dis = min_dis

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'normal', 'xyz', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'sdf'}

    def trace(self, nef, rays, channels, extra_channels, lod_idx=None, num_steps=64, step_size=1.0, min_dis=0.0001):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            channels (set): The set of requested channels. The trace method can return channels that
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  query those extra channels at surface intersection points.
            lod_idx (int): LOD index to render at.
            num_steps (int): The number of steps to use for sphere tracing.
            step_size (float): The multiplier for the sphere tracing steps. 
                               Use a value <1.0 for conservative tracing.
            min_dis (float): The termination distance for sphere tracing.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        assert nef.grid is not None and 'this tracer requires a grid'
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1
        invres = 1.0
        raytrace_results = nef.grid.raytrace(rays, nef.grid.active_lods[lod_idx], with_exit=True)
        ridx = raytrace_results.ridx
        pidx = raytrace_results.pidx
        depth = raytrace_results.depth
        depth[..., 0:1] += 1e-05
        first_hit = spc_render.mark_pack_boundaries(ridx)
        curr_idxes = torch.nonzero(first_hit)[..., 0].int()
        first_ridx = ridx[first_hit].long()
        nug_o = rays.origins[first_ridx]
        nug_d = rays.dirs[first_ridx]
        mask = torch.ones([first_ridx.shape[0]], device=nug_o.device).bool()
        hit = torch.zeros_like(mask).bool()
        t = depth[first_hit][..., 0:1]
        x = torch.addcmul(nug_o, nug_d, t)
        dist = torch.zeros_like(t)
        curr_pidx = pidx[first_hit].long()
        with torch.no_grad():
            sdf = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels='sdf') * invres * step_size
            dist[mask] = sdf
            dist[~mask] = 20
            dist_prev = dist.clone()
            for i in range(num_steps):
                t += dist
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                hit = torch.where(mask, torch.abs(dist)[..., 0] < min_dis * invres, hit)
                hit |= torch.where(mask, torch.abs(dist + dist_prev)[..., 0] * 0.5 < min_dis * 5 * invres, hit)
                mask = torch.where(mask, (t < rays.dist_max)[..., 0], mask)
                mask &= ~hit
                if not mask.any():
                    break
                dist_prev = torch.where(mask.view(mask.shape[0], 1), dist, dist_prev)
                next_idxes = find_depth_bound(t, depth, first_hit, curr_idxes=curr_idxes)
                mask &= next_idxes != -1
                aabb_mask = next_idxes != curr_idxes
                curr_idxes = torch.where(mask, next_idxes, curr_idxes)
                t = torch.where((mask & aabb_mask).view(mask.shape[0], 1), depth[curr_idxes.long(), 0:1], t)
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                curr_pidx = torch.where(mask, pidx[curr_idxes.long()].long(), curr_pidx)
                if not mask.any():
                    break
                sdf = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels='sdf') * invres * step_size
                dist[mask] = sdf
        x_buffer = torch.zeros_like(rays.origins)
        depth_buffer = torch.zeros_like(rays.origins[..., 0:1])
        hit_buffer = torch.zeros_like(rays.origins[..., 0]).bool()
        normal_buffer = torch.zeros_like(rays.origins)
        rgb_buffer = torch.zeros(*rays.origins.shape[:-1], 3, device=rays.origins.device)
        alpha_buffer = torch.zeros(*rays.origins.shape[:-1], 1, device=rays.origins.device)
        hit_buffer[first_ridx] = hit
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=x[hit], lod_idx=lod_idx, channels=channel)
            extra_buffer = torch.zeros(*rays.origins.shape[:-1], feats.shape[-1], device=feats.device)
            extra_buffer[hit_buffer] = feats
        x_buffer[hit_buffer] = x[hit]
        depth_buffer[hit_buffer] = t[hit]
        if 'rgb' in channels or 'normal' in channels:
            grad = finitediff_gradient(x[hit], nef.get_forward_function('sdf'))
            normal_buffer[hit_buffer] = F.normalize(grad, p=2, dim=-1, eps=1e-05)
            rgb_buffer[..., :3] = (normal_buffer + 1.0) / 2.0
        alpha_buffer[hit_buffer] = 1.0
        return RenderBuffer(xyz=x_buffer, depth=depth_buffer, hit=hit_buffer, normal=normal_buffer, rgb=rgb_buffer, alpha=alpha_buffer, **extra_outputs)


class PackedSPCTracer(BaseTracer):
    """Tracer class for sparse point clouds (packed rays).
    The logic of this tracer is straightforward and does not involve any neural operations:
    rays are intersected against the SPC points (cell centers).
    Each ray returns the color of the intersected cell, if such exists.

    See: https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/examples/spc_browser
    See also: https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html#spc
    """

    def __init__(self):
        """Set the default trace() arguments. """
        super().__init__(bg_color=(0.0, 0.0, 0.0))

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.

        Returns:
            (set): Set of channel strings.
        """
        return {'rgb'}

    def trace(self, nef, rays, channels, extra_channels, lod_idx=None):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            channels (set): The set of requested channels. The trace method can return channels that
                            were not requested since those channels often had to be computed anyways.
            lod_idx (int): LOD index to render at.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = nef.grid.blas.max_level
        raytrace_results = nef.grid.blas.raytrace(rays, lod_idx, with_exit=False)
        ridx = raytrace_results.ridx
        pidx = raytrace_results.pidx
        depths = raytrace_results.depth
        first_hits_mask = spc_render.mark_pack_boundaries(ridx)
        first_hits_point = pidx[first_hits_mask]
        first_hits_ray = ridx[first_hits_mask]
        first_hits_depth = depths[first_hits_mask]
        color = nef(ridx_hit=first_hits_point.long(), channels='rgb')
        del ridx, pidx, rays
        ray_colors = color.squeeze(1)
        ray_depth = first_hits_depth
        depth = torch.zeros(N, 1, device=ray_depth.device)
        depth[first_hits_ray.long(), :] = ray_depth
        alpha = torch.ones([color.shape[0], 1], device=color.device)
        hit = torch.zeros(N, device=color.device).bool()
        rgb = torch.zeros(N, 3, device=color.device)
        out_alpha = torch.zeros(N, 1, device=color.device)
        color = alpha * ray_colors
        hit[first_hits_ray.long()] = alpha[..., 0] > 0.0
        rgb[first_hits_ray.long(), :3] = color
        out_alpha[first_hits_ray.long()] = alpha
        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicDecoder,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'activation': torch.nn.ReLU(), 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrobeniusLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FullSort,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L_1_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L_inf_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MinMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (Pipeline,
     lambda: ([], {'nef': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (PositionalEmbedder,
     lambda: ([], {'num_freq': 4, 'max_freq_log2': 4}),
     lambda: ([torch.rand([4, 16])], {})),
    (RasterizationPipeline,
     lambda: ([], {'rasterizer': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (SigDecoder,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'hidden_dim': 4, 'activation': torch.nn.ReLU(), 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

