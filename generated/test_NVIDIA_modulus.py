
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


import torch.nn as nn


from typing import Dict


from typing import Optional


import time


from torch.nn.parallel import DistributedDataParallel


import math


import random


import numpy as np


from torch.utils.tensorboard import SummaryWriter


import logging


from typing import Union


from torch import Tensor


from collections import defaultdict


from functools import partial


from typing import Mapping


from typing import Any


from math import ceil


from torch.nn import MSELoss


from torch.optim import Adam


from torch.optim import lr_scheduler


import matplotlib.pyplot as plt


from torch import FloatTensor


from torch import cat


from torch.utils.data import DataLoader


from typing import Tuple


from torch.utils.data.distributed import DistributedSampler


from itertools import chain


import torch.nn.functional as F


import scipy.io


from torch.utils.data import Dataset


import scipy.io as scio


from torch.optim import AdamW


import uuid


from typing import List


from typing import Literal


from torch.multiprocessing import set_start_method


from typing import Iterable


import itertools


from collections.abc import Callable


from collections.abc import Iterable


from collections.abc import Mapping


import pandas as pd


import matplotlib


import torch.distributed as dist


import matplotlib as mpl


import warnings


from typing import Callable


import logging.config


import torch.distributed


import torch.utils


import torch.utils.data


from matplotlib import animation


from matplotlib import tri as mtri


from matplotlib.patches import Rectangle


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.utils import data


import functools


from math import pi


from math import gamma


from math import sqrt


from collections import OrderedDict


from torch.cuda import amp


from torch import vmap


from sklearn.neighbors import NearestNeighbors


import torch.optim as optim


from abc import ABC


from abc import abstractmethod


import copy


import torch._dynamo


from torch.distributed import gather


import torchvision.utils as tvu


import torchvision.transforms as transforms


import re


import torch as th


import scipy


from scipy.spatial import cKDTree


from torch import nn


from typing import Sequence


from typing import Type


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.profiler import profile


from torch.profiler import record_function


from torch.profiler import ProfilerActivity


import torch.cuda.profiler as profiler


from torch.optim.lr_scheduler import SequentialLR


from torch.optim.lr_scheduler import LinearLR


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import LambdaLR


import torch.optim as torch_optimizers


from scipy.io import netcdf_file


from enum import Enum


import scipy.sparse


from torch.nn import functional as F


from typing import DefaultDict


import queue


from warnings import warn


from typing import NewType


from torch.optim.lr_scheduler import _LRScheduler


from numpy.fft import fft


from numpy.fft import fftfreq


import torch.fft


from torch.nn.functional import silu


from torch.utils.checkpoint import checkpoint


import enum


from torch.autograd.function import once_differentiable


import torch.onnx


from torch.autograd import Function


from torch.nn import Dropout


from torch.nn import LayerNorm


from torch.nn import Linear


from torch.nn import Module


from torch.nn import ModuleList


from torch.nn import MultiheadAttention


from collections.abc import Sequence


import inspect


import torch.utils.checkpoint as checkpoint


from torch.nn import Embedding


from torch.nn import ReLU


from logging import Logger


import types


from torch.nn.functional import interpolate


from torch import testing


import torch.onnx.utils


from torch.testing import assert_close


class AttentionBlock(nn.Module):
    """
    Attention block for the skip connections using LayerNorm instead of BatchNorm.

    Parameters:
    ----------
        F_g (int): Number of channels in the decoder's features (query).
        F_l (int): Number of channels in the encoder's features (key/value).
        F_int (int): Number of intermediate channels (reduction in feature maps before attention computation).

    Returns:
    -------
        torch.Tensor: The attended skip feature map.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), ReshapedLayerNorm(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), ReshapedLayerNorm(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), ReshapedLayerNorm(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """
    A decoder block that sequentially applies multiple transposed convolutional blocks, optionally concatenating features from the corresponding encoder.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        feature_map_channels (List[int]): List of the number of channels for each deconv block within this decoder.
        model_depth (int): Number of times the deconv operation should be repeated.
        num_conv_blocks (int): Number of deconvolutional blocks per depth level.
        conv_activation (Optional[str]): Type of activation to usein conv layers. Default is 'relu'.
        conv_transpose_activation (Optional[str]): Type of activation to use in deconv layers. Default is None.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(self, out_channels: 'int', feature_map_channels: 'List[int]', kernel_size: 'Union[int, tuple]'=3, stride: 'Union[int, tuple]'=1, model_depth: 'int'=3, num_conv_blocks: 'int'=2, conv_activation: 'Optional[str]'='relu', conv_transpose_activation: 'Optional[str]'=None, padding: 'int'=1, padding_mode: 'str'='zeros', normalization: 'Optional[str]'='groupnorm', normalization_args: 'Optional[dict]'=None):
        super().__init__()
        if len(feature_map_channels) != model_depth * num_conv_blocks + 1:
            raise ValueError('The length of feature_map_channels in the decoder block should be equal to model_depth * num_conv_blocks + 1')
        self.layers = nn.ModuleList()
        current_channels = feature_map_channels[0]
        feature_map_channels = feature_map_channels[1:]
        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                if i == 0:
                    self.layers.append(ConvTranspose(in_channels=current_channels, out_channels=current_channels, activation=conv_transpose_activation))
                    current_channels += feature_map_channels[depth * num_conv_blocks + i]
                self.layers.append(ConvBlock(in_channels=current_channels, out_channels=feature_map_channels[depth * num_conv_blocks + i], kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, activation=conv_activation, normalization=normalization, normalization_args=normalization_args))
                current_channels = feature_map_channels[depth * num_conv_blocks + i]
        self.layers.append(ConvBlock(in_channels=current_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, activation=None, normalization=None))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Pool3d(nn.Module):
    """
    A pooling block that applies a specified 3D pooling operation over an input signal.

    Parameters:
    ----------
        pooling_type (str): Type of pooling operation ('AvgPool3d', 'MaxPool3d', or custom types if supported).
        kernel_size (int, tuple): Size of the window to take a pool over.
        stride (int, tuple, None): Stride of the pooling operation. Default is None (same as kernel_size).
        padding (int, tuple): Implicit zero padding to be added on both sides of the input. Default is 0.
        dilation (int, tuple): Control the spacing between the kernel points; useful for dilated pooling. Default is 1.
        ceil_mode (bool): When True, will use ceil instead of floor to compute the output shape. Default is False.
        count_include_pad (bool): Only used for AvgPool3d. If True, will include the zero-padding in the averaging calculation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(self, pooling_type: 'str'='AvgPool3d', kernel_size: 'Union[int, tuple]'=2, stride: 'Optional[Union[int, tuple]]'=None, padding: 'Union[int, tuple]'=0, dilation: 'Union[int, tuple]'=1, ceil_mode: 'bool'=False, count_include_pad: 'bool'=True):
        super().__init__()
        if pooling_type not in ['AvgPool3d', 'MaxPool3d']:
            raise ValueError(f"Invalid pooling_type '{pooling_type}'. Please choose from ['AvgPool3d', 'MaxPool3d'] or implement additional types.")
        if pooling_type == 'AvgPool3d':
            self.pooling = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        elif pooling_type == 'MaxPool3d':
            self.pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.pooling(x)


class EncoderBlock(nn.Module):
    """
    An encoder block that sequentially applies multiple convolutional blocks followed by a pooling operation, aggregating features at multiple scales.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        feature_map_channels (List[int]): List of the number of channels for each conv block within this encoder.
        model_depth (int): Number of times the conv-pool operation should be repeated.
        num_conv_blocks (int): Number of convolutional blocks per depth level.
        activation (Optional[str]): Type of activation to use. Default is 'relu'.
        pooling_type (str): Type of pooling to use ('AvgPool3d', 'MaxPool3d').
        pool_size (int): Size of the window for the pooling operation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(self, in_channels: 'int', feature_map_channels: 'List[int]', kernel_size: 'Union[int, tuple]'=3, stride: 'Union[int, tuple]'=1, model_depth: 'int'=4, num_conv_blocks: 'int'=2, activation: 'Optional[str]'='relu', padding: 'int'=1, padding_mode: 'str'='zeros', pooling_type: 'str'='AvgPool3d', pool_size: 'int'=2, normalization: 'Optional[str]'='groupnorm', normalization_args: 'Optional[dict]'=None):
        super().__init__()
        if len(feature_map_channels) != model_depth * num_conv_blocks:
            raise ValueError('The length of feature_map_channels should be equal to model_depth * num_conv_blocks')
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                self.layers.append(ConvBlock(in_channels=current_channels, out_channels=feature_map_channels[depth * num_conv_blocks + i], kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, activation=activation, normalization=normalization, normalization_args=normalization_args))
                current_channels = feature_map_channels[depth * num_conv_blocks + i]
            if depth < model_depth - 1:
                self.layers.append(Pool3d(pooling_type=pooling_type, kernel_size=pool_size))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UNet(Module):
    """
    U-Net model, featuring an encoder-decoder architecture with skip connections.
    Default parameters are set to replicate the architecture here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output segmentation map.
        model_depth (int): Number of levels in the U-Net, not counting the bottleneck layer.
        feature_map_channels (List[int]): Number of channels for each conv block in the encoder and decoder.
        num_conv_blocks (int): Number of convolutional blocks per level in the encoder and decoder.
        conv_activation (Optional[str]): Type of activation to usein conv layers. Default is 'relu'.
        conv_transpose_activation (Optional[str]): Type of activation to use in deconv layers. Default is None.
        pooling_type (str): Type of pooling operation used in the encoder. Supports "AvgPool3d", "MaxPool3d".
        pool_size (int): Size of the window for the pooling operation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, tuple]'=3, stride: 'Union[int, tuple]'=1, model_depth: 'int'=5, feature_map_channels: 'List[int]'=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], num_conv_blocks: 'int'=2, conv_activation: 'Optional[str]'='relu', conv_transpose_activation: 'Optional[str]'=None, padding: 'int'=1, padding_mode: 'str'='zeros', pooling_type: 'str'='MaxPool3d', pool_size: 'int'=2, normalization: 'Optional[str]'='groupnorm', normalization_args: 'Optional[dict]'=None, use_attn_gate: 'bool'=False, attn_decoder_feature_maps=None, attn_feature_map_channels=None, attn_intermediate_channels=None, gradient_checkpointing: 'bool'=True):
        super().__init__(meta=MetaData())
        self.use_attn_gate = use_attn_gate
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = EncoderBlock(in_channels=in_channels, feature_map_channels=feature_map_channels, kernel_size=kernel_size, stride=stride, model_depth=model_depth, num_conv_blocks=num_conv_blocks, activation=conv_activation, padding=padding, padding_mode=padding_mode, pooling_type=pooling_type, pool_size=pool_size, normalization=normalization, normalization_args=normalization_args)
        decoder_feature_maps = feature_map_channels[::-1][1:]
        self.decoder = DecoderBlock(out_channels=out_channels, feature_map_channels=decoder_feature_maps, kernel_size=kernel_size, stride=stride, model_depth=model_depth - 1, num_conv_blocks=num_conv_blocks, conv_activation=conv_activation, conv_transpose_activation=conv_transpose_activation, padding=padding, padding_mode=padding_mode, normalization=normalization, normalization_args=normalization_args)
        if self.use_attn_gate:
            self.attention_blocks = nn.ModuleList([AttentionBlock(F_g=attn_decoder_feature_maps[i], F_l=attn_feature_map_channels[i], F_int=attn_intermediate_channels) for i in range(model_depth - 1)])

    def checkpointed_forward(self, layer, x):
        """Wrapper to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing:
            return checkpoint.checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        skip_features = []
        for layer in self.encoder.layers:
            if isinstance(layer, Pool3d):
                skip_features.append(x)
            x = self.checkpointed_forward(layer, x)
        skip_features = skip_features[::-1]
        concats = 0
        for layer in self.decoder.layers:
            if isinstance(layer, ConvTranspose):
                x = self.checkpointed_forward(layer, x)
                if self.use_attn_gate:
                    skip_att = self.attention_blocks[concats](x, skip_features[concats])
                    x = torch.cat([x, skip_att], dim=1)
                else:
                    x = torch.cat([x, skip_features[concats]], dim=1)
                concats += 1
            else:
                x = self.checkpointed_forward(layer, x)
        return x


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: 'Tensor', target: 'Tensor'):
        return (torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)).mean()


class TruncatedMSELoss(nn.Module):
    """Truncated MSR loss."""

    def __init__(self, reduction='mean', threshold=1.0):
        super().__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input, target):
        loss = (input - target) ** 2
        loss[loss > self.threshold] = self.threshold
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class R2Loss(nn.Module):
    """
    Compute the R^2 loss.
    """

    def __init__(self, epsilon: 'float'=1e-06):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_true: 'Tensor', y_pred: 'Tensor'):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return ss_res / (ss_tot + self.epsilon)


class FactorizedSpectralConv1d(nn.Module):
    """1D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', rank: 'float', factorization: 'str', fixed_rank_modes: 'bool', decomposition_kwargs: 'dict'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.reset_parameters()

    def compl_mul1d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum('bix,iox->box', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        bsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(bsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)


class FactorizedSpectralConv2d(nn.Module):
    """2D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', modes2: 'int', rank: 'float', factorization: 'str', fixed_rank_modes: 'bool', decomposition_kwargs: 'dict'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights2 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.reset_parameters()

    def compl_mul2d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum('bixy,ioxy->boxy', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)


class FactorizedSpectralConv3d(nn.Module):
    """3D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', modes2: 'int', modes3: 'int', rank: 'float', factorization: 'str', fixed_rank_modes: 'bool', decomposition_kwargs: 'dict'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights2 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights3 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights4 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.reset_parameters()

    def compl_mul3d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum('bixyz,ioxyz->boxyz', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)


class FactorizedSpectralConv4d(nn.Module):
    """4D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', modes2: 'int', modes3: 'int', modes4: 'int', rank: 'float', factorization: 'str', fixed_rank_modes: 'bool', decomposition_kwargs: 'dict'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights2 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights3 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights4 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights5 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights6 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights7 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.weights8 = tltorch.FactorizedTensor.new((in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, 2), rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **decomposition_kwargs)
        self.reset_parameters()

    def compl_mul4d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum('bixyzt,ioxyzt->boxyzt', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights4)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights5)
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights6)
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights7)
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*N(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)
        self.weights5.normal_(0, self.scale)
        self.weights6.normal_(0, self.scale)
        self.weights7.normal_(0, self.scale)
        self.weights8.normal_(0, self.scale)


class TFNO1DEncoder(nn.Module):
    """1D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True, rank: 'float'=1.0, factorization: 'str'='cp', fixed_rank_modes: 'List[int]'=None, decomposition_kwargs: 'dict'=dict()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(FactorizedSpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0], self.rank, self.factorization, self.fixed_rank_modes, self.decomposition_kwargs))
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


class TFNO2DEncoder(nn.Module):
    """2D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True, rank: 'float'=1.0, factorization: 'str'='cp', fixed_rank_modes: 'List[int]'=None, decomposition_kwargs: 'dict'=dict()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]
        self.pad = padding[:2]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct TFNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(FactorizedSpectralConv2d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], self.rank, self.factorization, self.fixed_rank_modes, self.decomposition_kwargs))
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if x.dim() != 4:
            raise ValueError('Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO')
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))


class TFNO3DEncoder(nn.Module):
    """3D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True, rank: 'float'=1.0, factorization: 'str'='cp', fixed_rank_modes: 'List[int]'=None, decomposition_kwargs: 'dict'=dict()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]
        self.pad = padding[:3]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(FactorizedSpectralConv3d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], num_fno_modes[2], self.rank, self.factorization, self.fixed_rank_modes, self.decomposition_kwargs))
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1], :self.ipad[2]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


class TFNO4DEncoder(nn.Module):
    """4D Spectral encoder for TFNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True, rank: 'float'=1.0, factorization: 'str'='cp', fixed_rank_modes: 'List[int]'=None, decomposition_kwargs: 'dict'=dict()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        if self.coord_features:
            self.in_channels = self.in_channels + 4
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        padding = padding + [0, 0, 0, 0]
        self.pad = padding[:4]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.ConvNdFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.ConvNdFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct TFNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(FactorizedSpectralConv4d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], num_fno_modes[2], num_fno_modes[3], self.rank, self.factorization, self.fixed_rank_modes, self.decomposition_kwargs))
            self.conv_layers.append(layers.ConvNdKernel1Layer(self.fno_width, self.fno_width))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[3], 0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1], :self.ipad[2], :self.ipad[3]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 4D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z, size_t = shape[0], shape[2], shape[3], shape[4], shape[5]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(grid_x, grid_y, grid_z, grid_t, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 5, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], shape[5], value.size(-1))
        return torch.permute(output, (0, 5, 1, 2, 3, 4))


class Identity(nn.Module):
    """Identity activation function

    Dummy function for removing activations from a model

    Example
    -------
    >>> idnt_func = modulus.models.layers.Identity()
    >>> input = torch.randn(2, 2)
    >>> output = idnt_func(input)
    >>> torch.allclose(input, output)
    True
    """

    def forward(self, x: 'Tensor') ->Tensor:
        return x


def weight_fact(w, mean=1.0, stddev=0.1):
    """
    Randomly factorize the weight matrix into a product of vectors and a matrix

    Parameters
    ----------
    w : torch.Tensor
    mean : float, optional, default=1.0, mean of the normal distribution to sample the random scale factor
    stddev: float, optional, default=0.1, standard deviation of the normal distribution to sample the random scale factor
    """
    g = torch.normal(mean, stddev, size=(w.shape[0], 1))
    g = torch.exp(g)
    v = w / g
    return g, v


class WeightFactLinear(nn.Module):
    """Weight Factorization Layer for 2D Tensors, more details in https://arxiv.org/abs/2210.01274

    Parameters
    ----------
    in_features : int
        Size of the input features
    out_features : int
        Size of the output features
    bias : bool, optional
        Apply the bias to the output of linear layer, by default True
    reparam : dict, optional
        Dictionary with the mean and standard deviation to reparametrize the weight matrix,
        by default {'mean': 1.0, 'stddev': 0.1}

    Example
    -------
    >>> wfact = modulus.models.layers.WeightFactLinear(2,4)
    >>> input = torch.rand(2,2)
    >>> output = wfact(input)
    >>> output.size()
    torch.Size([2, 4])
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, mean: 'float'=1.0, stddev=0.1) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = mean
        self.stddev = stddev
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Factorize weights and reset bias"""
        nn.init.xavier_uniform_(self.weight)
        g, v = weight_fact(self.weight.detach(), mean=self.mean, stddev=self.stddev)
        self.g = nn.Parameter(g)
        self.v = nn.Parameter(v)
        self.weight = None
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: 'Tensor') ->Tensor:
        weight = self.g * self.v
        return F.linear(input, weight, self.bias)

    def extra_repr(self) ->str:
        """Print information about weight factorization"""
        return 'in_features={}, out_features={}, bias={}, mean = {}, stddev = {}'.format(self.in_features, self.out_features, self.bias is not None, self.mean, self.stddev)


class WeightNormLinear(nn.Module):
    """Weight Norm Layer for 1D Tensors

    Parameters
    ----------
    in_features : int
        Size of the input features
    out_features : int
        Size of the output features
    bias : bool, optional
        Apply the bias to the output of linear layer, by default True

    Example
    -------
    >>> wnorm = modulus.models.layers.WeightNormLinear(2,4)
    >>> input = torch.rand(2,2)
    >>> output = wnorm(input)
    >>> output.size()
    torch.Size([2, 4])
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_g = nn.Parameter(torch.empty((out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset normalization weights"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: 'Tensor') ->Tensor:
        norm = self.weight.norm(dim=1, p=2, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return F.linear(input, weight, self.bias)

    def extra_repr(self) ->str:
        """Print information about weight norm"""
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class FCLayer(nn.Module):
    """Densely connected NN layer

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    weight_norm : bool, optional
        Applies weight normalization to the layer, by default False
    weight_fact : bool, optional
        Applies weight factorization to the layer, by default False
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, in_features: 'int', out_features: 'int', activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, weight_norm: 'bool'=False, weight_fact: 'bool'=False, activation_par: 'Union[nn.Parameter, None]'=None) ->None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = Identity()
        else:
            self.activation_fn = activation_fn
        self.weight_norm = weight_norm
        self.weight_fact = weight_fact
        self.activation_par = activation_par
        if weight_norm and weight_fact:
            raise ValueError('Cannot apply both weight normalization and weight factorization together, please select one.')
        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True)
        elif weight_fact:
            self.linear = WeightFactLinear(in_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=True)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset fully connected weights"""
        if not self.weight_norm and not self.weight_fact:
            nn.init.constant_(self.linear.bias, 0)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.linear(x)
        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)
        return x


class CappedGELU(torch.nn.Module):
    """
    Implements a GELU with capped maximum value.

    Example
    -------
    >>> capped_gelu_func = modulus.models.layers.CappedGELU()
    >>> input = torch.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_gelu_func(input)
    tensor([[-0.0455, -0.1587],
            [ 0.0000,  0.8413],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        **kwargs:
             Keyword arguments to be passed to the `torch.nn.GELU` function
        """
        super().__init__()
        self.add_module('gelu', torch.nn.GELU(**kwargs))
        self.register_buffer('cap', torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class CappedLeakyReLU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.

    Example
    -------
    >>> capped_leakyReLU_func = modulus.models.layers.CappedLeakyReLU()
    >>> input = torch.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_leakyReLU_func(input)
    tensor([[-0.0200, -0.0100],
            [ 0.0000,  1.0000],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        **kwargs:
             Keyword arguments to be passed to the `torch.nn.LeakyReLU` function
        """
        super().__init__()
        self.add_module('leaky_relu', torch.nn.LeakyReLU(**kwargs))
        self.register_buffer('cap', torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.leaky_relu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class SquarePlus(nn.Module):
    """Squareplus activation

    Note
    ----
    Reference: arXiv preprint arXiv:2112.11687

    Example
    -------
    >>> sqr_func = modulus.models.layers.SquarePlus()
    >>> input = torch.Tensor([[1,2],[3,4]])
    >>> sqr_func(input)
    tensor([[1.6180, 2.4142],
            [3.3028, 4.2361]])
    """

    def __init__(self):
        super().__init__()
        self.b = 4

    def forward(self, x: 'Tensor') ->Tensor:
        return 0.5 * (x + torch.sqrt(x * x + self.b))


class Stan(nn.Module):
    """Self-scalable Tanh (Stan) for 1D Tensors

    Parameters
    ----------
    out_features : int, optional
        Number of features, by default 1

    Note
    ----
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.

    Example
    -------
    >>> stan_func = modulus.models.layers.Stan(out_features=1)
    >>> input = torch.Tensor([[0],[1],[2]])
    >>> stan_func(input)
    tensor([[0.0000],
            [1.5232],
            [2.8921]], grad_fn=<MulBackward0>)
    """

    def __init__(self, out_features: 'int'=1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x: 'Tensor') ->Tensor:
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(f'The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}')
        return torch.tanh(x) * (1.0 + self.beta * x)


ACT2FN = {'relu': nn.ReLU, 'leaky_relu': (nn.LeakyReLU, {'negative_slope': 0.1}), 'prelu': nn.PReLU, 'relu6': nn.ReLU6, 'elu': nn.ELU, 'selu': nn.SELU, 'silu': nn.SiLU, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid, 'logsigmoid': nn.LogSigmoid, 'softplus': nn.Softplus, 'softshrink': nn.Softshrink, 'softsign': nn.Softsign, 'tanh': nn.Tanh, 'tanhshrink': nn.Tanhshrink, 'threshold': (nn.Threshold, {'threshold': 1.0, 'value': 1.0}), 'hardtanh': nn.Hardtanh, 'identity': Identity, 'stan': Stan, 'squareplus': SquarePlus, 'cappek_leaky_relu': CappedLeakyReLU, 'capped_gelu': CappedGELU}


def get_activation(activation: 'str') ->nn.Module:
    """Returns an activation function given a string

    Parameters
    ----------
    activation : str
        String identifier for the desired activation function

    Returns
    -------
    Activation function

    Raises
    ------
    KeyError
        If the specified activation function is not found in the dictionary
    """
    try:
        activation = activation.lower()
        module = ACT2FN[activation]
        if isinstance(module, tuple):
            return module[0](**module[1])
        else:
            return module()
    except KeyError:
        raise KeyError(f'Activation function {activation} not found. Available options are: {list(ACT2FN.keys())}')


class FullyConnected(Module):
    """A densely-connected MLP architecture

    Parameters
    ----------
    in_features : int, optional
        Size of input features, by default 512
    layer_size : int, optional
        Size of every hidden layer, by default 512
    out_features : int, optional
        Size of output features, by default 512
    num_layers : int, optional
        Number of hidden layers, by default 6
    activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'silu'
    skip_connections : bool, optional
        Add skip connections every 2 hidden layers, by default False
    adaptive_activations : bool, optional
        Use an adaptive activation function, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default False
    weight_fact : bool, optional
        Use weight factorization on fully connected layers, by default False

    Example
    -------
    >>> model = modulus.models.mlp.FullyConnected(in_features=32, out_features=64)
    >>> input = torch.randn(128, 32)
    >>> output = model(input)
    >>> output.size()
    torch.Size([128, 64])
    """

    def __init__(self, in_features: 'int'=512, layer_size: 'int'=512, out_features: 'int'=512, num_layers: 'int'=6, activation_fn: 'Union[str, List[str]]'='silu', skip_connections: 'bool'=False, adaptive_activations: 'bool'=False, weight_norm: 'bool'=False, weight_fact: 'bool'=False) ->None:
        super().__init__(meta=MetaData())
        self.skip_connections = skip_connections
        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None
        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * num_layers
        if len(activation_fn) < num_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (num_layers - len(activation_fn))
        activation_fn = [get_activation(a) for a in activation_fn]
        self.layers = nn.ModuleList()
        layer_in_features = in_features
        for i in range(num_layers):
            self.layers.append(FCLayer(layer_in_features, layer_size, activation_fn[i], weight_norm, weight_fact, activation_par))
            layer_in_features = layer_size
        self.final_layer = FCLayer(in_features=layer_size, out_features=out_features, activation_fn=None, weight_norm=False, weight_fact=False, activation_par=None)

    def forward(self, x: 'Tensor') ->Tensor:
        x_skip: 'Optional[Tensor]' = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x
        x = self.final_layer(x)
        return x


class TFNO(nn.Module):
    """Tensor Factorized Fourier neural operator (FNO) model.

    Note
    ----
    The TFNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    rank : float, optional
        Rank of the decomposition, by default 1.0
    factorization : {'CP', 'TT', 'Tucker'}, optional
        Tensor factorization to use to decompose the tensor, by default 'CP'
    fixed_rank_modes : List[int], optional
        A list of modes for which the initial value is not modified, by default None
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict, optional
        Additional arguments to initialization of factorized tensors, by default dict()

    Example
    -------
    >>> # define the 2d TFNO model
    >>> model = modulus.models.fno.TFNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Rosofsky, Shawn G. and Huerta, E. A. "Magnetohydrodynamics with
    Physics Informed Neural Operators." arXiv preprint arXiv:2302.08332 (2023).
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', decoder_layers: 'int'=1, decoder_layer_size: 'int'=32, decoder_activation_fn: 'str'='silu', dimension: 'int'=2, latent_channels: 'int'=32, num_fno_layers: 'int'=4, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'int'=8, padding_type: 'str'='constant', activation_fn: 'str'='gelu', coord_features: 'bool'=True, rank: 'float'=1.0, factorization: 'str'='cp', fixed_rank_modes: 'List[int]'=None, decomposition_kwargs: 'dict'=dict()) ->None:
        super().__init__()
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.decoder_net = FullyConnected(in_features=latent_channels, layer_size=decoder_layer_size, out_features=out_channels, num_layers=decoder_layers, activation_fn=decoder_activation_fn)
        TFNOModel = self.getTFNOEncoder()
        self.spec_encoder = TFNOModel(in_channels, num_fno_layers=self.num_fno_layers, fno_layer_size=latent_channels, num_fno_modes=self.num_fno_modes, padding=self.padding, padding_type=self.padding_type, activation_fn=self.activation_fn, coord_features=self.coord_features, rank=self.rank, factorization=self.factorization, fixed_rank_modes=self.fixed_rank_modes, decomposition_kwargs=self.decomposition_kwargs)

    def getTFNOEncoder(self):
        """Return correct TFNO ND Encoder"""
        if self.dimension == 1:
            return TFNO1DEncoder
        elif self.dimension == 2:
            return TFNO2DEncoder
        elif self.dimension == 3:
            return TFNO3DEncoder
        elif self.dimension == 4:
            return TFNO4DEncoder
        else:
            raise NotImplementedError('Invalid dimensionality. Only 1D, 2D, 3D and 4D FNO implemented')

    def forward(self, x: 'Tensor') ->Tensor:
        y_latent = self.spec_encoder(x)
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)
        y = self.decoder_net(y_latent)
        y = self.spec_encoder.points_to_grid(y, y_shape)
        return y


class DNN(torch.nn.Module):
    """
    Custom PyTorch model
    """

    def __init__(self, layers, fourier_features=64):
        super().__init__()
        self.depth = len(layers) - 1
        self.fourier_features = fourier_features
        self.register_buffer('B', 10 * torch.randn((layers[0], fourier_features)))
        self.activation = torch.nn.GELU
        layer_list = list()
        for i in range(1, self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.layers(x_proj)
        return out


class ShallowWaterSolver(nn.Module):
    """
    SWE solver class. Interface inspired bu pyspharm and SHTns
    """

    def __init__(self, nlat, nlon, dt, lmax=None, mmax=None, grid='legendre-gauss', radius=6371220.0, omega=7.292e-05, gravity=9.80616, havg=10000.0, hamp=120.0):
        super().__init__()
        self.dt = dt
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.register_buffer('radius', torch.as_tensor(radius, dtype=torch.float64))
        self.register_buffer('omega', torch.as_tensor(omega, dtype=torch.float64))
        self.register_buffer('gravity', torch.as_tensor(gravity, dtype=torch.float64))
        self.register_buffer('havg', torch.as_tensor(havg, dtype=torch.float64))
        self.register_buffer('hamp', torch.as_tensor(hamp, dtype=torch.float64))
        self.sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.vsht = harmonics.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False)
        self.lmax = lmax or self.sht.lmax
        self.mmax = lmax or self.sht.mmax
        if self.grid == 'legendre-gauss':
            cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(self.nlat, -1, 1)
        elif self.grid == 'lobatto':
            cost, quad_weights = harmonics.quadrature.lobatto_weights(self.nlat, -1, 1)
        elif self.grid == 'equiangular':
            cost, quad_weights = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)
        quad_weights = torch.as_tensor(quad_weights).reshape(-1, 1)
        lats = -torch.as_tensor(np.arcsin(cost))
        lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[:nlon]
        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax
        l = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l = l.expand(self.lmax, self.mmax)
        lap = -l * (l + 1) / self.radius ** 2
        invlap = -self.radius ** 2 / l / (l + 1)
        invlap[0] = 0.0
        coriolis = 2 * self.omega * torch.sin(lats).reshape(self.nlat, 1)
        hyperdiff = torch.exp(torch.asarray(-self.dt / 2 / 3600.0 * (lap / lap[-1, 0]) ** 4))
        self.register_buffer('lats', lats)
        self.register_buffer('lons', lons)
        self.register_buffer('l', l)
        self.register_buffer('lap', lap)
        self.register_buffer('invlap', invlap)
        self.register_buffer('coriolis', coriolis)
        self.register_buffer('hyperdiff', hyperdiff)
        self.register_buffer('quad_weights', quad_weights)

    def grid2spec(self, ugrid):
        """
        spectral coefficients from spatial data
        """
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)

    def vrtdivspec(self, ugrid):
        """spatial data from spectral coefficients"""
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht(self.invlap * vrtdivspec / self.radius)

    def gethuv(self, uspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        hgrid = self.spec2grid(uspec[:1])
        uvgrid = self.getuv(uspec[1:])
        return torch.cat((hgrid, uvgrid), dim=-3)

    def potential_vorticity(self, uspec):
        """
        Compute potential vorticity
        """
        ugrid = self.spec2grid(uspec)
        pvrt = 0.5 * self.havg * self.gravity / self.omega * (ugrid[1] + self.coriolis) / ugrid[0]
        return pvrt

    def dimensionless(self, uspec):
        """
        Remove dimensions from variables
        """
        uspec[0] = (uspec[0] - self.havg * self.gravity) / self.hamp / self.gravity
        uspec[1:] = uspec[1:] * self.radius / torch.sqrt(self.gravity * self.havg)
        return uspec

    def dudtspec(self, uspec):
        """
        Compute time derivatives from solution represented in spectral coefficients
        """
        dudtspec = torch.zeros_like(uspec)
        ugrid = self.spec2grid(uspec)
        uvgrid = self.getuv(uspec[1:])
        tmp = uvgrid * (ugrid[1] + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        dudtspec[2] = tmpspec[0]
        dudtspec[1] = -1 * tmpspec[1]
        tmp = uvgrid * ugrid[0]
        tmp = self.vrtdivspec(tmp)
        dudtspec[0] = -1 * tmp[1]
        tmpspec = self.grid2spec(ugrid[0] + 0.5 * (uvgrid[0] ** 2 + uvgrid[1] ** 2))
        dudtspec[2] = dudtspec[2] - self.lap * tmpspec
        return dudtspec

    def galewsky_initial_condition(self):
        """
        Initializes non-linear barotropically unstable shallow water test case of Galewsky et al. (2004, Tellus, 56A, 429-440).

        [1] Galewsky; An initial-value problem for testing numerical models of the global shallow-water equations;
            DOI: 10.1111/j.1600-0870.2004.00071.x; http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
        """
        device = self.lap.device
        umax = 80.0
        phi0 = torch.asarray(torch.pi / 7.0, device=device)
        phi1 = torch.asarray(0.5 * torch.pi - phi0, device=device)
        phi2 = 0.25 * torch.pi
        en = torch.exp(torch.asarray(-4.0 / (phi1 - phi0) ** 2, device=device))
        alpha = 1.0 / 3.0
        beta = 1.0 / 15.0
        lats, lons = torch.meshgrid(self.lats, self.lons)
        u1 = umax / en * torch.exp(1.0 / ((lats - phi0) * (lats - phi1)))
        ugrid = torch.where(torch.logical_and(lats < phi1, lats > phi0), u1, torch.zeros(self.nlat, self.nlon, device=device))
        vgrid = torch.zeros((self.nlat, self.nlon), device=device)
        hbump = self.hamp * torch.cos(lats) * torch.exp(-((lons - torch.pi) / alpha) ** 2) * torch.exp(-(phi2 - lats) ** 2 / beta)
        ugrid = torch.stack((ugrid, vgrid))
        vrtdivspec = self.vrtdivspec(ugrid)
        vrtdivgrid = self.spec2grid(vrtdivspec)
        tmp = ugrid * (vrtdivgrid + self.coriolis)
        tmpspec = self.vrtdivspec(tmp)
        tmpspec[1] = self.grid2spec(0.5 * torch.sum(ugrid ** 2, dim=0))
        phispec = self.invlap * tmpspec[0] - tmpspec[1] + self.grid2spec(self.gravity * (self.havg + hbump))
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=vrtdivspec.dtype, device=device)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec
        return torch.tril(uspec)

    def random_initial_condition(self, mach=0.1) ->torch.Tensor:
        """
        random initial condition on the sphere
        """
        device = self.lap.device
        ctype = torch.complex128 if self.lap.dtype == torch.float64 else torch.complex64
        llimit = mlimit = 80
        uspec = torch.zeros(3, self.lmax, self.mmax, dtype=ctype, device=self.lap.device)
        uspec[:, :llimit, :mlimit] = torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit + 1), device=device, dtype=ctype)) * torch.randn_like(uspec[:, :llimit, :mlimit])
        uspec[0] = self.gravity * self.hamp * uspec[0]
        uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * self.havg * self.gravity
        uspec[1:] = mach * uspec[1:] * torch.sqrt(self.gravity * self.havg) / self.radius
        return torch.tril(uspec)

    def timestep(self, uspec: 'torch.Tensor', nsteps: 'int') ->torch.Tensor:
        """
        Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps.
        """
        dudtspec = torch.zeros(3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device)
        inew = 0
        inow = 1
        iold = 2
        for iter in range(nsteps):
            dudtspec[inew] = self.dudtspec(uspec)
            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]
            uspec = uspec + self.dt * (23.0 / 12.0 * dudtspec[inew] - 16.0 / 12.0 * dudtspec[inow] + 5.0 / 12.0 * dudtspec[iold])
            uspec[1:] = self.hyperdiff * uspec[1:]
            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3
        return uspec

    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius ** 2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius ** 2, dim=(-2, -1))
        return out

    def plot_griddata(self, data, fig, cmap='twilight_shifted', vmax=None, vmin=None, projection='3d', title=None, antialiased=False):
        """
        plotting routine for data on the grid. Requires cartopy for 3d plots.
        """
        import matplotlib.pyplot as plt
        lons = self.lons.squeeze() - torch.pi
        lats = self.lats.squeeze()
        if data.is_cuda:
            data = data.cpu()
            lons = lons.cpu()
            lats = lats.cpu()
        Lons, Lats = np.meshgrid(lons, lats)
        if projection == 'mollweide':
            ax = fig.add_subplot(projection=projection)
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.colorbar(im, orientation='horizontal')
            plt.title(title)
        elif projection == '3d':
            proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)
            ax = fig.add_subplot(projection=proj)
            Lons = Lons * 180 / np.pi
            Lats = Lats * 180 / np.pi
            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
            plt.title(title, y=1.05)
        else:
            raise NotImplementedError
        return im

    def plot_specdata(self, data, fig, **kwargs):
        return self.plot_griddata(self.isht(data), fig, **kwargs)


def _correct_lat_at_pole(lat, dlat):
    """Adjust latitude at the poles to avoid zero weight at pole."""
    correction = dlat / 4
    if lat == 90:
        lat -= correction
    elif lat == -90:
        lat += correction
    return lat


class GeometricL2Loss(nn.Module):
    """L2 loss on a lat-lon grid where the loss is computed over the sphere
    i.e. the errors are weighted by cos(lat).
    """

    def __init__(self, lat_range: 'Tuple[int, int]'=(-90, 90), num_lats: 'int'=721, lat_indices_used: 'Tuple[int, int]'=(0, 720), input_dims: 'int'=4):
        super().__init__()
        lats = torch.linspace(lat_range[0], lat_range[1], num_lats)
        dlat = lats[1] - lats[0]
        lats[0] = _correct_lat_at_pole(lats[0], dlat)
        lats[-1] = _correct_lat_at_pole(lats[-1], dlat)
        lats = torch.deg2rad(lats[lat_indices_used[0]:lat_indices_used[1]])
        weights = torch.cos(lats)
        weights = weights / torch.sum(weights)
        weights = torch.reshape(weights, (1,) * (input_dims - 2) + (lat_indices_used[1] - lat_indices_used[0], 1))
        self.register_buffer('weights', weights)

    def forward(self, pred: 'Tensor', true: 'Tensor') ->Tensor:
        err = torch.square(pred - true)
        err = torch.sum(err * self.weights, dim=-2)
        return torch.mean(err)


class BaseMSE(th.nn.MSELoss):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """

    def __init__(self):
        """Constructer for BaseMSE"""
        super().__init__()
        self.device = None

    def setup(self, trainer):
        """
        Nothing to implement here
        """
        pass

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the base MSE class
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError('Expected predictions to have 6 dimensions')
        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.mean(d)
        else:
            return d


class WeightedMSE(th.nn.MSELoss):
    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(self, weights: 'Sequence'=[]):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device
        """
        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError('Length of outputs and loss_weights is not the same!')
        self.loss_weights = self.loss_weights

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedMSE pass
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError('Expected predictions to have 6 dimensions')
        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5)) * self.loss_weights
        if average_channels:
            return th.mean(d)
        else:
            return d


class OceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss weighted by a land-sea-mask field.
    """

    def __init__(self, lsm_file: 'str', open_dict: 'dict'={'engine': 'zarr'}, selection_dict: 'dict'={'channel_c': 'lsm'}):
        """
        Parameters
        ----------
        lsm_file: str
            land-sea-mask file
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None

    def setup(self, trainer):
        """
        reshape lsm and put on device
        """
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(self.selection_dict)
        self.lsm_tensor = 1 - th.tensor(np.expand_dims(self.lsm_ds.values, (0, 2, 3)))

    def forward(self, prediction, target, average_channels=True):
        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(dim=(0, 1, 2, 4, 5))
            self.lsm_sum_calculated = True
        ocean_err = (target - prediction) ** 2 * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum


class SSIM(torch.nn.Module):
    """
    This class provides a differential structural similarity (SSIM) as loss for training an artificial neural network. The
    advantage of SSIM over the conventional mean squared error is a relation to images where SSIM incorporates the local
    neighborhood when determining the quality of an individual pixel. Results are less blurry, as demonstrated here
    https://ece.uwaterloo.ca/~z70wang/research/ssim/

    Code is origininally taken from https://github.com/Po-Hsun-Su/pytorch-ssim
    Modifications include comments and an optional training phase with the mean squared error (MSE) preceding the SSIM
    loss, to bring the weights on track. Otherwise, SSIM often gets stuck early in a local minimum.
    """

    def __init__(self, window_size: 'int'=11, time_series_forecasting: 'bool'=False, padding_mode: 'str'='constant', mse: 'bool'=False, mse_epochs: 'int'=0):
        """
        Constructor method.

        param window_size: int, optional
            The patch size over which the SSIM is computed
        param time_series_forecasting: bool ,optional
            Boolean indicating whether time series forecasting is the task
        param padding_mode: str
            Padding mode used for padding input images, e.g. 'zeros', 'replicate', 'reflection'
        param mse: torch.nn.Module
            Uses MSE parallel
        param mse_epochs: int, optional
            Number of MSE epochs preceding the SSIM epochs during training
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.time_series_forecasting = time_series_forecasting
        self.padding_mode = padding_mode
        self.mse = torch.nn.MSELoss() if mse else None
        self.mse_epochs = mse_epochs
        self.c1, self.c2 = 0.01 ** 2, 0.03 ** 2
        self.register_buffer('window', self._create_window(window_size), persistent=False)

    def forward(self, predicted: 'torch.Tensor', target: 'torch.Tensor', mask: 'torch.Tensor'=None, epoch: 'int'=0) ->torch.Tensor:
        """
        Forward pass of the SSIM loss

        param predicted: torch.Tensor
            Predicted image of shape
            [B, T, C, F, H, W] with time series forcasting
            [B, C, F, H, W] without time series forcasting
        param target: torch.Tensor
            Ground truth image of shape
            [B, T, C, F, H, W] with time series forcasting
            [B, C, F, H, W] without time series forcasting
        param mask: torch.Tensor, optional
            Mask for excluding pixels
        param epoch: int, optional
            The current epoch

        Returns
        -------
        torch.Tensor
            The structural similarity loss
        """
        if predicted.ndim != target.ndim:
            raise AssertionError('Predicted and target tensor need to have the same number of dimensions')
        if not self.time_series_forecasting and not (predicted.ndim == 4 or predicted.ndim == 5):
            raise AssertionError('Need 4 or 5 dimensions when not using time series forecasting')
        if self.time_series_forecasting and not (predicted.ndim == 5 or predicted.ndim == 6):
            raise AssertionError('Need 5 or 6 dimensions when using time series forecasting')
        predicted = predicted.transpose(dim0=2, dim1=3)
        target = target.transpose(dim0=2, dim1=3)
        if self.time_series_forecasting:
            predicted = torch.flatten(predicted, start_dim=0, end_dim=2)
            target = torch.flatten(target, start_dim=0, end_dim=2)
        window = self.window.expand(predicted.shape[1], -1, -1, -1)
        window = window
        return self._ssim(predicted, target, window, mask, epoch)

    @staticmethod
    def _gaussian(window_size: 'int', sigma: 'float') ->torch.Tensor:
        """
        Computes a Gaussian over the size of the window to weigh distant pixels less.

        Parameters
        ----------
        window_size: int
            The size of the patches
        sigma: float
            The width of the Gaussian curve

        Returns
        -------
        torch.Tensor: A tensor representing the weights for each pixel in the window or patch
        """
        x = torch.arange(0, window_size) - window_size // 2
        gauss = torch.exp(-x.div(2 * sigma) ** 2)
        return gauss / gauss.sum()

    def _create_window(self, window_size: 'int', sigma: 'float'=1.5) ->torch.Tensor:
        """
        Creates the weights of the window or patches.

        Parameters
        ----------
        window_size: int
            The size of the patches
        sigma: float, optional default 1.5
            The width of the Gaussian curve

        Returns
        -------
            torch.Tensor: The weights of the window
        """
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(-1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window

    def _ssim(self, predicted: 'torch.Tensor', target: 'torch.Tensor', window: 'torch.Tensor', mask: 'torch.Tensor'=None, epoch: 'int'=0) ->torch.Tensor:
        """
        Computes the SSIM loss between two image tensors

        Parameters
        ----------
        _predicted: torch.Tensor
            The predicted image tensor
        _target: torch.Tensor
            The target image tensor
        window: torch.Tensor
            The weights for each pixel in the window over which the SSIM is computed
        mask: torch.Tensor, optional default None
            Mask for excluding pixels
        epoch: int, optional default 0
            The current epoch

        Returns
        -------
        torch.Tensor The SSIM between predicted and target
        """
        if epoch < self.mse_epochs:
            return F.mse_loss(predicted, target)
        channels = window.shape[0]
        window_size = window.shape[2]
        window = window
        _predicted = F.pad(predicted, pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2, (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2], mode=self.padding_mode)
        _target = F.pad(target, pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2, (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2], mode=self.padding_mode)
        mu1 = F.conv2d(_predicted, window, padding=0, groups=channels)
        mu2 = F.conv2d(_target, window, padding=0, groups=channels)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(_predicted * _predicted, window, padding=0, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(_target * _target, window, padding=0, groups=channels) - mu2_sq
        sigma12_sq = F.conv2d(_predicted * _target, window, padding=0, groups=channels) - mu1_mu2
        ssim_map = (2 * mu1_mu2 + self.c1) * (2 * sigma12_sq + self.c2) / ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
        if mask is not None:
            ssim_map = ssim_map[..., mask]
            predicted = predicted[..., mask]
            target = target[..., mask]
        ssim = ssim_map.mean().abs()
        if self.mse:
            ssim = ssim + self.mse(predicted, target)
        return ssim


class MSE_SSIM(torch.nn.Module):
    """
    This class provides a compound loss formulation combining differential structural similarity (SSIM) and mean squared
    error (MSE). Calling this class will compute the loss using SSIM for fields indicated by model attributes
    (model.ssim_fields).
    """

    def __init__(self, mse_params=None, ssim_params=None, ssim_variables=['ttr1h', 'tcwv0'], weights=[0.5, 0.5]):
        """
        Parameters:
        ----------
        mse_params: optional
            parameters to pass to MSE constructor
        ssim_params: optional
            dictionary of parameters to pass to SSIM constructor
        ssim variables: list, optional
            list of variables over which loss will be calculated using DSSIM and MSE
        param weights: list, optional
            variables identified as requireing SSIM-loss calculation
            will have their loss calculated by a weighted average od the DSSIM metric and MSE.
            The weights of this weighted average are  [MSE_weight, DSSIM_weight]
        """
        super(MSE_SSIM, self).__init__()
        if ssim_params is None:
            self.ssim = SSIM()
        else:
            self.ssim = SSIM(**ssim_params)
        if mse_params is None:
            self.mse = torch.nn.MSELoss()
        else:
            self.mse = torch.nn.MSELoss(**mse_params)
        if np.sum(weights) == 1:
            self.mse_dssim_weights = weights
        else:
            raise ValueError('Weights passed to MSE_SSIM loss must sum to 1')
        self.ssim_variables = ssim_variables

    def forward(self, prediction: 'torch.tensor', targets: 'torch.tensor', model: 'torch.nn.Module'):
        """
        Forward pass of the MSE_SSIM loss

        param prediction: torch.Tensor
            Predicted image of shape [B, T, C, F, H, W]
        param targets: torch.Tensor
            Ground truth image of shape [B, T, C, F, H, W]
        param model: torch.nn.Module
            model over which loss is being computed

        Returns
        -------
        torch.Tensor
            The structural similarity loss
        """
        if prediction.shape[-1] != prediction.shape[-2]:
            raise AssertionError(f'Spatial dims H and W must match: got {prediction.shape[-2]} and {prediction.shape[-1]}')
        if prediction.shape[3] != 12:
            raise AssertionError(f'Spatial dim F must be 12: got {prediction.shape[3]}')
        if prediction.shape[2] != model.output_channels:
            raise AssertionError(f"model output channels and prediction output channels don't match: got {model.output_channels} for model and {prediction.shape[2]} for input")
        if not (prediction.shape[1] == model.output_time_dim or prediction.shape[1] == model.output_time_dim // model.input_time_dim):
            raise AssertionError('Number of time steps in prediction must equal to model output time dim, or model output time dime divided by model input time dim')
        device = prediction.device
        loss_by_var = torch.empty([prediction.shape[2]], device=device)
        weights = torch.tensor(self.mse_dssim_weights, device=device)
        for i, v in enumerate(model.output_variables):
            var_mse = self.mse(prediction[:, :, i:i + 1, :, :, :], targets[:, :, i:i + 1, :, :, :])
            if v in self.ssim_variables:
                var_dssim = torch.min(torch.tensor([1.0, float(var_mse)])) * (1 - self.ssim(prediction[:, :, i:i + 1, :, :, :], targets[:, :, i:i + 1, :, :, :]))
                loss_by_var[i] = torch.sum(weights * torch.stack([var_mse, var_dssim]))
            else:
                loss_by_var[i] = var_mse
        loss = loss_by_var.mean()
        return loss


class AFNOMlp(nn.Module):
    """Fully-connected Multi-layer perception used inside AFNO

    Parameters
    ----------
    in_features : int
        Input feature size
    latent_features : int
        Latent feature size
    out_features : int
        Output feature size
    activation_fn :  nn.Module, optional
        Activation function, by default nn.GELU
    drop : float, optional
        Drop out rate, by default 0.0
    """

    def __init__(self, in_features: 'int', latent_features: 'int', out_features: 'int', activation_fn: 'nn.Module'=nn.GELU(), drop: 'float'=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2DLayer(nn.Module):
    """AFNO spectral convolution layer

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    hidden_size_factor : int, optional
        Factor to increase spectral features by after weight multiplication, by default 1
    """

    def __init__(self, hidden_size: 'int', num_blocks: 'int'=8, sparsity_threshold: 'float'=0.01, hard_thresholding_fraction: 'float'=1, hidden_size_factor: 'int'=1):
        super().__init__()
        if not hidden_size % num_blocks == 0:
            raise ValueError(f'hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}')
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x: 'Tensor') ->Tensor:
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        x = fft.rfft2(x, dim=(1, 2), norm='ortho')
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(torch.einsum('nyxbi,bio->nyxbo', x_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[0]) - torch.einsum('nyxbi,bio->nyxbo', x_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[1]) + self.b1[0])
        o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(torch.einsum('nyxbi,bio->nyxbo', x_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[0]) + torch.einsum('nyxbi,bio->nyxbo', x_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[1]) + self.b1[1])
        o2[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, ..., 0] = torch.einsum('nyxbi,bio->nyxbo', o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) - torch.einsum('nyxbi,bio->nyxbo', o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + self.b2[0]
        o2[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, ..., 1] = torch.einsum('nyxbi,bio->nyxbo', o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) + torch.einsum('nyxbi,bio->nyxbo', o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + self.b2[1]
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.type(dtype)
        return x + bias


class ScaleShiftMlp(nn.Module):
    """MLP used to compute the scale and shift parameters of the ModAFNO block

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    hidden_features : int, optional
        Hidden feature size, defaults to 2 * out_features
    hidden_layers : int, optional
        Number of hidden layers, defaults to 0
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    """

    def __init__(self, in_features: 'int', out_features: 'int', hidden_features: 'Union[int, None]'=None, hidden_layers: 'int'=0, activation_fn: 'Type[nn.Module]'=nn.GELU):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features * 2
        sequence = [nn.Linear(in_features, hidden_features), activation_fn()]
        for _ in range(hidden_layers):
            sequence += [nn.Linear(hidden_features, hidden_features), activation_fn()]
        sequence.append(nn.Linear(hidden_features, out_features * 2))
        self.net = nn.Sequential(*sequence)

    def forward(self, x: 'Tensor'):
        scale, shift = torch.chunk(self.net(x), 2, dim=1)
        return 1 + scale, shift


class ModAFNO2DLayer(AFNO2DLayer):
    """AFNO spectral convolution layer

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality
    mod_features : int
        Number of modulation features
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    hidden_size_factor : int, optional
        Factor to increase spectral features by after weight multiplication, by default 1
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
    """

    def __init__(self, hidden_size: 'int', mod_features: 'int', num_blocks: 'int'=8, sparsity_threshold: 'float'=0.01, hard_thresholding_fraction: 'float'=1, hidden_size_factor: 'int'=1, scale_shift_kwargs: 'Union[dict, None]'=None, scale_shift_mode: "Literal['complex', 'real']"='complex'):
        super().__init__(hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction, hidden_size_factor=hidden_size_factor)
        if scale_shift_mode not in ('complex', 'real'):
            raise ValueError("scale_shift_mode must be 'real' or 'complex'")
        self.scale_shift_mode = scale_shift_mode
        self.channel_mul = 1 if scale_shift_mode == 'real' else 2
        if scale_shift_kwargs is None:
            scale_shift_kwargs = {}
        self.scale_shift = ScaleShiftMlp(mod_features, self.num_blocks * self.block_size * self.hidden_size_factor * self.channel_mul, **scale_shift_kwargs)

    def forward(self, x: 'Tensor', mod_embed: 'Tensor') ->Tensor:
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        x = fft.rfft2(x, dim=(1, 2), norm='ortho')
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        o1_shape = B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor
        scale_shift_shape = B, self.channel_mul, 1, o1_shape[3], o1_shape[4]
        o1_real = torch.zeros(o1_shape, device=x.device)
        o1_imag = torch.zeros(o1_shape, device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)
        total_modes = min(H, W) // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        o1_re = torch.einsum('nyxbi,bio->nyxbo', x_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[0]) - torch.einsum('nyxbi,bio->nyxbo', x_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[1]) + self.b1[0]
        o1_im = torch.einsum('nyxbi,bio->nyxbo', x_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[0]) + torch.einsum('nyxbi,bio->nyxbo', x_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w1[1]) + self.b1[1]
        scale, shift = self.scale_shift(mod_embed)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)
        if self.scale_shift_mode == 'real':
            o1_re = o1_re * scale + shift
            o1_im = o1_im * scale + shift
        elif self.scale_shift_mode == 'complex':
            scale_re, scale_im = torch.chunk(scale, 2, dim=1)
            shift_re, shift_im = torch.chunk(shift, 2, dim=1)
            o1_re, o1_im = o1_re * scale_re - o1_im * scale_im + shift_re, o1_im * scale_re + o1_re * scale_im + shift_im
        o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(o1_re)
        o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes] = F.relu(o1_im)
        o2[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, ..., 0] = torch.einsum('nyxbi,bio->nyxbo', o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) - torch.einsum('nyxbi,bio->nyxbo', o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + self.b2[0]
        o2[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, ..., 1] = torch.einsum('nyxbi,bio->nyxbo', o1_imag[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[0]) + torch.einsum('nyxbi,bio->nyxbo', o1_real[:, total_modes - kept_modes:total_modes + kept_modes, :kept_modes], self.w2[1]) + self.b2[1]
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.type(dtype)
        return x + bias


class ModAFNOMlp(AFNOMlp):
    """Modulated MLP used inside ModAFNO

    Parameters
    ----------
    in_features : int
        Input feature size
    latent_features : int
        Latent feature size
    out_features : int
        Output feature size
    activation_fn :  nn.Module, optional
        Activation function, by default nn.GELU
    drop : float, optional
        Drop out rate, by default 0.0
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters
    """

    def __init__(self, in_features: 'int', latent_features: 'int', out_features: 'int', mod_features: 'int', activation_fn: 'nn.Module'=nn.GELU(), drop: 'float'=0.0, scale_shift_kwargs: 'Union[dict, None]'=None):
        super().__init__(in_features=in_features, latent_features=latent_features, out_features=out_features, activation_fn=activation_fn, drop=drop)
        if scale_shift_kwargs is None:
            scale_shift_kwargs = {}
        self.scale_shift = ScaleShiftMlp(mod_features, latent_features, **scale_shift_kwargs)

    def forward(self, x: 'Tensor', mod_embed: 'Tensor') ->Tensor:
        scale, shift = self.scale_shift(mod_embed)
        scale_shift_shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[1],)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)
        x = self.fc1(x)
        x = x * scale + shift
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """AFNO block, spectral convolution and MLP

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality
    mod_dim : int
        Modululation input dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    mlp_ratio : float, optional
        Ratio of MLP latent variable size to input feature size, by default 4.0
    drop : float, optional
        Drop out rate in MLP, by default 0.0
    activation_fn: nn.Module, optional
        Activation function used in MLP, by default nn.GELU
    norm_layer : nn.Module, optional
        Normalization function, by default nn.LayerNorm
    double_skip : bool, optional
        Residual, by default True
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    modulate_filter: bool, optional
        Whether to compute the modulation for the FFT filter
    modulate_mlp: bool, optional
        Whether to compute the modulation for the MLP
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
    """

    def __init__(self, embed_dim: 'int', mod_dim: 'int', num_blocks: 'int'=8, mlp_ratio: 'float'=4.0, drop: 'float'=0.0, activation_fn: 'nn.Module'=nn.GELU(), norm_layer: 'nn.Module'=nn.LayerNorm, double_skip: 'bool'=True, sparsity_threshold: 'float'=0.01, hard_thresholding_fraction: 'float'=1.0, modulate_filter: 'bool'=True, modulate_mlp: 'bool'=True, scale_shift_mode: "Literal['complex', 'real']"='real'):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        if modulate_filter:
            self.filter = ModAFNO2DLayer(embed_dim, mod_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction, scale_shift_mode=scale_shift_mode)
            self.apply_filter = lambda x, mod_embed: self.filter(x, mod_embed)
        else:
            self.filter = AFNO2DLayer(embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
            self.apply_filter = lambda x, mod_embed: self.filter(x)
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        if modulate_mlp:
            self.mlp = ModAFNOMlp(in_features=embed_dim, latent_features=mlp_latent_dim, out_features=embed_dim, mod_features=mod_dim, activation_fn=activation_fn, drop=drop)
            self.apply_mlp = lambda x, mod_embed: self.mlp(x, mod_embed)
        else:
            self.mlp = AFNOMlp(in_features=embed_dim, latent_features=mlp_latent_dim, out_features=embed_dim, activation_fn=activation_fn, drop=drop)
            self.apply_mlp = lambda x, mod_embed: self.mlp(x)
        self.double_skip = double_skip
        self.modulate_filter = modulate_filter
        self.modulate_mlp = modulate_mlp

    def forward(self, x: 'Tensor', mod_embed: 'Tensor') ->Tensor:
        residual = x
        x = self.norm1(x)
        x = self.apply_filter(x, mod_embed)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.apply_mlp(x, mod_embed)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer

    Converts 2D patch into a 1D vector for input to AFNO

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    """

    def __init__(self, inp_shape: 'List[int]', in_channels: 'int', patch_size: 'List[int]'=[16, 16], embed_dim: 'int'=256):
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError('inp_shape should be a list of length 2')
        if len(patch_size) != 2:
            raise ValueError('patch_size should be a list of length 2')
        num_patches = inp_shape[1] // patch_size[1] * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: 'Tensor') ->Tensor:
        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(f"Input image size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AFNO(Module):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    Example
    -------
    >>> model = modulus.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """

    def __init__(self, inp_shape: 'List[int]', in_channels: 'int', out_channels: 'int', patch_size: 'List[int]'=[16, 16], embed_dim: 'int'=256, depth: 'int'=4, mlp_ratio: 'float'=4.0, drop_rate: 'float'=0.0, num_blocks: 'int'=16, sparsity_threshold: 'float'=0.01, hard_thresholding_fraction: 'float'=1.0) ->None:
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError('inp_shape should be a list of length 2')
        if len(patch_size) != 2:
            raise ValueError('patch_size should be a list of length 2')
        if not (inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0):
            raise ValueError(f'input shape {inp_shape} should be divisible by patch_size {patch_size}')
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbed(inp_shape=inp_shape, in_channels=self.in_chans, patch_size=self.patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]
        self.blocks = nn.ModuleList([Block(embed_dim=embed_dim, num_blocks=self.num_blocks, mlp_ratio=mlp_ratio, drop=drop_rate, norm_layer=norm_layer, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) for i in range(depth)])
        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: 'Tensor') ->Tensor:
        """Forward pass of core AFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        return out


class ModulusUndefinedGroupError(Exception):
    """Exception for querying an undefined process group using the Modulus DistributedManager"""

    def __init__(self, name: 'str'):
        """

        Parameters
        ----------
        name : str
            Name of the process group being queried.

        """
        message = f"Cannot query process group '{name}' before it is explicitly created."
        super().__init__(message)


class ModulusUninitializedDistributedManagerWarning(Warning):
    """Warning to indicate usage of an uninitialized DistributedManager"""

    def __init__(self):
        message = 'A DistributedManager object is being instantiated before ' + 'this singleton class has been initialized. Instantiating a manager before ' + 'initialization can lead to unexpected results where processes fail ' + 'to communicate. Initialize the distributed manager via ' + 'DistributedManager.initialize() before instantiating.'
        super().__init__(message)


class DistributedManager(object):
    """Distributed Manager for setting up distributed training environment.

    This is a singleton that creates a persistance class instance for storing parallel
    environment information through out the life time of the program. This should be
    used to help set up Distributed Data Parallel and parallel datapipes.

    Note
    ----
    One should call `DistributedManager.initialize()` prior to constructing a manager
    object

    Example
    -------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """
    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if not hasattr(obj, '_rank'):
            obj._rank = 0
        if not hasattr(obj, '_world_size'):
            obj._world_size = 1
        if not hasattr(obj, '_local_rank'):
            obj._local_rank = 0
        if not hasattr(obj, '_distributed'):
            obj._distributed = False
        if not hasattr(obj, '_device'):
            obj._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not hasattr(obj, '_cuda'):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, '_broadcast_buffers'):
            obj._broadcast_buffers = False
        if not hasattr(obj, '_find_unused_parameters'):
            obj._find_unused_parameters = False
        if not hasattr(obj, '_initialization_method'):
            obj._initialization_method = 'None'
        if not hasattr(obj, '_groups'):
            obj._groups = {}
        if not hasattr(obj, '_group_ranks'):
            obj._group_ranks = {}
        if not hasattr(obj, '_group_names'):
            obj._group_names = {}
        if not hasattr(obj, '_is_initialized'):
            obj._is_initialized = False
        return obj

    def __init__(self):
        if not self._is_initialized:
            raise ModulusUninitializedDistributedManagerWarning()
        super().__init__()

    @property
    def rank(self):
        """Process rank"""
        return self._rank

    @property
    def local_rank(self):
        """Process rank on local machine"""
        return self._local_rank

    @property
    def world_size(self):
        """Number of processes in distributed enviroment"""
        return self._world_size

    @property
    def device(self):
        """Process device"""
        return self._device

    @property
    def distributed(self):
        """Distributed enviroment"""
        return self._distributed

    @property
    def cuda(self):
        """If cuda is available"""
        return self._cuda

    @property
    def group_names(self):
        """
        Returns a list of all named process groups created
        """
        return self._groups.keys()

    def group(self, name=None):
        """
        Returns a process group with the given name
        If name is None, group is also None indicating the default process group
        If named group does not exist, ModulusUndefinedGroupError exception is raised
        """
        if name in self._groups.keys():
            return self._groups[name]
        elif name is None:
            return None
        else:
            raise ModulusUndefinedGroupError(name)

    def group_size(self, name=None):
        """
        Returns the size of named process group
        """
        if name is None:
            return self._world_size
        group = self.group(name)
        return dist.get_world_size(group=group)

    def group_rank(self, name=None):
        """
        Returns the rank in named process group
        """
        if name is None:
            return self._rank
        group = self.group(name)
        return dist.get_rank(group=group)

    def group_name(self, group=None):
        """
        Returns the name of process group
        """
        if group is None:
            return None
        return self._group_names[group]

    @property
    def broadcast_buffers(self):
        """broadcast_buffers in PyTorch DDP"""
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: 'bool'):
        """Setter for broadcast_buffers"""
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        """find_unused_parameters in PyTorch DDP"""
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: 'bool'):
        """Setter for find_unused_parameters"""
        if find_params:
            warn('Setting `find_unused_parameters` in DDP to true, use only if necessary.')
        self._find_unused_parameters = find_params

    def __str__(self):
        output = f"Initialized process {self.rank} of {self.world_size} using method '{self._initialization_method}'. Device set to {str(self.device)}"
        return output

    @classmethod
    def is_initialized(cls) ->bool:
        """If manager singleton has been initialized"""
        return cls._shared_state.get('_is_initialized', False)

    @staticmethod
    def get_available_backend():
        """Get communication backend"""
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            return 'nccl'
        else:
            return 'gloo'

    @staticmethod
    def initialize_env():
        """Setup method using generic initialization"""
        rank = int(os.environ.get('RANK'))
        world_size = int(os.environ.get('WORLD_SIZE'))
        if 'LOCAL_RANK' in os.environ:
            local_rank = os.environ.get('LOCAL_RANK')
            if local_rank is not None:
                local_rank = int(local_rank)
            else:
                local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = rank % torch.cuda.device_count()
        addr = os.environ.get('MASTER_ADDR')
        port = os.environ.get('MASTER_PORT')
        DistributedManager.setup(rank=rank, world_size=world_size, local_rank=local_rank, addr=addr, port=port, backend=DistributedManager.get_available_backend())

    @staticmethod
    def initialize_open_mpi(addr, port):
        """Setup method using OpenMPI initialization"""
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
        DistributedManager.setup(rank=rank, world_size=world_size, local_rank=local_rank, addr=addr, port=port, backend=DistributedManager.get_available_backend(), method='openmpi')

    @staticmethod
    def initialize_slurm(port):
        """Setup method using SLURM initialization"""
        rank = int(os.environ.get('SLURM_PROCID'))
        world_size = int(os.environ.get('SLURM_NPROCS'))
        local_rank = int(os.environ.get('SLURM_LOCALID'))
        addr = os.environ.get('SLURM_LAUNCH_NODE_IPADDR')
        DistributedManager.setup(rank=rank, world_size=world_size, local_rank=local_rank, addr=addr, port=port, backend=DistributedManager.get_available_backend(), method='slurm')

    @staticmethod
    def initialize():
        """
        Initialize distributed manager

        Current supported initialization methods are:
            `ENV`: PyTorch environment variable initialization
                 https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            `SLURM`: Initialization on SLURM systems.
                   Uses `SLURM_PROCID`, `SLURM_NPROCS`, `SLURM_LOCALID` and
                   `SLURM_LAUNCH_NODE_IPADDR` environment variables.
            `OPENMPI`: Initialization for OpenMPI launchers.
                     Uses `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE` and
                     `OMPI_COMM_WORLD_LOCAL_RANK` environment variables.

        Initialization by default is done using the first valid method in the order
        listed above. Initialization method can also be explicitly controlled using the
        `MODULUS_DISTRIBUTED_INITIALIZATION_METHOD` environment variable and setting it
        to one of the options above.
        """
        if DistributedManager.is_initialized():
            warn('Distributed manager is already intialized')
            return
        addr = os.getenv('MASTER_ADDR', 'localhost')
        port = os.getenv('MASTER_PORT', '12355')
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
        initialization_method = os.getenv('MODULUS_DISTRIBUTED_INITIALIZATION_METHOD')
        if initialization_method is None:
            try:
                DistributedManager.initialize_env()
            except TypeError:
                if 'SLURM_PROCID' in os.environ:
                    DistributedManager.initialize_slurm(port)
                elif 'OMPI_COMM_WORLD_RANK' in os.environ:
                    DistributedManager.initialize_open_mpi(addr, port)
                else:
                    warn('Could not initialize using ENV, SLURM or OPENMPI methods. Assuming this is a single process job')
                    DistributedManager._shared_state['_is_initialized'] = True
        elif initialization_method == 'ENV':
            DistributedManager.initialize_env()
        elif initialization_method == 'SLURM':
            DistributedManager.initialize_slurm(port)
        elif initialization_method == 'OPENMPI':
            DistributedManager.initialize_open_mpi(addr, port)
        else:
            raise RuntimeError(f'Unknown initialization method {initialization_method}. Supported values for MODULUS_DISTRIBUTED_INITIALIZATION_METHOD are ENV, SLURM and OPENMPI')
        np.random.seed(seed=DistributedManager().rank)

    @staticmethod
    def setup(rank=0, world_size=1, local_rank=None, addr='localhost', port='12355', backend='nccl', method='env'):
        """Set up PyTorch distributed process group and update manager attributes"""
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = str(port)
        DistributedManager._shared_state['_is_initialized'] = True
        manager = DistributedManager()
        manager._distributed = torch.distributed.is_available()
        if manager._distributed:
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank
        manager._device = torch.device(f'cuda:{manager.local_rank}' if torch.cuda.is_available() else 'cpu')
        if manager._distributed:
            try:
                dist.init_process_group(backend, rank=manager.rank, world_size=manager.world_size, device_id=manager.device)
            except TypeError:
                dist.init_process_group(backend, rank=manager.rank, world_size=manager.world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(manager.device)
            torch.device(manager.device)
            torch.cuda.empty_cache()
        manager._initialization_method = method

    @staticmethod
    def create_process_subgroup(name: 'str', size: 'int', group_name: 'Optional[str]'=None, verbose: 'bool'=False):
        """
        Create a process subgroup of a parent process group. This must be a collective
        call by all processes participating in this application.

        Parameters
        ----------
        name : str
            Name of the process subgroup to be created.

        size : int
            Size of the process subgroup to be created. This must be an integer factor of
            the parent group's size.

        group_name : Optional[str]
            Name of the parent process group, optional. If None, the default process group
            will be used. Default None.

        verbose : bool
            Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError('torch.distributed is unavailable. Check pytorch build to ensure the distributed package is available. If building PyTorch from source, set `USE_DISTRIBUTED=1` to enable the distributed package')
        if name in manager._groups:
            raise AssertionError(f'Group with name {name} already exists')
        group = manager._groups[group_name] if group_name else None
        group_size = dist.get_world_size(group=group)
        num_groups = manager.world_size // group_size
        if group_size % size != 0:
            raise AssertionError(f'Cannot divide group size {group_size} evenly into subgroups of size {size}')
        num_subgroups = group_size // size
        manager._group_ranks[name] = []
        for g in range(num_groups):
            for i in range(num_subgroups):
                start = i * size
                end = start + size
                if group_name:
                    ranks = manager._group_ranks[group_name][g][start:end]
                else:
                    ranks = list(range(start, end))
                tmp_group = dist.new_group(ranks=ranks)
                manager._group_ranks[name].append(ranks)
                if manager.rank in ranks:
                    manager._groups[name] = tmp_group
                    manager._group_names[tmp_group] = name
        if verbose and manager.rank == 0:
            None
            for grp in manager._group_ranks[name]:
                None

    @staticmethod
    def create_orthogonal_process_group(orthogonal_group_name: 'str', group_name: 'str', verbose: 'bool'=False):
        """
        Create a process group that is orthogonal to the specified process group.

        Parameters
        ----------
        orthogonal_group_name : str
            Name of the orthogonal process group to be created.

        group_name : str
            Name of the existing process group.

        verbose : bool
            Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError('torch.distributed is unavailable. Check pytorch build to ensure the distributed package is available. If building PyTorch from source, set `USE_DISTRIBUTED=1` to enable the distributed package')
        if group_name not in manager._groups:
            raise ValueError(f'Group with name {group_name} does not exist')
        if orthogonal_group_name in manager._groups:
            raise ValueError(f'Group with name {orthogonal_group_name} already exists')
        group_ranks = manager._group_ranks[group_name]
        orthogonal_ranks = [list(i) for i in zip(*group_ranks)]
        for ranks in orthogonal_ranks:
            tmp_group = dist.new_group(ranks=ranks)
            if manager.rank in ranks:
                manager._groups[orthogonal_group_name] = tmp_group
                manager._group_names[tmp_group] = orthogonal_group_name
        manager._group_ranks[orthogonal_group_name] = orthogonal_ranks
        if verbose and manager.rank == 0:
            None
            for grp in manager._group_ranks[orthogonal_group_name]:
                None

    @staticmethod
    def create_group_from_node(node: 'ProcessGroupNode', parent: 'Optional[str]'=None, verbose: 'bool'=False):
        if node.size is None:
            raise AssertionError('Cannot create groups from a ProcessGroupNode that is not fully populated. Ensure that config.set_leaf_group_sizes is called first with `update_parent_sizes = True`')
        DistributedManager.create_process_subgroup(node.name, node.size, group_name=parent, verbose=verbose)
        orthogonal_group = f'__orthogonal_to_{node.name}'
        DistributedManager.create_orthogonal_process_group(orthogonal_group, node.name, verbose=verbose)
        return orthogonal_group

    @staticmethod
    def create_groups_from_config(config: 'ProcessGroupConfig', verbose: 'bool'=False):
        q = queue.Queue()
        q.put(config.root_id)
        DistributedManager.create_group_from_node(config.root)
        while not q.empty():
            node_id = q.get()
            if verbose:
                None
            children = config.tree.children(node_id)
            if verbose:
                None
            parent_group = node_id
            for child in children:
                parent_group = DistributedManager.create_group_from_node(child.data, parent=parent_group)
                q.put(child.identifier)

    @staticmethod
    def cleanup():
        """Clean up distributed group and singleton"""
        if '_is_initialized' in DistributedManager._shared_state and DistributedManager._shared_state['_is_initialized'] and '_distributed' in DistributedManager._shared_state and DistributedManager._shared_state['_distributed']:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[DistributedManager().local_rank])
            else:
                dist.barrier()
            dist.destroy_process_group()
        DistributedManager._shared_state = {}


@torch.jit.script
def compl_mul_add_fwd(a: 'torch.Tensor', b: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
    tmp = torch.einsum('bkixys,kiot->stbkoxy', a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1) + c
    return res


@torch.jit.script
def compl_mul_add_fwd_c(a: 'torch.Tensor', b: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum('bkixy,kio->bkoxy', ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


def compute_split_shapes(size: 'int', num_chunks: 'int') ->List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]
    return sections


def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(f'Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}')
    if tensor.shape[dim] < num_chunks:
        raise ValueError('Error, cannot split dim {dim} of size {tensor.shape[dim]} into         {num_chunks} chunks. Empty slices are currently not supported.')
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)
    return tensor_list


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    input_format = get_memory_format(input_)
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)
    return output


def all_gather_v_wrapper(tensor: 'torch.Tensor', sizes: 'Optional[List[int]]'=None, dim: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """
    comm_size = dist.get_world_size(group=group)
    if sizes is not None and len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if comm_size == 1:
        return tensor
    tensor_shape = list(tensor.shape)
    tensor_format = get_memory_format(tensor)
    if sizes is not None:
        tensor_list = [None] * comm_size
        for src in range(comm_size):
            tensor_shape[dim] = sizes[src]
            tensor_list[src] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        tensor_list = [torch.empty_like(tensor) for _ in range(comm_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    output = torch.cat(tensor_list, dim=dim).contiguous(memory_format=tensor_format)
    return output


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_, shapes_):
        return all_gather_v_wrapper(input_, shapes_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, group_):
        ctx.dim = dim_
        ctx.group = group_
        return all_gather_v_wrapper(input_, shapes_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=DistributedManager().group(ctx.group)), None, None, None


def gather_from_parallel_region(input, dim, shapes, group):
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromParallelRegion.apply(input, dim, shapes, group)


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_):
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, group_):
        ctx.dim = dim_
        ctx.group = group_
        ctx.split_shapes = compute_split_shapes(input_.shape[dim_], DistributedManager().group_size(group_))
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):
        return all_gather_v_wrapper(grad_output, ctx.split_shapes, ctx.dim, group=DistributedManager().group(ctx.group)), None, None


def scatter_to_parallel_region(input, dim, group):
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToParallelRegion.apply(input, dim, group)


class DistributedAFNO2D(nn.Module):

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, input_is_matmul_parallel=False, output_is_matmul_parallel=False):
        super(DistributedAFNO2D, self).__init__()
        if not hidden_size % num_blocks == 0:
            raise ValueError(f'hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}')
        matmul_comm_size = DistributedManager().group_size('model_parallel')
        self.fft_handle = torch.fft.rfft2
        self.ifft_handle = torch.fft.irfft2
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        if not self.num_blocks % matmul_comm_size == 0:
            raise ValueError('Error, num_blocks needs to be divisible by matmul_parallel_size')
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        use_complex_mult = False
        self.mult_handle = compl_mul_add_fwd_c if use_complex_mult else compl_mul_add_fwd
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, self.block_size * self.hidden_size_factor, 2))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, 1, 1, 2))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size * self.hidden_size_factor, self.block_size, 2))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2))
        self.w1.is_shared_spatial = True
        self.b1.is_shared_spatial = True
        self.w2.is_shared_spatial = True
        self.b2.is_shared_spatial = True

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            num_chans = x.shape[1]
            x = scatter_to_parallel_region(x, dim=1, group='model_parallel')
        bias = x
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        x = self.fft_handle(x, (H, W), (-2, -1), 'ortho')
        x = x.view(B, self.num_blocks_local, self.block_size, H, W // 2 + 1)
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)
        o1 = F.relu(self.mult_handle(x[:, :, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, :], self.w1, self.b1))
        o2[:, :, :, total_modes - kept_modes:total_modes + kept_modes, :kept_modes, :] = self.mult_handle(o1, self.w2, self.b2)
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), 'ortho')
        x = x.type(dtype) + bias
        if not self.output_is_matmul_parallel:
            gather_shapes = compute_split_shapes(num_chans, DistributedManager().group_size('model_parallel'))
            x = gather_from_parallel_region(x, dim=1, shapes=gather_shapes, group='model_parallel')
        return x


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""
    if dist.get_world_size(group=group) == 1:
        return input_
    if use_fp32 and input_.dtype.itemsize < 4 and input_.dtype.is_floating_point:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_
    else:
        dist.all_reduce(input_, group=group)
    return input_


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):
        return input_

    @staticmethod
    def forward(ctx, input_, group_):
        ctx.group = group_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=DistributedManager().group(ctx.group)), None


def copy_to_parallel_region(input, group):
    """Copy input"""
    return _CopyToParallelRegion.apply(input, group)


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, group_):
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def reduce_from_parallel_region(input, group):
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromParallelRegion.apply(input, group)


def _trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    u1 = norm_cdf((a - mean) / std)
    u2 = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * u1 - 1, 2 * u2 - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Cut & paste from timm master
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class DistributedMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, input_is_matmul_parallel=False, output_is_matmul_parallel=False):
        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        comm_size = DistributedManager().group_size('model_parallel')
        if not hidden_features % comm_size == 0:
            raise ValueError('Error, hidden_features needs to be divisible by matmul_parallel_size')
        hidden_features_local = hidden_features // comm_size
        self.w1 = nn.Parameter(torch.ones(hidden_features_local, in_features, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_features_local))
        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features_local, 1, 1))
        self.b2 = nn.Parameter(torch.zeros(out_features))
        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        if self.input_is_matmul_parallel:
            self.gather_shapes = compute_split_shapes(in_features, DistributedManager().group_size('model_parallel'))
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        trunc_normal_(self.w2, std=0.02)
        nn.init.constant_(self.b2, 0.0)

    def forward(self, x):
        if self.input_is_matmul_parallel:
            x = gather_from_parallel_region(x, dim=1, shapes=self.gather_shapes, group='model_parallel')
        x = copy_to_parallel_region(x, group='model_parallel')
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = F.conv2d(x, self.w2, bias=None)
        x = reduce_from_parallel_region(x, group='model_parallel')
        x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        x = self.drop(x)
        if self.output_is_matmul_parallel:
            x = scatter_to_parallel_region(x, dim=1, group='model_parallel')
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False, scale_by_keep: 'bool'=True):
    """Cut & paste from timm master
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Cut & paste from timm master
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: 'float'=0.0, scale_by_keep: 'bool'=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class DistributedBlock(nn.Module):

    def __init__(self, h, w, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, double_skip=True, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, input_is_matmul_parallel=False, output_is_matmul_parallel=False):
        super(DistributedBlock, self).__init__()
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        self.norm1 = norm_layer((h, w))
        self.filter = DistributedAFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction, input_is_matmul_parallel=True, output_is_matmul_parallel=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer((h, w))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DistributedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, input_is_matmul_parallel=True, output_is_matmul_parallel=True)
        self.double_skip = double_skip

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            scatter_shapes = compute_split_shapes(x.shape[1], DistributedManager().group_size('model_parallel'))
            x = scatter_to_parallel_region(x, dim=1, group='model_parallel')
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        if not self.output_is_matmul_parallel:
            x = gather_from_parallel_region(x, dim=1, shapes=scatter_shapes, group='model_parallel')
        return x


class DistributedPatchEmbed(nn.Module):

    def __init__(self, inp_shape=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, input_is_matmul_parallel=False, output_is_matmul_parallel=True):
        super(DistributedPatchEmbed, self).__init__()
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel
        matmul_comm_size = DistributedManager().group_size('model_parallel')
        num_patches = inp_shape[1] // patch_size[1] * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape[0], inp_shape[1]
        self.patch_size = patch_size
        self.num_patches = num_patches
        if self.input_parallel:
            if not in_chans % matmul_comm_size == 0:
                raise ValueError('Error, the in_chans needs to be divisible by matmul_parallel_size')
            self.in_shapes = compute_split_shapes(in_chans, DistributedManager().group_size('model_parallel'))
        if self.output_parallel:
            if not embed_dim % matmul_comm_size == 0:
                raise ValueError('Error, the embed_dim needs to be divisible by matmul_parallel_size')
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim
        self.proj = nn.Conv2d(in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size)
        self.proj.weight.is_shared_spatial = True
        self.proj.bias.is_shared_spatial = True

    def forward(self, x):
        if self.input_parallel:
            x = gather_from_parallel_region(x, dim=1, shapes=self.in_shapes, group='model_parallel')
        if self.output_parallel:
            x = copy_to_parallel_region(x, group='model_parallel')
        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(f"Input input size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]}).")
        x = self.proj(x).flatten(2)
        return x


class DistributedAFNONet(nn.Module):

    def __init__(self, inp_shape=(720, 1440), patch_size=(16, 16), in_chans=2, out_chans=2, embed_dim=768, depth=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, num_blocks=16, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, input_is_matmul_parallel=False, output_is_matmul_parallel=False):
        super().__init__()
        matmul_comm_size = DistributedManager().group_size('model_parallel')
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = DistributedPatchEmbed(inp_shape=inp_shape, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim, input_is_matmul_parallel=self.input_is_matmul_parallel, output_is_matmul_parallel=True)
        num_patches = self.patch_embed.num_patches
        self.embed_dim_local = self.embed_dim // matmul_comm_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim_local, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]
        blks = []
        for i in range(0, depth):
            input_is_matmul_parallel = True
            output_is_matmul_parallel = True if i < depth - 1 else False
            blks.append(DistributedBlock(h=self.h, w=self.w, dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction, input_is_matmul_parallel=input_is_matmul_parallel, output_is_matmul_parallel=output_is_matmul_parallel))
        self.blocks = nn.ModuleList(blks)
        if self.output_is_matmul_parallel:
            self.out_chans_local = (self.out_chans + matmul_comm_size - 1) // matmul_comm_size
        else:
            self.out_chans_local = self.out_chans
        self.head = nn.Conv2d(self.embed_dim, self.out_chans_local * self.patch_size[0] * self.patch_size[1], 1, bias=False)
        self.synchronized_head = False
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, self.embed_dim_local, self.h, self.w)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.output_is_matmul_parallel:
            x = copy_to_parallel_region(x, group='model_parallel')
        elif not self.synchronized_head:
            for param in self.head.parameters():
                dist.broadcast(param, 0, group=DistributedManager().group('model_parallel'))
            self.synchronized_head = True
        x = self.head(x)
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(b, -1, self.h * self.patch_size[0], self.w * self.patch_size[1])
        return x


class OneHotEmbedding(nn.Module):
    """
    A module for generating one-hot embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: 'int'):
        super().__init__()
        self.num_channels = num_channels
        ind = torch.arange(num_channels)
        ind = ind.view(1, len(ind))
        self.register_buffer('indices', ind)

    def forward(self, t: 'Tensor') ->Tensor:
        ind = t * (self.num_channels - 1)
        return torch.clamp(1 - torch.abs(ind - self.indices), min=0)


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.

    """

    def __init__(self, num_channels: 'int', max_positions: 'int'=10000, endpoint: 'bool'=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class ModEmbedNet(nn.Module):
    """
    A network that generates a timestep embedding and processes it with an MLP.

    Parameters:
    -----------
    max_time : float, optional
        Maximum input time. The inputs to `forward` is should be in the range [0, max_time].
    dim : int, optional
        The dimensionality of the time embedding.
    depth : int, optional
        The number of layers in the MLP.
    activation_fn:
        The activation function, default GELU.
    method : str, optional
        The embedding method. Either "sinusoidal" (default) or "onehot".
    """

    def __init__(self, max_time: 'float'=1.0, dim: 'int'=64, depth: 'int'=1, activation_fn: 'Type[nn.Module]'=nn.GELU, method: 'str'='sinusoidal'):
        super().__init__()
        self.max_time = max_time
        self.method = method
        if method == 'onehot':
            self.onehot_embed = OneHotEmbedding(dim)
        elif method == 'sinusoidal':
            self.sinusoid_embed = PositionalEmbedding(dim)
        else:
            raise ValueError(f"Embedding '{method}' not supported")
        self.dim = dim
        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: 'Tensor') ->Tensor:
        t = t / self.max_time
        if self.method == 'onehot':
            emb = self.onehot_embed(t)
        elif self.method == 'sinusoidal':
            emb = self.sinusoid_embed(t)
        return self.mlp(emb)


class ModAFNO(Module):
    """Modulated Adaptive Fourier neural operator (ModAFNO) model.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int, optional
        Number of input channels
    out_channels: int, optional
        Number of output channels
    embed_model: dict, optional
        Dictionary of arguments to pass to the `ModEmbedNet` embedding model
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    mod_dim : int
        Modululation input dimensionality
    modulate_filter: bool, optional
        Whether to compute the modulation for the FFT filter, by default True
    modulate_mlp: bool, optional
        Whether to compute the modulation for the MLP, by default True
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    The default settings correspond to the implementation in the paper cited below.

    Example
    -------
    >>> import torch
    >>> from modulus.models.afno import ModAFNO
    >>> model = ModAFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> time = torch.full((32, 1), 0.5)
    >>> output = model(input, time)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Leinonen et al. "Modulated Adaptive Fourier Neural Operators
    for Temporal Interpolation of Weather Forecasts." arXiv preprint arXiv:TODO (2024).
    """

    def __init__(self, inp_shape: 'List[int]', in_channels: 'int'=155, out_channels: 'int'=73, embed_model: 'Union[dict, None]'=None, patch_size: 'List[int]'=[2, 2], embed_dim: 'int'=512, mod_dim: 'int'=64, modulate_filter: 'bool'=True, modulate_mlp: 'bool'=True, scale_shift_mode: "Literal['complex', 'real']"='complex', depth: 'int'=12, mlp_ratio: 'float'=2.0, drop_rate: 'float'=0.0, num_blocks: 'int'=1, sparsity_threshold: 'float'=0.01, hard_thresholding_fraction: 'float'=1.0) ->None:
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError('inp_shape should be a list of length 2')
        if len(patch_size) != 2:
            raise ValueError('patch_size should be a list of length 2')
        if not (inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0):
            raise ValueError(f'input shape {inp_shape} should be divisible by patch_size {patch_size}')
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.modulate_filter = modulate_filter
        self.modulate_mlp = modulate_mlp
        self.scale_shift_mode = scale_shift_mode
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbed(inp_shape=inp_shape, in_channels=self.in_chans, patch_size=self.patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]
        self.blocks = nn.ModuleList([Block(embed_dim=embed_dim, mod_dim=mod_dim, num_blocks=self.num_blocks, mlp_ratio=mlp_ratio, drop=drop_rate, norm_layer=norm_layer, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction, modulate_filter=modulate_filter, modulate_mlp=modulate_mlp, scale_shift_mode=scale_shift_mode) for i in range(depth)])
        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.mod_additive_proj = nn.Linear(mod_dim, embed_dim)
        if not (modulate_mlp or modulate_filter):
            self.mod_embed_net = nn.Identity()
        else:
            embed_model = {} if embed_model is None else embed_model
            self.mod_embed_net = ModEmbedNet(**embed_model)

    def _init_weights(self, m: 'nn.Module'):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: 'Tensor', mod: 'Tensor') ->Tensor:
        """Forward pass of core ModAFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        mod_embed = self.mod_embed_net(mod)
        mod_additive = self.mod_additive_proj(mod_embed).unsqueeze(dim=1)
        x = x + mod_additive
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x, mod_embed=mod_embed)
        return x

    def forward(self, x: 'Tensor', mod: 'Tensor') ->Tensor:
        """The full ModAFNO model logic."""
        x = self.forward_features(x, mod)
        x = self.head(x)
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        return out


def weight_init(shape: 'tuple', mode: 'str', fan_in: 'int', fan_out: 'int'):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel: 'int', bias: 'bool'=True, up: 'bool'=False, down: 'bool'=False, resample_filter: 'List[int]'=[1, 1], fused_resample: 'bool'=False, init_mode: 'str'='kaiming_normal', init_weight: 'float'=1.0, init_bias: 'float'=0.0):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight if self.weight is not None else None
        b = self.bias if self.bias is not None else None
        f = self.resample_filter if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.

    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(self, num_channels: 'int', num_groups: 'int'=32, min_channels_per_group: 'int'=4, eps: 'float'=1e-05):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if self.training:
            x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)
        else:
            dtype = x.dtype
            x = x.float()
            x = rearrange(x, 'b (g c) h w -> b g c h w', g=self.num_groups)
            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)
            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, 'b g c h w -> b (g c) h w')
            weight = rearrange(self.weight, 'c -> 1 c 1 1')
            bias = rearrange(self.bias, 'c -> 1 c 1 1')
            x = x * weight + bias
            x = x.type(dtype)
        return x


class AttentionOp(torch.autograd.Function):
    """
    Attention weight computation, i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.
    """

    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / torch.sqrt(torch.tensor(k.shape[1]))).to(torch.float32)).softmax(dim=2)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw, output=w, dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional
        If True, applies upsampling in the forward pass. By default False.
    down : bool, optional
        If True, applies downsampling in the forward pass. By default False.
    attention : bool, optional
        If True, enables the self-attention mechanism in the block. By default False.
    num_heads : int, optional
        Number of attention heads. If None, defaults to `out_channels // 64`.
    channels_per_head : int, optional
        Number of channels per attention head. By default 64.
    dropout : float, optional
        Dropout probability. By default 0.0.
    skip_scale : float, optional
        Scale factor applied to skip connections. By default 1.0.
    eps : float, optional
        Epsilon value used for normalization layers. By default 1e-5.
    resample_filter : List[int], optional
        Filter for resampling layers. By default [1, 1].
    resample_proj : bool, optional
        If True, resampling projection is enabled. By default False.
    adaptive_scale : bool, optional
        If True, uses adaptive scaling in the forward pass. By default True.
    init : dict, optional
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional
        Initialization parameters with zero weights for certain layers. By default
        {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', emb_channels: 'int', up: 'bool'=False, down: 'bool'=False, attention: 'bool'=False, num_heads: 'int'=None, channels_per_head: 'int'=64, dropout: 'float'=0.0, skip_scale: 'float'=1.0, eps: 'float'=1e-05, resample_filter: 'List[int]'=[1, 1], resample_proj: 'bool'=False, adaptive_scale: 'bool'=True, init: 'Dict[str, Any]'=dict(), init_zero: 'Dict[str, Any]'=dict(init_weight=0), init_attn: 'Any'=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1, **init_attn if init_attn is not None else init)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        torch.cuda.nvtx.range_push('UNetBlock')
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        torch.cuda.nvtx.range_pop()
        return x


class DhariwalUNet(Module):
    """
    Reimplementation of the ADM architecture, a U-Net variant, with optional
    self-attention.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    Parameters
    -----------
    img_resolution : int
        The resolution of the input/output image.
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels in the output image.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels; 0 means no augmentation. By default 0.
    model_channels : int, optional
        Base multiplier for the number of channels across the network, by default 192.
    channel_mult : List[int], optional
        Per-resolution multipliers for the number of channels. By default [1,2,3,4].
    channel_mult_emb : int, optional
        Multiplier for the dimensionality of the embedding vector. By default 4.
    num_blocks : int, optional
        Number of residual blocks per resolution. By default 3.
    attn_resolutions : List[int], optional
        Resolutions at which self-attention layers are applied. By default [32, 16, 8].
    dropout : float, optional
        Dropout probability applied to intermediate activations. By default 0.10.
    label_dropout : float, optional
       Dropout probability of class labels for classifier-free guidance. By default 0.0.

    Reference
    ----------
    Reference: Dhariwal, P. and Nichol, A., 2021. Diffusion models beat gans on image
    synthesis. Advances in neural information processing systems, 34, pp.8780-8794.

    Note
    -----
    Equivalent to the original implementation by Dhariwal and Nichol, available at
    https://github.com/openai/guided-diffusion

    Example
    --------
    >>> model = DhariwalUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    """

    def __init__(self, img_resolution: 'int', in_channels: 'int', out_channels: 'int', label_dim: 'int'=0, augment_dim: 'int'=0, model_channels: 'int'=192, channel_mult: 'List[int]'=[1, 2, 3, 4], channel_mult_emb: 'int'=4, num_blocks: 'int'=3, attn_resolutions: 'List[int]'=[32, 16, 8], dropout: 'float'=0.1, label_dropout: 'float'=0.0):
        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=res in attn_resolutions, **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=res in attn_resolutions, **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, init_mode: 'str'='kaiming_normal', init_weight: 'int'=1, init_bias: 'int'=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias is not None:
            x = x.add_(self.bias)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    """

    def __init__(self, num_channels: 'int', scale: 'int'=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger(2 * np.pi * self.freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class VPPrecond(Module):
    """
    Preconditioning corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    beta_d : float
        Extent of the noise level schedule, by default 19.9.
    beta_min : float
        Initial slope of the noise level schedule, by default 0.1.
    M : int
        Original number of timesteps in the DDPM formulation, by default 1000.
    epsilon_t : float
        Minimum t-value used during training, by default 1e-5.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(self, img_resolution: 'int', img_channels: 'int', label_dim: 'int'=0, use_fp16: 'bool'=False, beta_d: 'float'=19.9, beta_min: 'float'=0.1, M: 'int'=1000, epsilon_t: 'float'=1e-05, model_type: 'str'='SongUNet', **model_kwargs: dict):
        super().__init__(meta=VPPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        model_class = getattr(network_module, model_type)
        self.model = model_class(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype and not torch.is_autocast_enabled():
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        D_x = c_skip * x + c_out * F_x
        return D_x

    def sigma(self, t: 'Union[float, torch.Tensor]'):
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * t ** 2 + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma: 'Union[float, torch.Tensor]'):
        """
        Compute the inverse of the sigma function for a given sigma.

        This function effectively calculates t from a given sigma(t) based on the
        parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        sigma : Union[float, torch.Tensor]
            The sigma(t) value or set of sigma(t) values for which to compute the
            inverse.

        Returns
        -------
        torch.Tensor
            The computed t value(s) corresponding to the provided sigma(t).
        """
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma: 'Union[float, List, torch.Tensor]'):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


class VEPrecond(Module):
    """
    Preconditioning corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(self, img_resolution: 'int', img_channels: 'int', label_dim: 'int'=0, use_fp16: 'bool'=False, sigma_min: 'float'=0.02, sigma_max: 'float'=100.0, model_type: 'str'='SongUNet', **model_kwargs: dict):
        super().__init__(meta=VEPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        model_class = getattr(network_module, model_type)
        self.model = model_class(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype and not torch.is_autocast_enabled():
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma: 'Union[float, List, torch.Tensor]'):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


class iDDPMPrecond(Module):
    """
    Preconditioning corresponding to the improved DDPM (iDDPM) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    C_1 : float
        Timestep adjustment at low noise levels., by default 0.001.
    C_2 : float
        Timestep adjustment at high noise levels., by default 0.008.
    M: int
        Original number of timesteps in the DDPM formulation, by default 1000.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Nichol, A.Q. and Dhariwal, P., 2021, July. Improved denoising diffusion
    probabilistic models. In International Conference on Machine Learning
    (pp. 8162-8171). PMLR.
    """

    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False, C_1=0.001, C_2=0.008, M=1000, model_type='DhariwalUNet', **model_kwargs):
        super().__init__(meta=iDDPMPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        model_class = getattr(network_module, model_type)
        self.model = model_class(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels * 2, label_dim=label_dim, **model_kwargs)
        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True)
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype and not torch.is_autocast_enabled():
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels]
        return D_x

    def alpha_bar(self, j):
        """
        Compute the alpha_bar(j) value for a given j based on the iDDPM formulation.

        Parameters
        ----------
        j : Union[int, torch.Tensor]
            The timestep or set of timesteps for which to compute alpha_bar(j).

        Returns
        -------
        torch.Tensor
            The computed alpha_bar(j) value(s).
        """
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        """
        Round the provided sigma value(s) to the nearest value(s) in a
        pre-defined set `u`.

        Parameters
        ----------
        sigma : Union[float, list, torch.Tensor]
            The sigma value(s) to round.
        return_index : bool, optional
            Whether to return the index/indices of the rounded value(s) in `u` instead
            of the rounded value(s) themselves, by default False.

        Returns
        -------
        torch.Tensor
            The rounded sigma value(s) or their index/indices in `u`, depending on the
            value of `return_index`.
        """
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()]
        return result.reshape(sigma.shape)


class EDMPrecond(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False, sigma_min=0.0, sigma_max=float('inf'), sigma_data=0.5, model_type='DhariwalUNet', **model_kwargs):
        super().__init__(meta=EDMPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        model_class = getattr(network_module, model_type)
        self.model = model_class(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype and not torch.is_autocast_enabled():
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        D_x = c_skip * x + c_out * F_x
        return D_x

    @staticmethod
    def round_sigma(sigma: 'Union[float, List, torch.Tensor]'):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


class VEPrecond_dfsr(torch.nn.Module):
    """
    Preconditioning for dfsr model, modified from class VEPrecond, where the input
    argument 'sigma' in forward propagation function is used to receive the timestep
    of the backward diffusion process.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models.
    Advances in neural information processing systems. 2020;33:6840-51.
    """

    def __init__(self, img_resolution: 'int', img_channels: 'int', label_dim: 'int'=0, use_fp16: 'bool'=False, sigma_min: 'float'=0.02, sigma_max: 'float'=100.0, dataset_mean: 'float'=5.85e-05, dataset_scale: 'float'=4.79, model_type: 'str'='SongUNet', **model_kwargs: dict):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=self.img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_in = 1
        c_noise = sigma
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype:
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        return F_x


class VEPrecond_dfsr_cond(torch.nn.Module):
    """
    Preconditioning for dfsr model with physics-informed conditioning input, modified
    from class VEPrecond, where the input argument 'sigma' in forward propagation function
    is used to receive the timestep of the backward diffusion process. The gradient of PDE
    residual with respect to the vorticity in the governing Navier-Stokes equation is computed
    as the physics-informed conditioning variable and is combined with the backward diffusion
    timestep before being sent to the underlying model for noise prediction.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference:
    [1] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    [2] Shu D, Li Z, Farimani AB. A physics-informed diffusion model for high-fidelity
    flow field reconstruction. Journal of Computational Physics. 2023 Apr 1;478:111972.
    """

    def __init__(self, img_resolution: 'int', img_channels: 'int', label_dim: 'int'=0, use_fp16: 'bool'=False, sigma_min: 'float'=0.02, sigma_max: 'float'=100.0, dataset_mean: 'float'=5.85e-05, dataset_scale: 'float'=4.79, model_type: 'str'='SongUNet', **model_kwargs: dict):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=model_kwargs['model_channels'] * 2, out_channels=img_channels, label_dim=label_dim, **model_kwargs)
        self.conv_in = torch.nn.Conv2d(img_channels, model_kwargs['model_channels'], kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.emb_conv = torch.nn.Sequential(torch.nn.Conv2d(img_channels, model_kwargs['model_channels'], kernel_size=1, stride=1, padding=0), torch.nn.GELU(), torch.nn.Conv2d(model_kwargs['model_channels'], model_kwargs['model_channels'], kernel_size=3, stride=1, padding=1, padding_mode='circular'))
        self.dataset_mean = dataset_mean
        self.dataset_scale = dataset_scale

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.reshape(-1, self.label_dim)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32
        c_in = 1
        c_noise = sigma
        dx = self.voriticity_residual(x * self.dataset_scale + self.dataset_mean) / self.dataset_scale
        x = self.conv_in(x)
        cond_emb = self.emb_conv(dx)
        x = torch.cat((x, cond_emb), dim=1)
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        if F_x.dtype != dtype:
            raise ValueError(f'Expected the dtype to be {dtype}, but got {F_x.dtype} instead.')
        return F_x

    def voriticity_residual(self, w, re=1000.0, dt=1 / 32):
        """
        Compute the gradient of PDE residual with respect to a given vorticity w using the
        spectrum method.

        Parameters
        ----------
        w: torch.Tensor
            The fluid flow data sample (vorticity).
        re: float
            The value of Reynolds number used in the governing Navier-Stokes equation.
        dt: float
            Time step used to compute the time-derivative of vorticity included in the governing
            Navier-Stokes equation.

        Returns
        -------
        torch.Tensor
            The computed vorticity gradient.
        """
        w = w.clone()
        w.requires_grad_(True)
        nx = w.size(2)
        device = w.device
        w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
        k_max = nx // 2
        N = nx
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1, 1, N, N)
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1, 1, N, N)
        lap = k_x ** 2 + k_y ** 2
        lap[..., 0, 0] = 1.0
        psi_h = w_h / lap
        u_h = 1.0j * k_y * psi_h
        v_h = -1.0j * k_x * psi_h
        wx_h = 1.0j * k_x * w_h
        wy_h = 1.0j * k_y * w_h
        wlap_h = -lap * w_h
        u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
        v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
        wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
        wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
        wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
        advection = u * wx + v * wy
        wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
        x = torch.linspace(0, 2 * np.pi, nx + 1, device=device)
        x = x[0:-1]
        X, Y = torch.meshgrid(x, x)
        f = -4 * torch.cos(4 * Y)
        residual = wt + (advection - 1.0 / re * wlap + 0.1 * w[:, 1:-1]) - f
        residual_loss = (residual ** 2).mean()
        dw = torch.autograd.grad(residual_loss, w)[0]
        return dw


def _get_same_padding(x: 'int', k: 'int', s: 'int') ->int:
    """Function to compute "same" padding. Inspired from:
    https://github.com/huggingface/pytorch-image-models/blob/0.5.x/timm/models/layers/padding.py
    """
    return max(s * math.ceil(x / s) - s - x + k, 0)


def _pad_periodically_equatorial(main_face, left_face, right_face, top_face, bottom_face, nr_rot, size=2):
    if nr_rot != 0:
        top_face = torch.rot90(top_face, k=nr_rot, dims=(-2, -1))
        bottom_face = torch.rot90(bottom_face, k=nr_rot, dims=(-1, -2))
    padded_data_temp = torch.cat((left_face[..., :, -size:], main_face, right_face[..., :, :size]), dim=-1)
    top_pad = torch.cat((top_face[..., :, :size], top_face, top_face[..., :, -size:]), dim=-1)
    bottom_pad = torch.cat((bottom_face[..., :, :size], bottom_face, bottom_face[..., :, -size:]), dim=-1)
    padded_data = torch.cat((bottom_pad[..., -size:, :], padded_data_temp, top_pad[..., :size, :]), dim=-2)
    return padded_data


def _pad_periodically_polar(main_face, left_face, right_face, top_face, bottom_face, rot_axis_left, rot_axis_right, size=2):
    left_face = torch.rot90(left_face, dims=rot_axis_left)
    right_face = torch.rot90(right_face, dims=rot_axis_right)
    padded_data_temp = torch.cat((bottom_face[..., -size:, :], main_face, top_face[..., :size, :]), dim=-2)
    left_pad = torch.cat((left_face[..., :size, :], left_face, left_face[..., -size:, :]), dim=-2)
    right_pad = torch.cat((right_face[..., :size, :], right_face, right_face[..., -size:, :]), dim=-2)
    padded_data = torch.cat((left_pad[..., :, -size:], padded_data_temp, right_pad[..., :, :size]), dim=-1)
    return padded_data


def _cubed_conv_wrapper(faces, equator_conv, polar_conv):
    padding_size = _get_same_padding(x=faces[0].size(-1), k=equator_conv.kernel_size[0], s=equator_conv.stride[0])
    padding_size = padding_size // 2
    output = []
    if padding_size != 0:
        for i in range(6):
            if i == 0:
                x = _pad_periodically_equatorial(faces[0], faces[3], faces[1], faces[5], faces[4], nr_rot=0, size=padding_size)
                output.append(equator_conv(x))
            elif i == 1:
                x = _pad_periodically_equatorial(faces[1], faces[0], faces[2], faces[5], faces[4], nr_rot=1, size=padding_size)
                output.append(equator_conv(x))
            elif i == 2:
                x = _pad_periodically_equatorial(faces[2], faces[1], faces[3], faces[5], faces[4], nr_rot=2, size=padding_size)
                output.append(equator_conv(x))
            elif i == 3:
                x = _pad_periodically_equatorial(faces[3], faces[2], faces[0], faces[5], faces[4], nr_rot=3, size=padding_size)
                output.append(equator_conv(x))
            elif i == 4:
                x = _pad_periodically_polar(faces[4], faces[3], faces[1], faces[0], faces[5], rot_axis_left=(-1, -2), rot_axis_right=(-2, -1), size=padding_size)
                output.append(polar_conv(x))
            else:
                x = _pad_periodically_polar(faces[5], faces[3], faces[1], faces[4], faces[0], rot_axis_left=(-2, -1), rot_axis_right=(-1, -2), size=padding_size)
                x = torch.flip(x, [-1])
                x = polar_conv(x)
                output.append(torch.flip(x, [-1]))
    else:
        for i in range(6):
            if i in [0, 1, 2, 3]:
                output.append(equator_conv(faces[i]))
            elif i == 4:
                output.append(polar_conv(faces[i]))
            else:
                x = torch.flip(faces[i], [-1])
                x = polar_conv(x)
                output.append(torch.flip(x, [-1]))
    return output


def _cubed_non_conv_wrapper(faces, layer):
    output = [layer(faces[i]) for i in range(6)]
    return output


class DLWP(Module):
    """A Convolutional model for Deep Learning Weather Prediction that
    works on Cubed-sphere grids.

    This model expects the input to be of shape [N, C, 6, Res, Res]

    Parameters
    ----------
    nr_input_channels : int
        Number of channels in the input
    nr_output_channels : int
        Number of channels in the output
    nr_initial_channels : int
        Number of channels in the initial convolution. This governs the overall channels
        in the model.
    activation_fn : str
        Activation function for the convolutions
    depth : int
        Depth for the U-Net
    clamp_activation : Tuple of ints, floats or None
        The min and max value used for torch.clamp()

    Example
    -------
    >>> model = modulus.models.dlwp.DLWP(
    ... nr_input_channels=2,
    ... nr_output_channels=4,
    ... )
    >>> input = torch.randn(4, 2, 6, 64, 64) # [N, C, F, Res, Res]
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 4, 6, 64, 64])

    Note
    ----
    Reference: Weyn, Jonathan A., et al. "Sub‐seasonal forecasting with a large ensemble
     of deep‐learning weather prediction models." Journal of Advances in Modeling Earth
     Systems 13.7 (2021): e2021MS002502.
    """

    def __init__(self, nr_input_channels: 'int', nr_output_channels: 'int', nr_initial_channels: 'int'=64, activation_fn: 'str'='leaky_relu', depth: 'int'=2, clamp_activation: 'Tuple[Union[float, int, None], Union[float, int, None]]'=(None, 10.0)):
        super().__init__(meta=MetaData())
        self.nr_input_channels = nr_input_channels
        self.nr_output_channels = nr_output_channels
        self.nr_initial_channels = nr_initial_channels
        self.activation_fn = get_activation(activation_fn)
        self.depth = depth
        self.clamp_activation = clamp_activation
        self.avg_pool = nn.AvgPool2d(2)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.equatorial_downsample = []
        self.equatorial_upsample = []
        self.equatorial_mid_layers = []
        self.polar_downsample = []
        self.polar_upsample = []
        self.polar_mid_layers = []
        for i in range(depth):
            if i == 0:
                ins = self.nr_input_channels
            else:
                ins = self.nr_initial_channels * 2 ** (i - 1)
            outs = self.nr_initial_channels * 2 ** i
            self.equatorial_downsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_downsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.equatorial_downsample.append(nn.Conv2d(outs, outs, kernel_size=3))
            self.polar_downsample.append(nn.Conv2d(outs, outs, kernel_size=3))
        for i in range(2):
            if i == 0:
                ins = outs
                outs = ins * 2
            else:
                ins = outs
                outs = ins // 2
            self.equatorial_mid_layers.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_mid_layers.append(nn.Conv2d(ins, outs, kernel_size=3))
        for i in range(depth - 1, -1, -1):
            if i == 0:
                outs = self.nr_initial_channels
                outs_final = outs
            else:
                outs = self.nr_initial_channels * 2 ** i
                outs_final = outs // 2
            ins = outs * 2
            self.equatorial_upsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_upsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.equatorial_upsample.append(nn.Conv2d(outs, outs_final, kernel_size=3))
            self.polar_upsample.append(nn.Conv2d(outs, outs_final, kernel_size=3))
        self.equatorial_downsample = nn.ModuleList(self.equatorial_downsample)
        self.polar_downsample = nn.ModuleList(self.polar_downsample)
        self.equatorial_mid_layers = nn.ModuleList(self.equatorial_mid_layers)
        self.polar_mid_layers = nn.ModuleList(self.polar_mid_layers)
        self.equatorial_upsample = nn.ModuleList(self.equatorial_upsample)
        self.polar_upsample = nn.ModuleList(self.polar_upsample)
        self.equatorial_last = nn.Conv2d(outs, self.nr_output_channels, kernel_size=1)
        self.polar_last = nn.Conv2d(outs, self.nr_output_channels, kernel_size=1)

    def activation(self, x: 'Tensor'):
        x = self.activation_fn(x)
        if any(isinstance(c, (float, int)) for c in self.clamp_activation):
            x = torch.clamp(x, min=self.clamp_activation[0], max=self.clamp_activation[1])
        return x

    def forward(self, cubed_sphere_input):
        if cubed_sphere_input.size(-3) != 6:
            raise ValueError('The input must have 6 faces.')
        if cubed_sphere_input.size(-2) != cubed_sphere_input.size(-1):
            raise ValueError('The input must have equal height and width')
        faces = torch.split(cubed_sphere_input, split_size_or_sections=1, dim=2)
        faces = [torch.squeeze(face, dim=2) for face in faces]
        encoder_states = []
        for i, (equatorial_layer, polar_layer) in enumerate(zip(self.equatorial_downsample, self.polar_downsample)):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)
            if i % 2 != 0:
                encoder_states.append(faces)
                faces = _cubed_non_conv_wrapper(faces, self.avg_pool)
        for i, (equatorial_layer, polar_layer) in enumerate(zip(self.equatorial_mid_layers, self.polar_mid_layers)):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)
        j = 0
        for i, (equatorial_layer, polar_layer) in enumerate(zip(self.equatorial_upsample, self.polar_upsample)):
            if i % 2 == 0:
                encoder_faces = encoder_states[len(encoder_states) - j - 1]
                faces = _cubed_non_conv_wrapper(faces, self.upsample_layer)
                faces = [torch.cat((face_1, face_2), dim=1) for face_1, face_2 in zip(faces, encoder_faces)]
                j += 1
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)
        faces = _cubed_conv_wrapper(faces, self.equatorial_last, self.polar_last)
        output = torch.stack(faces, dim=2)
        return output


class HEALPixFoldFaces(th.nn.Module):
    """Class that folds the faces of a HealPIX tensor"""

    def __init__(self, enable_nhwc: 'bool'=False):
        """
        Parameters
        ----------
        enable_nhwc: bool, optional
            Use nhwc format instead of nchw format
        """
        super().__init__()
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass that folds a HEALPix tensor
        [B, F, C, H, W] -> [B*F, C, H, W]

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to fold

        Returns
        -------
        torch.Tensor
            the folded tensor

        """
        N, F, C, H, W = tensor.shape
        tensor = torch.reshape(tensor, shape=(N * F, C, H, W))
        if self.enable_nhwc:
            tensor = tensor
        return tensor


class HEALPixUnfoldFaces(th.nn.Module):
    """Class that unfolds the faces of a HealPIX tensor"""

    def __init__(self, num_faces=12, enable_nhwc=False):
        """
        Parameters
        ----------
        num_faces: int, optional
            The number of faces on the grid, default 12
        enable_nhwc: bool, optional
            If nhwc format is being used, default False
        """
        super().__init__()
        self.num_faces = num_faces
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass that unfolds a HEALPix tensor
        [B*F, C, H, W] -> [B, F, C, H, W]

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to unfold

        Returns
        -------
        torch.Tensor
            The unfolded tensor

        """
        NF, C, H, W = tensor.shape
        tensor = torch.reshape(tensor, shape=(-1, self.num_faces, C, H, W))
        return tensor


class HEALPixRecUNet(Module):
    """Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh."""

    def __init__(self, encoder: 'DictConfig', decoder: 'DictConfig', input_channels: 'int', output_channels: 'int', n_constants: 'int', decoder_input_channels: 'int', input_time_dim: 'int', output_time_dim: 'int', delta_time: 'str'='6h', reset_cycle: 'str'='24h', presteps: 'int'=1, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False, couplings: 'list'=[]):
        """
        Parameters
        ----------
        encoder: DictConfig
            dictionary of instantiable parameters for the U-net encoder
        decoder: DictConfig
            dictionary of instantiable parameters for the U-net decoder
        input_channels: int
            number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        output_channels: int
            number of output channels expected in the output array schema, or output variables
        n_constants: int
            number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        decoder_input_channels: int
            number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        delta_time: str, optional
            hours between two consecutive data points
        reset_cycle: str, optional
            hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        presteps: int, optional
            number of model steps to initialize recurrent states.
        enable_nhwc: bool, optional
            Model with [N, H, W, C] instead of [N, C, H, W]
        enable_healpixpad: bool, optional
            Enable CUDA HEALPixPadding if installed
        couplings: list, optional
            sequence of dictionaries that describe coupling mechanisms
        """
        super().__init__()
        self.channel_dim = 2
        self.input_channels = input_channels
        if n_constants == 0 and decoder_input_channels == 0:
            raise NotImplementedError('support for models with no constant fields and no decoder inputs (TOA insolation) is not available at this time.')
        if len(couplings) > 0:
            if n_constants == 0:
                raise NotImplementedError('support for coupled models with no constant fields is not available at this time.')
            if decoder_input_channels == 0:
                raise NotImplementedError('support for coupled models with no decoder inputs (TOA insolation) is not available at this time.')
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.delta_t = int(pd.Timedelta(delta_time).total_seconds() // 3600)
        self.reset_cycle = int(pd.Timedelta(reset_cycle).total_seconds() // 3600)
        self.presteps = presteps
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and self.output_time_dim % self.input_time_dim != 0:
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got {self.output_time_dim} and {self.input_time_dim})")
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.encoder = instantiate(config=encoder, input_channels=self._compute_input_channels(), enable_nhwc=self.enable_nhwc, enable_healpixpad=self.enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder, output_channels=self._compute_output_channels(), enable_nhwc=self.enable_nhwc, enable_healpixpad=self.enable_healpixpad)

    @property
    def integration_steps(self):
        """Number of integration steps"""
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) ->int:
        """Calculate total number of input channels in the model"""
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants + self.coupled_channels

    def _compute_coupled_channels(self, couplings):
        """Get number of coupled channels

        Returns
        -------
        int
            The number of coupled channels
        """
        c_channels = 0
        for c in couplings:
            c_channels += len(c['params']['variables']) * len(c['params']['input_times'])
        return c_channels

    def _compute_output_channels(self) ->int:
        """Compute the total number of output channels in the model"""
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: 'Sequence', step: 'int'=0) ->th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model encoder
        """
        if len(self.couplings) > 0:
            result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])), inputs[3].permute(0, 2, 1, 3, 4)]
            res = th.cat(result, dim=self.channel_dim)
        else:
            if self.n_constants == 0:
                result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1)]
                res = th.cat(result, dim=self.channel_dim)
                res = self.fold(res)
                return res
            if self.decoder_input_channels == 0:
                result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1]))]
                res = th.cat(result, dim=self.channel_dim)
                res = self.fold(res)
                return res
            result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))]
            res = th.cat(result, dim=self.channel_dim)
        res = self.fold(res)
        return res

    def _reshape_outputs(self, outputs: 'th.Tensor') ->th.Tensor:
        """Returns a maultiple tensors to from the model decoder.
        Splits the time/channel dimensions.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model outputs
        """
        outputs = self.unfold(outputs)
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        return res

    def _initialize_hidden(self, inputs: 'Sequence', outputs: 'Sequence', step: 'int') ->None:
        """Initialize the hidden layers

        Parameters
        ----------
        inputs: Sequence
            Inputs to use to initialize the hideen layers
        outputs: Sequence
            Outputs to use to initialize the hideen layers
        step: int
            Current step number of the initialization
        """
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(inputs=[inputs[0][:, :, s * self.input_time_dim:(s + 1) * self.input_time_dim]] + list(inputs[1:3]) + [inputs[3][prestep]], step=step + prestep)
                else:
                    input_tensor = self._reshape_inputs(inputs=[inputs[0][:, :, s * self.input_time_dim:(s + 1) * self.input_time_dim]] + list(inputs[1:]), step=step + prestep)
            else:
                s = step - self.presteps + prestep
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(inputs=[outputs[s - 1]] + list(inputs[1:3]) + [inputs[3][step - (prestep - self.presteps)]], step=s + 1)
                else:
                    input_tensor = self._reshape_inputs(inputs=[outputs[s - 1]] + list(inputs[1:]), step=s + 1)
            self.decoder(self.encoder(input_tensor))

    def forward(self, inputs: 'Sequence', output_only_last=False) ->th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Parameters
        ----------
        inputs: Sequence
            Inputs to the model, of the form [prognostics|TISR|constants]
            [B, F, T, C, H, W] is the format for prognostics and TISR
            [F, C, H, W] is the format for constants
        output_only_last: bool, optional
            If only the last dimension of the outputs should be returned

        Returns
        -------
        th.Tensor: Predicted outputs
        """
        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            if step * (self.delta_t * self.input_time_dim) % self.reset_cycle == 0:
                self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)
            if step == 0:
                s = self.presteps
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(inputs=[inputs[0][:, :, s * self.input_time_dim:(s + 1) * self.input_time_dim]] + list(inputs[1:3]) + [inputs[3][s]], step=s)
                else:
                    input_tensor = self._reshape_inputs(inputs=[inputs[0][:, :, s * self.input_time_dim:(s + 1) * self.input_time_dim]] + list(inputs[1:]), step=s)
            elif len(self.couplings) > 0:
                input_tensor = self._reshape_inputs(inputs=[outputs[-1]] + list(inputs[1:3]) + [inputs[3][self.presteps + step]], step=step + self.presteps)
            else:
                input_tensor = self._reshape_inputs(inputs=[outputs[-1]] + list(inputs[1:]), step=step + self.presteps)
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels * self.input_time_dim] + decodings)
            outputs.append(reshaped)
        if output_only_last:
            return outputs[-1]
        return th.cat(outputs, dim=self.channel_dim)

    def reset(self):
        """Resets the state of the network"""
        self.encoder.reset()
        self.decoder.reset()


class HEALPixUNet(Module):
    """Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh."""

    def __init__(self, encoder: 'DictConfig', decoder: 'DictConfig', input_channels: 'int', output_channels: 'int', n_constants: 'int', decoder_input_channels: 'int', input_time_dim: 'int', output_time_dim: 'int', presteps: 'int'=0, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False, couplings: 'list'=[]):
        """
        Parameters
        ----------
        encoder: DictConfig
            dictionary of instantiable parameters for the U-net encoder
        decoder: DictConfig
            dictionary of instantiable parameters for the U-net decoder
        input_channels: int
            number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        output_channels: int
            number of output channels expected in the output array schema, or output variables
        n_constants: int
            number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        decoder_input_channels: int
            number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        input_time_dim: int
            number of time steps in the input array
        output_time_dim: int
            number of time steps in the output array
        presteps: int, optional
            number of model steps to initialize recurrent states. default: 0
        enable_nhwc: bool, optional
            Model with [N, H, W, C] instead of [N, C, H, W]. default: False
        enable_healpixpad: bool, optional
            Enable CUDA HEALPixPadding if installed. default: False
        couplings: list, optional
            sequence of dictionaries that describe coupling mechanisms
        """
        super().__init__()
        if n_constants == 0 and decoder_input_channels == 0:
            raise NotImplementedError('support for models with no constant fields and no decoder inputs (TOA insolation) is not available at this time.')
        if len(couplings) > 0:
            if n_constants == 0:
                raise NotImplementedError('support for coupled models with no constant fields is not available at this time.')
            if decoder_input_channels == 0:
                raise NotImplementedError('support for coupled models with no decoder inputs (TOA insolation) is not available at this time.')
        self.coupled_channels = self._compute_coupled_channels(couplings)
        self.couplings = couplings
        self.train_couplers = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and self.output_time_dim % self.input_time_dim != 0:
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got {self.output_time_dim} and {self.input_time_dim})")
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.encoder = instantiate(config=encoder, input_channels=self._compute_input_channels(), enable_nhwc=self.enable_nhwc, enable_healpixpad=self.enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder, output_channels=self._compute_output_channels(), enable_nhwc=self.enable_nhwc, enable_healpixpad=self.enable_healpixpad)

    @property
    def integration_steps(self):
        """Number of integration steps"""
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) ->int:
        """Calculate total number of input channels in the model"""
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants + self.coupled_channels

    def _compute_coupled_channels(self, couplings):
        c_channels = 0
        for c in couplings:
            c_channels += len(c['params']['variables']) * len(c['params']['input_times'])
        return c_channels

    def _compute_output_channels(self) ->int:
        """Compute the total number of output channels in the model"""
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: 'Sequence', step: 'int'=0) ->th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_stepsi. default: 0

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model encoder
        """
        if len(self.couplings) > 0:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                raise NotImplementedError('support for coupled models with no constant fields or decoder inputs (TOA insolation) is not available at this time.')
            if self.n_constants == 0:
                raise NotImplementedError('support for coupled models with no constant fields or decoder inputs (TOA insolation) is not available at this time.')
            if self.decoder_input_channels == 0:
                raise NotImplementedError('support for coupled models with no constant fields is not available at this time.')
            result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1])), inputs[3].permute(0, 2, 1, 3, 4)]
            res = th.cat(result, dim=self.channel_dim)
        else:
            if not (self.n_constants > 0 or self.decoder_input_channels > 0):
                return inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1)
            if self.n_constants == 0:
                result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim + 1)]
                res = th.cat(result, dim=self.channel_dim)
                res = self.fold(res)
                return res
            if self.decoder_input_channels == 0:
                result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape) * [-1]))]
                res = th.cat(result, dim=self.channel_dim)
                res = self.fold(res)
                return res
            result = [inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim + 1), inputs[1][:, :, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim + 1), inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))]
            res = th.cat(result, dim=self.channel_dim)
        res = self.fold(res)
        return res

    def _reshape_outputs(self, outputs: 'th.Tensor') ->th.Tensor:
        """Returns a maultiple tensors to from the model decoder.
        Splits the time/channel dimensions.

        Parameters
        ----------
        inputs: Sequence
            list of expected input tensors (inputs, decoder_inputs, constants)
        step: int, optional
            step number in the sequence of integration_steps

        Returns
        -------
        torch.Tensor: reshaped Tensor in expected shape for model outputs
        """
        outputs = self.unfold(outputs)
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        return res

    def forward(self, inputs: 'Sequence', output_only_last=False) ->th.Tensor:
        """
        Forward pass of the HEALPixUnet

        Parameters
        ----------
        inputs: Sequence
            Inputs to the model, of the form [prognostics|TISR|constants]
            [B, F, T, C, H, W] is the format for prognostics and TISR
            [F, C, H, W] is the format for constants
        output_only_last: bool, optional
            If only the last dimension of the outputs should be returned. default: False

        Returns
        -------
        th.Tensor: Predicted outputs
        """
        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                if len(self.couplings) > 0:
                    input_tensor = self._reshape_inputs(list(inputs[0:3]) + [inputs[3][step]], step)
                else:
                    input_tensor = self._reshape_inputs(inputs, step)
            elif len(self.couplings) > 0:
                input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:3]) + [inputs[3][step]], step)
            else:
                input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:]), step)
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            reshaped = self._reshape_outputs(decodings)
            outputs.append(reshaped)
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)
        return res


class HEALPixPadding(th.nn.Module):
    """
    Padding layer for data on a HEALPix sphere. The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined above.
    """

    def __init__(self, padding: 'int', enable_nhwc: 'bool'=False):
        """
        Parameters
        ----------
        padding: int
            The padding size
        enable_nhwc: bool, optional
            If nhwc format is being used, default False
        """
        super().__init__()
        self.p = padding
        self.d = [-2, -1]
        self.enable_nhwc = enable_nhwc
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(f"invalid value for 'padding', expected int > 0 but got {padding}")
        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=self.enable_nhwc)

    def forward(self, data: 'th.Tensor') ->th.Tensor:
        """
        Pad each face consistently with its according neighbors in the HEALPix (see ordering and neighborhoods above).
        Assumes the Tensor is folded

        Parmaters
        ---------
        data: torch.Tensor
            The input tensor of shape [..., F, H, W] where each face is to be padded in its HPX context

        Returns
        -------
        torch.Tensor
            The padded tensor where each face's height and width are increased by 2*p
        """
        torch.cuda.nvtx.range_push('HEALPixPadding:forward')
        data = self.unfold(data)
        f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [torch.squeeze(x, dim=1) for x in th.split(tensor=data, split_size_or_sections=1, dim=1)]
        p00 = self.pn(c=f00, t=f01, tl=f02, lft=f03, bl=f03, b=f04, br=f08, rgt=f05, tr=f01)
        p01 = self.pn(c=f01, t=f02, tl=f03, lft=f00, bl=f00, b=f05, br=f09, rgt=f06, tr=f02)
        p02 = self.pn(c=f02, t=f03, tl=f00, lft=f01, bl=f01, b=f06, br=f10, rgt=f07, tr=f03)
        p03 = self.pn(c=f03, t=f00, tl=f01, lft=f02, bl=f02, b=f07, br=f11, rgt=f04, tr=f00)
        p04 = self.pe(c=f04, t=f00, tl=self.tl(f00, f03), lft=f03, bl=f07, b=f11, br=self.br(f11, f08), rgt=f08, tr=f05)
        p05 = self.pe(c=f05, t=f01, tl=self.tl(f01, f00), lft=f00, bl=f04, b=f08, br=self.br(f08, f09), rgt=f09, tr=f06)
        p06 = self.pe(c=f06, t=f02, tl=self.tl(f02, f01), lft=f01, bl=f05, b=f09, br=self.br(f09, f10), rgt=f10, tr=f07)
        p07 = self.pe(c=f07, t=f03, tl=self.tl(f03, f02), lft=f02, bl=f06, b=f10, br=self.br(f10, f11), rgt=f11, tr=f04)
        p08 = self.ps(c=f08, t=f05, tl=f00, lft=f04, bl=f11, b=f11, br=f10, rgt=f09, tr=f09)
        p09 = self.ps(c=f09, t=f06, tl=f01, lft=f05, bl=f08, b=f08, br=f11, rgt=f10, tr=f10)
        p10 = self.ps(c=f10, t=f07, tl=f02, lft=f06, bl=f09, b=f09, br=f08, rgt=f11, tr=f11)
        p11 = self.ps(c=f11, t=f04, tl=f03, lft=f07, bl=f10, b=f10, br=f09, rgt=f08, tr=f08)
        res = th.stack((p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1)
        res = self.fold(res)
        torch.cuda.nvtx.range_pop()
        return res

    def pn(self, c: 'th.Tensor', t: 'th.Tensor', tl: 'th.Tensor', lft: 'th.Tensor', bl: 'th.Tensor', b: 'th.Tensor', br: 'th.Tensor', rgt: 'th.Tensor', tr: 'th.Tensor') ->th.Tensor:
        """
        Applies padding to a northern hemisphere face c under consideration of its given neighbors.

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor:
            The padded tensor p
        """
        p = self.p
        d = self.d
        c = th.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = th.cat((tl.rot90(2, d)[..., -p:, -p:], lft.rot90(-1, d)[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)
        return th.cat((left, c, right), dim=-1)

    def pe(self, c: 'th.Tensor', t: 'th.Tensor', tl: 'th.Tensor', lft: 'th.Tensor', bl: 'th.Tensor', b: 'th.Tensor', br: 'th.Tensor', rgt: 'th.Tensor', tr: 'th.Tensor') ->th.Tensor:
        """
        Applies padding to an equatorial face c under consideration of its given neighbors.

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor:
            The padded tensor p
        """
        p = self.p
        c = th.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)
        return th.cat((left, c, right), dim=-1)

    def ps(self, c: 'th.Tensor', t: 'th.Tensor', tl: 'th.Tensor', lft: 'th.Tensor', bl: 'th.Tensor', b: 'th.Tensor', br: 'th.Tensor', rgt: 'th.Tensor', tr: 'th.Tensor') ->th.Tensor:
        """
        Applies padding to a southern hemisphere face c under consideration of its given neighbors.

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor:
            The padded tensor p
        """
        p = self.p
        d = self.d
        c = th.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]), dim=-2)
        return th.cat((left, c, right), dim=-1)

    def tl(self, top: 'th.Tensor', lft: 'th.Tensor') ->th.Tensor:
        """
        Assembles the top left corner of a center face in the cases where no according top left face is defined on the
        HPX.

        Parameters
        ----------
        top: torch.Tensor
            The face above the center face
        lft: torch.Tensor
            The face left of the center face

        Returns
        -------
            The assembled top left corner (only the sub-part that is required for padding)
        """
        ret = th.zeros_like(top)[..., :self.p, :self.p]
        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]
        for i in range(1, self.p):
            ret[..., -i - 1, -i:] = top[..., -i - 1, :i]
            ret[..., -i:, -i - 1] = lft[..., :i, -i - 1]
            ret[..., -i - 1, -i - 1] = 0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
        return ret

    def br(self, b: 'th.Tensor', r: 'th.Tensor') ->th.Tensor:
        """
        Assembles the bottom right corner of a center face in the cases where no according bottom right face is defined
        on the HPX.

        Parameters
        ----------
        b: torch.Tensor
            The face below the center face
        r: torch.Tensor
            The face right of the center face

        Returns
        -------
        torch.Tensor
            The assembled bottom right corner (only the sub-part that is required for padding)
        """
        ret = th.zeros_like(b)[..., :self.p, :self.p]
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]
        return ret


class HEALPixPaddingv2(th.nn.Module):
    """
    Padding layer for data on a HEALPix sphere. This version uses a faster method to calculate the padding.
    The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined above.

    TODO: Missing library to use this class. Need to see if we can get it, if not needs to be removed
    """

    def __init__(self, padding: 'int'):
        """
        Parameters
        ----------
        padding: int
            The padding size
        """
        super().__init__()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.fold = HEALPixFoldFaces()
        self.padding = HEALPixPad(padding=padding)

    def forward(self, x):
        """
        Pad each face consistently with its according neighbors in the HEALPix (see ordering and neighborhoods above).
        Assumes the Tensor is folded

        Parmaters
        ---------
        data: torch.Tensor
            The input tensor of shape [..., F, H, W] where each face is to be padded in its HPX context

        Returns
        -------
        torch.Tensor
            The padded tensor where each face's height and width are increased by 2*p
        """
        torch.cuda.nvtx.range_push('HEALPixPaddingv2:forward')
        x = self.unfold(x)
        xp = self.padding(x)
        xp = self.fold(xp)
        torch.cuda.nvtx.range_pop()
        return xp


have_healpixpad = True


class HEALPixLayer(th.nn.Module):
    """Pytorch module for applying any base torch Module on a HEALPix tensor. Expects all input/output tensors to have a
    shape [..., 12, H, W], where 12 is the dimension of the faces.
    """

    def __init__(self, layer, **kwargs):
        """
        Parameters
        ----------
        layer: torch.nn.Module
            Any torch layer function, e.g., th.nn.Conv2d
        kwargs:
            The arguments that are passed to the torch layer function, e.g., kernel_size
        """
        super().__init__()
        layers = []
        if 'enable_nhwc' in kwargs:
            enable_nhwc = kwargs['enable_nhwc']
            del kwargs['enable_nhwc']
        else:
            enable_nhwc = False
        if 'enable_healpixpad' in kwargs:
            enable_healpixpad = kwargs['enable_healpixpad']
            del kwargs['enable_healpixpad']
        else:
            enable_healpixpad = False
        if layer.__bases__[0] is th.nn.modules.conv._ConvNd and kwargs['kernel_size'] > 1:
            kwargs['padding'] = 0
            kernel_size = 3 if 'kernel_size' not in kwargs else kwargs['kernel_size']
            dilation = 1 if 'dilation' not in kwargs else kwargs['dilation']
            padding = (kernel_size - 1) // 2 * dilation
            if enable_healpixpad and have_healpixpad and th.cuda.is_available() and not enable_nhwc:
                layers.append(HEALPixPaddingv2(padding=padding))
            else:
                layers.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))
        layers.append(layer(**kwargs))
        self.layers = th.nn.Sequential(*layers)
        if enable_nhwc:
            self.layers = self.layers

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        """
        Performs the forward pass using the defined layer function and the given data.

        :param x: The input tensor of shape [..., F=12, H, W]
        :return: The output tensor of this HEALPix layer
        """
        res = self.layers(x)
        return res


class ConvGRUBlock(th.nn.Module):
    """Class that implements a Convolutional GRU
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, kernel_size: 'int'=1, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.channels = in_channels
        self.conv_gates = geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels + self.channels, out_channels=2 * self.channels, kernel_size=kernel_size, padding='same', enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        self.conv_can = geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels + self.channels, out_channels=self.channels, kernel_size=kernel_size, padding='same', enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: 'Sequence') ->Sequence:
        """Forward pass of the ConvGRUBlock

        Parameters
        ----------
        inputs: Sequence
            Input to the forward pass

        Returns
        -------
        Sequence
            Result of the forward pass
        """
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)
        combined = th.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)
        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next
        return inputs + h_next

    def reset(self):
        """Reset the update gates"""
        self.h = th.zeros_like(self.h)


class BasicConvBlock(th.nn.Module):
    """Convolution block consisting of n subsequent convolutions and activations"""

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, out_channels: 'int'=1, kernel_size: 'int'=3, dilation: 'int'=1, n_layers: 'int'=1, latent_channels: 'int'=None, activation: 'th.nn.Module'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        n_layers:
            Number of convolutional layers
        latent_channels:
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels if n == 0 else latent_channels, out_channels=out_channels if n == n_layers - 1 else latent_channels, kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
            if activation is not None:
                convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the BasicConvBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.convblock(x)


class ConvNeXtBlock(th.nn.Module):
    """Class implementing a modified ConvNeXt network as described in https://arxiv.org/pdf/2201.03545.pdf
    and shown in figure 4
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, latent_channels: 'int'=1, out_channels: 'int'=1, kernel_size: 'int'=3, dilation: 'int'=1, n_layers: 'int'=1, upscale_factor: 'int'=4, activation: 'th.nn.Module'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        latent_channels: int, optional
            Number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if in_channels == out_channels:
            self.skip_module = lambda x: x
        else:
            self.skip_module = geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=out_channels, kernel_size=1, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        convblock = []
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=int(latent_channels * upscale_factor), kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels * upscale_factor), out_channels=int(latent_channels * upscale_factor), kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels * upscale_factor), out_channels=out_channels, kernel_size=1, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the ConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.skip_module(x) + self.convblock(x)


class DoubleConvNeXtBlock(th.nn.Module):
    """Modification of ConvNeXtBlock block this time putting two sequentially
    in a single block with the number of channels in the middle being the
    number of latent channels
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, out_channels: 'int'=1, kernel_size: 'int'=3, dilation: 'int'=1, n_layers: 'int'=1, upscale_factor: 'int'=4, latent_channels: 'int'=1, activation: 'th.nn.Module'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters:
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if in_channels == int(latent_channels):
            self.skip_module1 = lambda x: x
        else:
            self.skip_module1 = geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=int(latent_channels), kernel_size=1, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        if out_channels == int(latent_channels):
            self.skip_module2 = lambda x: x
        else:
            self.skip_module2 = geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=out_channels, kernel_size=1, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        convblock1 = []
        convblock1.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=int(latent_channels), kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock1.append(activation)
        convblock1.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=int(latent_channels * upscale_factor), kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock1.append(activation)
        convblock1.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels * upscale_factor), out_channels=int(latent_channels), kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock1.append(activation)
        self.convblock1 = th.nn.Sequential(*convblock1)
        convblock2 = []
        convblock2.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=int(latent_channels), kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock2.append(activation)
        convblock2.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=int(latent_channels * upscale_factor), kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock2.append(activation)
        convblock2.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels * upscale_factor), out_channels=out_channels, kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock2.append(activation)
        self.convblock2 = th.nn.Sequential(*convblock2)

    def forward(self, x):
        """Forward pass of the DoubleConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        x1 = self.skip_module1(x) + self.convblock1(x)
        return self.skip_module2(x1) + self.convblock2(x1)


class SymmetricConvNeXtBlock(th.nn.Module):
    """Another modification of ConvNeXtBlock block this time using 4 layers and adding
    a layer that instead of going from in_channels to latent*upscale channesl goes to
    latent channels first
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, latent_channels: 'int'=1, out_channels: 'int'=1, kernel_size: 'int'=3, dilation: 'int'=1, n_layers: 'int'=1, upscale_factor: 'int'=4, activation: 'th.nn.Module'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to torch.nn.Conv2d
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: torch.nn.Module, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if in_channels == int(latent_channels):
            self.skip_module = lambda x: x
        else:
            self.skip_module = geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=out_channels, kernel_size=1, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
        convblock = []
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=in_channels, out_channels=int(latent_channels), kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=int(latent_channels * upscale_factor), kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels * upscale_factor), out_channels=int(latent_channels), kernel_size=1, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        convblock.append(geometry_layer(layer=torch.nn.Conv2d, in_channels=int(latent_channels), out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the SymmetricConvNextBlock

        Parameters
        ----------
        x: torch.Tensor
            inputs to the forward pass

        Returns
        -------
        torch.Tensor
            result of the forward pass
        """
        return self.skip_module(x) + self.convblock(x)


class MaxPool(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.MaxPool2d class.
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, pooling: 'int'=2, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.maxpool = geometry_layer(layer=torch.nn.MaxPool2d, kernel_size=pooling, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)

    def forward(self, x):
        """Forward pass of the MaxPool

        Parameters
        ----------
        x: torch.Tensor
            The values to MaxPool

        Returns
        -------
        torch.Tensor
            The MaxPooled values
        """
        return self.maxpool(x)


class AvgPool(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.AvgPool2d class.
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, pooling: 'int'=2, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.avgpool = geometry_layer(layer=torch.nn.AvgPool2d, kernel_size=pooling, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)

    def forward(self, x):
        """Forward pass of the AvgPool layer

        Parameters
        ----------
        x: torch.Tensor
            The values to average

        Returns
        -------
        torch.Tensor
            The averaged values
        """
        return self.avgpool(x)


class TransposedConvUpsample(th.nn.Module):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the torch.nn.ConvTranspose2d class.
    """

    def __init__(self, geometry_layer: 'th.nn.Module'=HEALPixLayer, in_channels: 'int'=3, out_channels: 'int'=1, upsampling: 'int'=2, activation: 'th.nn.Module'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        geometry_layer: torch.nn.Module, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2d
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        upsampling: int, optional
            Size used for upsampling
        activation: torch.nn.Module, optional
            Activation function used in upsampling
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        upsampler = []
        upsampler.append(geometry_layer(layer=torch.nn.ConvTranspose2d, in_channels=in_channels, out_channels=out_channels, kernel_size=upsampling, stride=upsampling, padding=0, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the TransposedConvUpsample layer

        Parameters
        ----------
        x: torch.Tensor
            The values to upsample

        Returns
        -------
        torch.Tensor
            The upsampled values
        """
        return self.upsampler(x)


class Interpolate(th.nn.Module):
    """Helper class that handles interpolation
    This is done as a class so that scale and mode can be stored
    """

    def __init__(self, scale_factor: 'Union[int, Tuple]', mode: 'str'='nearest'):
        """
        Parameters:
        ----------
        scale_factor: Union[int , Tuple]
            Multiplier for spatial size, passed to torch.nn.functional.interpolate
        mode: str, optional
            Interpolation mode used for upsampling, passed to torch.nn.functional.interpolate
        """
        super().__init__()
        self.interp = th.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        """Forward pass of the Interpolate layer

        Parameters
        ----------
        x: torch.Tensor
            inputs to interpolate

        Returns
        -------
        torch.Tensor
            the interpolated values
        """
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)


class UNetDecoder(th.nn.Module):
    """Generic UNetDecoder that can be applied to arbitrary meshes."""

    def __init__(self, conv_block: 'DictConfig', up_sampling_block: 'DictConfig', output_layer: 'DictConfig', recurrent_block: 'DictConfig'=None, n_channels: 'Sequence'=(64, 32, 16), n_layers: 'Sequence'=(1, 2, 2), output_channels: 'int'=1, dilations: 'list'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        up_sampling_blockoder: DictConfig
            dictionary of instantiable parameters for the upsampling block
        output_layer: DictConfig
            dictionary of instantiable parameters for the output layer
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        n_channels: Sequence, optional
            The number of channels in each decoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        output_channels: int, optional
            Number of output channels
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used if installed
        """
        super().__init__()
        self.channel_dim = 1
        if dilations is None:
            dilations = [(1) for _ in range(len(n_channels))]
        self.decoder = []
        for n, curr_channel in enumerate(n_channels):
            if n == 0:
                up_sample_module = None
            else:
                up_sample_module = instantiate(config=up_sampling_block, in_channels=curr_channel, out_channels=curr_channel, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
            next_channel = n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            conv_module = instantiate(config=conv_block, in_channels=curr_channel * 2 if n > 0 else curr_channel, latent_channels=curr_channel, out_channels=next_channel, dilation=dilations[n], n_layers=n_layers[n], enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)
            if recurrent_block is not None:
                rec_module = instantiate(config=recurrent_block, in_channels=next_channel, enable_healpixpad=enable_healpixpad)
            else:
                rec_module = None
            self.decoder.append(th.nn.ModuleDict({'upsamp': up_sample_module, 'conv': conv_module, 'recurrent': rec_module}))
        self.decoder = th.nn.ModuleList(self.decoder)
        self.output_layer = instantiate(config=output_layer, in_channels=curr_channel, out_channels=output_channels, dilation=dilations[-1], enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad)

    def forward(self, inputs: 'Sequence') ->th.Tensor:
        """
        Forward pass of the HEALPix Unet decoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to decode

        Returns
        -------
        torch.Tensor: The decoded values
        """
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer['upsamp'] is not None:
                up = layer['upsamp'](x)
                x = th.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer['conv'](x)
            if layer['recurrent'] is not None:
                x = layer['recurrent'](x)
        return self.output_layer(x)

    def reset(self):
        """Resets the state of the decoder layers"""
        for layer in self.decoder:
            if layer['recurrent'] is not None:
                layer['recurrent'].reset()


class UNetEncoder(th.nn.Module):
    """Generic UNetEncoder that can be applied to arbitrary meshes."""

    def __init__(self, conv_block: 'DictConfig', down_sampling_block: 'DictConfig', recurrent_block: 'DictConfig'=None, input_channels: 'int'=3, n_channels: 'Sequence'=(16, 32, 64), n_layers: 'Sequence'=(2, 2, 1), dilations: 'list'=None, enable_nhwc: 'bool'=False, enable_healpixpad: 'bool'=False):
        """
        Parameters
        ----------
        conv_block: DictConfig
            dictionary of instantiable parameters for the convolutional block
        down_sampling_block: DictConfig
            dictionary of instantiable parameters for the downsample block
        recurrent_block: DictConfig, optional
            dictionary of instantiable parameters for the recurrent block
            recurrent blocks are not used if this is None
        input_channels: int, optional
            Number of input channels
        n_channels: Sequence, optional
            The number of channels in each encoder layer
        n_layers:, Sequence, optional
            Number of layers to use for the convolutional blocks
        dilations: list, optional
            List of dialtions to use for the the convolutional blocks
        enable_nhwc: bool, optional
            If channel last format should be used
        enable_healpixpad, bool, optional
            If the healpixpad library should be used (if installed)
        """
        super().__init__()
        self.n_channels = n_channels
        if dilations is None:
            dilations = [(1) for _ in range(len(n_channels))]
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(instantiate(config=down_sampling_block, enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
            modules.append(instantiate(config=conv_block, in_channels=old_channels, latent_channels=curr_channel, out_channels=curr_channel, dilation=dilations[n], n_layers=n_layers[n], enable_nhwc=enable_nhwc, enable_healpixpad=enable_healpixpad))
            old_channels = curr_channel
            self.encoder.append(th.nn.Sequential(*modules))
        self.encoder = th.nn.ModuleList(self.encoder)

    def forward(self, inputs: 'Sequence') ->Sequence:
        """
        Forward pass of the HEALPix Unet encoder

        Parameters
        ----------
        inputs: Sequence
            The inputs to enccode

        Returns
        -------
        Sequence: The encoded values
        """
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        """Resets the state of the decoder layers"""
        pass


RAD_PER_DEG = np.pi / 180.0


def _dali_mod(a, b):
    return a - b * dali.math.floor(a / b)


def _days_from_2000(model_time):
    """Get the days since year 2000."""
    return (model_time - DATETIME_2000) / (24.0 * 3600.0)


def _greenwich_mean_sidereal_time(model_time):
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = 67310.54841 + jul_centuries * (876600 * 3600 + 8640184.812866 + jul_centuries * (0.093104 - jul_centuries * 6.2 * 1e-05))
    theta_radians = _dali_mod(theta / 240.0 * RAD_PER_DEG, 2 * np.pi)
    return theta_radians


def _local_mean_sidereal_time(model_time, longitude):
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _local_hour_angle(model_time, longitude, right_ascension):
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    return _local_mean_sidereal_time(model_time, longitude) - right_ascension


def _obliquity_star(julian_centuries):
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic
    """
    return (23.0 + 26.0 / 60 + 21.406 / 3600.0 - (46.836769 * julian_centuries - 0.0001831 * julian_centuries ** 2 + 0.0020034 * julian_centuries ** 3 - 5.76e-07 * julian_centuries ** 4 - 4.34e-08 * julian_centuries ** 5) / 3600.0) * RAD_PER_DEG


def _sun_ecliptic_longitude(model_time):
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    mean_anomaly = (357.5291 + 35999.0503 * julian_centuries - 0.0001559 * julian_centuries * julian_centuries - 4.8e-07 * julian_centuries * julian_centuries * julian_centuries) * RAD_PER_DEG
    mean_longitude = (280.46645 + 36000.76983 * julian_centuries + 0.0003032 * julian_centuries ** 2) * RAD_PER_DEG
    d_l = ((1.9146 - 0.004817 * julian_centuries - 1.4e-05 * julian_centuries ** 2) * dali.math.sin(mean_anomaly) + (0.019993 - 0.000101 * julian_centuries) * dali.math.sin(2 * mean_anomaly) + 0.00029 * dali.math.sin(3 * mean_anomaly)) * RAD_PER_DEG
    return mean_longitude + d_l


def _right_ascension_declination(model_time):
    """
    Right ascension and declination of the sun.
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)
    eclon = _sun_ecliptic_longitude(model_time)
    x = dali.math.cos(eclon)
    y = dali.math.cos(eps) * dali.math.sin(eclon)
    z = dali.math.sin(eps) * dali.math.sin(eclon)
    r = dali.math.sqrt(1.0 - z * z)
    declination = dali.math.atan2(z, r)
    right_ascension = 2 * dali.math.atan2(y, x + r)
    return right_ascension, declination


def _star_cos_zenith(model_time, lat, lon):
    """
    Return cosine of star zenith angle
    lon,lat in radians
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """
    ra, dec = _right_ascension_declination(model_time)
    h_angle = _local_hour_angle(model_time, lon, ra)
    cosine_zenith = dali.math.sin(lat) * dali.math.sin(dec) + dali.math.cos(lat) * dali.math.cos(dec) * dali.math.cos(h_angle)
    return cosine_zenith


def cos_zenith_angle(time: 'dali.types.DALIDataType', latlon: 'dali.types.DALIDataType'):
    """
    Dali datapipe for computing Cosine of sun-zenith angle for lon, lat at time (UTC).

    Parameters
    ----------
    time : dali.types.DALIDataType
        Time in seconds since 2000-01-01 12:00:00 UTC. Shape `(seq_length,)`.
    latlon : dali.types.DALIDataType
        Latitude and longitude in degrees. Shape `(2, nr_lat, nr_lon)`.

    Returns
    -------
    dali.types.DALIDataType
        Cosine of sun-zenith angle. Shape `(seq_length, 1, nr_lat, nr_lon)`.
    """
    lat = latlon[dali.newaxis, 0:1, :, :] * RAD_PER_DEG
    lon = latlon[dali.newaxis, 1:2, :, :] * RAD_PER_DEG
    time = time[:, dali.newaxis, dali.newaxis, dali.newaxis]
    return _star_cos_zenith(time, lat, lon)


class _CosZenWrapper(torch.nn.Module):

    def __init__(self, model, lon, lat):
        super().__init__()
        self.model = model
        self.lon = lon
        self.lat = lat

    def forward(self, x, time):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz)
        x, z = torch.broadcast_tensors(x, z)
        x = torch.cat([x, z], dim=1)
        return self.model(x)


class _GraphCastWrapper(torch.nn.Module):

    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x):
        x = x
        y = self.model(x)
        return y


class _DLWPWrapper(torch.nn.Module):

    def __init__(self, model, lsm, longrid, latgrid, topographic_height, ll_to_cs_mapfile_path, cs_to_ll_mapfile_path):
        super(_DLWPWrapper, self).__init__()
        self.model = model
        self.lsm = lsm
        self.longrid = longrid
        self.latgrid = latgrid
        self.topographic_height = topographic_height
        self.input_map_wts = xarray.open_dataset(ll_to_cs_mapfile_path)
        self.output_map_wts = xarray.open_dataset(cs_to_ll_mapfile_path)

    def prepare_input(self, input, time):
        device = input.device
        dtype = input.dtype
        i = self.input_map_wts.row.values - 1
        j = self.input_map_wts.col.values - 1
        data = self.input_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype)
        N, T, C = input.shape[0], input.shape[1], input.shape[2]
        input = (M @ input.reshape(N * T * C, -1).T).T
        S = int((M.shape[0] / 6) ** 0.5)
        input = input.reshape(N, T, C, 6, S, S)
        input_list = list(torch.split(input, 1, dim=1))
        input_list = [tensor.squeeze(1) for tensor in input_list]
        repeat_vals = input.shape[0], -1, -1, -1, -1
        for i in range(len(input_list)):
            tisr = np.maximum(cos_zenith_angle(time - datetime.timedelta(hours=6 * (input.shape[1] - 1)) + datetime.timedelta(hours=6 * i), self.longrid, self.latgrid), 0) - 1 / np.pi
            tisr = torch.tensor(tisr, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0)
            tisr = tisr.expand(*repeat_vals)
            input_list[i] = torch.cat((input_list[i], tisr), dim=1)
        input_model = torch.cat(input_list, dim=1)
        lsm_tensor = torch.tensor(self.lsm, dtype=dtype).unsqueeze(dim=0)
        lsm_tensor = lsm_tensor.expand(*repeat_vals)
        topographic_height_tensor = torch.tensor((self.topographic_height - 3724.0) / 8349.0, dtype=dtype).unsqueeze(dim=0)
        topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)
        input_model = torch.cat((input_model, lsm_tensor, topographic_height_tensor), dim=1)
        return input_model

    def prepare_output(self, output):
        device = output.device
        dtype = output.dtype
        output = torch.split(output, output.shape[1] // 2, dim=1)
        output = torch.stack(output, dim=1)
        i = self.output_map_wts.row.values - 1
        j = self.output_map_wts.col.values - 1
        data = self.output_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype)
        N, T, C = output.shape[0], 2, output.shape[2]
        output = (M @ output.reshape(N * T * C, -1).T).T
        output = output.reshape(N, T, C, 721, 1440)
        return output

    def forward(self, x, time):
        x = self.prepare_input(x, time)
        y = self.model(x)
        return self.prepare_output(y)


class PatchRecovery2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)
        return output[:, :, padding_top:H - padding_bottom, padding_left:W - padding_right]


def get_earth_position_index(window_size, ndim=3):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    This function construct the position index to reuse symmetrical parameters of the position bias.
    implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of tensor, 3 or 2

    Returns:
        position_index (torch.Tensor): [win_pl * win_lat * win_lon, win_pl * win_lat * win_lon] or [win_lat * win_lon, win_lat * win_lon]
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
    elif ndim == 2:
        win_lat, win_lon = window_size
    if ndim == 3:
        coords_zi = torch.arange(win_pl)
        coords_zj = -torch.arange(win_pl) * win_pl
    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat
    coords_w = torch.arange(win_lon)
    if ndim == 3:
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    elif ndim == 2:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()
    if ndim == 3:
        coords[:, :, 2] += win_lon - 1
        coords[:, :, 1] *= 2 * win_lon - 1
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    elif ndim == 2:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1
    position_index = coords.sum(-1)
    return position_index


class EarthAttention2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        window_size (tuple[int]): [latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.type_of_windows = input_resolution[0] // window_size[0]
        self.earth_position_bias_table = nn.Parameter(torch.zeros(window_size[0] ** 2 * (window_size[1] * 2 - 1), self.type_of_windows, num_heads))
        earth_position_index = get_earth_position_index(window_size, ndim=2)
        self.register_buffer('earth_position_index', earth_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.earth_position_bias_table = trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: 'torch.Tensor', mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.type_of_windows, -1)
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)
        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'torch.Tensor'):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def crop2d(x: 'torch.Tensor', resolution):
    """
    Args:
        x (torch.Tensor): B, C, Lat, Lon
        resolution (tuple[int]): Lat, Lon
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[:, :, padding_top:Lat - padding_bottom, padding_left:Lon - padding_right]


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon
    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left
    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


def window_partition(x: 'torch.Tensor', window_size, ndim=3):
    """
    Args:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        ndim (int): dimension of window (3 or 2)

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(-1, Pl // win_pl * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = x.permute(0, 3, 1, 2, 4, 5).contiguous().view(-1, Lat // win_lat, win_lat, win_lon, C)
        return windows


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    Along the longitude dimension, the leftmost and rightmost indices are actually close to each other.
    If half windows apper at both leftmost and rightmost positions, they are dircetly merged into one window.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude] or [latitude, longitude]
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of window (3 or 2)

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon) or (n_lon, n_lat, win_lat*win_lon, win_lat*win_lon)
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size
        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size
        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))
    if ndim == 3:
        pl_slices = slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None)
    lat_slices = slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None)
    lon_slices = slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None)
    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]
    mask_windows = window_partition(img_mask, window_size, ndim=ndim)
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    mask_windows = mask_windows.view(mask_windows.shape[0], mask_windows.shape[1], win_total)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        Pl (int): pressure levels
        Lat (int): latitude
        Lon (int): longitude
        ndim (int): dimension of window (3 or 2)

    Returns:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Lat, Lon, -1)
        return x


class Transformer2DBlock(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (6, 12) if window_size is None else window_size
        shift_size = (3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        padding = get_pad2d(input_resolution, window_size)
        self.pad = nn.ZeroPad2d(padding)
        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[2] + padding[3]
        pad_resolution[1] += padding[0] + padding[1]
        self.attn = EarthAttention2D(dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        shift_lat, shift_lon = self.shift_size
        self.roll = shift_lon and shift_lat
        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size, ndim=2)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: 'torch.Tensor'):
        Lat, Lon = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Lat, Lon, C)
        x = self.pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        _, Lat_pad, Lon_pad, _ = x.shape
        shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_lat, -shift_lat), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)
        win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_lat * win_lon, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_lat, win_lon, C)
        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2)
            x = torch.roll(shifted_x, shifts=(shift_lat, shift_lon), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2)
            x = shifted_x
        x = crop2d(x.permute(0, 3, 1, 2), self.input_resolution).permute(0, 2, 3, 1)
        x = x.reshape(B, Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UpSample2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Up-sampling operation.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: 'torch.Tensor'):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        x = self.linear1(x)
        x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, in_lat * 2, in_lon * 2, -1)
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = x[:, pad_top:2 * in_lat - pad_bottom, pad_left:2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """A 2D Transformer Decoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Recovery.
        out_chans (int): number of output channels of Patch Recovery.
        dim (int): Number of input channels of transformer.
        output_resolution (tuple[int]): Input resolution for transformer after upsampling.
        middle_resolution (tuple[int]): Input resolution for transformer before upsampling.
        depth (int): Number of blocks for transformer after upsampling.
        depth_middle (int): Number of blocks for transformer before upsampling.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, img_size, patch_size, out_chans, dim, output_resolution, middle_resolution, depth, depth_middle, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.out_chans = out_chans
        self.dim = dim
        self.output_resolution = output_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads
        self.blocks_middle = nn.ModuleList([Transformer2DBlock(dim=dim * 2, input_resolution=middle_resolution, num_heads=num_heads_middle, window_size=window_size, shift_size=(0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle, norm_layer=norm_layer) for i in range(depth_middle)])
        self.upsample = UpSample2D(in_dim=dim * 2, out_dim=dim, input_resolution=middle_resolution, output_resolution=output_resolution)
        self.blocks = nn.ModuleList([Transformer2DBlock(dim=dim, input_resolution=output_resolution, num_heads=num_heads, window_size=window_size, shift_size=(0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.patchrecovery2d = PatchRecovery2D(img_size, patch_size, 2 * dim, out_chans)

    def forward(self, x, skip):
        B, Lat, Lon, C = skip.shape
        for blk in self.blocks_middle:
            x = blk(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        output = torch.concat([x, skip.reshape(B, -1, C)], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Lat, Lon)
        output = self.patchrecovery2d(output)
        return output


class DownSample2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Down-sampling operation

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: 'torch.Tensor'):
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_lat, in_lon, C)
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_lat * out_lon, 4 * C)
        x = self.norm(x)
        x = self.linear(x)
        return x


class PatchEmbed2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0
        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)
        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)
        self.pad = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: 'torch.Tensor'):
        B, C, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class EncoderLayer(nn.Module):
    """A 2D Transformer Encoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Embedding.
        in_chans (int): number of input channels of Patch Embedding.
        dim (int): Number of input channels of transformer.
        input_resolution (tuple[int]): Input resolution for transformer before downsampling.
        middle_resolution (tuple[int]): Input resolution for transformer after downsampling.
        depth (int): Number of blocks for transformer before downsampling.
        depth_middle (int): Number of blocks for transformer after downsampling.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, img_size, patch_size, in_chans, dim, input_resolution, middle_resolution, depth, depth_middle, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = in_chans
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads
        self.patchembed2d = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim)
        self.blocks = nn.ModuleList([Transformer2DBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=(0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.downsample = DownSample2D(in_dim=dim, input_resolution=input_resolution, output_resolution=middle_resolution)
        self.blocks_middle = nn.ModuleList([Transformer2DBlock(dim=dim * 2, input_resolution=middle_resolution, num_heads=num_heads_middle, window_size=window_size, shift_size=(0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle, norm_layer=norm_layer) for i in range(depth_middle)])

    def forward(self, x):
        x = self.patchembed2d(x)
        B, C, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        skip = x.reshape(B, Lat, Lon, C)
        x = self.downsample(x)
        for blk in self.blocks_middle:
            x = blk(x)
        return x, skip


class EarthAttention3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.type_of_windows = input_resolution[0] // window_size[0] * (input_resolution[1] // window_size[1])
        self.earth_position_bias_table = nn.Parameter(torch.zeros(window_size[0] ** 2 * window_size[1] ** 2 * (window_size[2] * 2 - 1), self.type_of_windows, num_heads))
        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer('earth_position_index', earth_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.earth_position_bias_table = trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: 'torch.Tensor', mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], self.type_of_windows, -1)
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)
        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def crop3d(x: 'torch.Tensor', resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]
    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front
    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top
    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[:, :, padding_front:Pl - padding_back, padding_top:Lat - padding_bottom, padding_left:Lon - padding_right]


class Transformer3DBlock(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)
        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]
        self.attn = EarthAttention3D(dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat
        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: 'torch.Tensor'):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape
        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lat), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)
        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad)
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad)
            x = shifted_x
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FuserLayer(nn.Module):
    """Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([Transformer3DBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path, norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Fengwu(Module):
    """
    FengWu PyTorch impl of: `FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`
    - https://arxiv.org/pdf/2304.02948.pdf

    Args:
        img_size: Image size(Lat, Lon). Default: (721,1440)
        pressure_level: Number of pressure_level. Default: 37
        embed_dim (int): Patch embedding dimension. Default: 192
        patch_size (tuple[int]): Patch token size. Default: (4,4)
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(self, img_size=(721, 1440), pressure_level=37, embed_dim=192, patch_size=(4, 4), num_heads=(6, 12, 12, 6), window_size=(2, 6, 12)):
        super().__init__(meta=MetaData())
        drop_path = np.linspace(0, 0.2, 8).tolist()
        drop_path_fuser = [0.2] * 6
        resolution_down1 = math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1])
        resolution_down2 = math.ceil(resolution_down1[0] / 2), math.ceil(resolution_down1[1] / 2)
        resolution = resolution_down1, resolution_down2
        self.encoder_surface = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=4, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.encoder_z = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=pressure_level, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.encoder_r = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=pressure_level, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.encoder_u = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=pressure_level, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.encoder_v = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=pressure_level, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.encoder_t = EncoderLayer(img_size=img_size, patch_size=patch_size, in_chans=pressure_level, dim=embed_dim, input_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.fuser = FuserLayer(dim=embed_dim * 2, input_resolution=(6, resolution[1][0], resolution[1][1]), depth=6, num_heads=num_heads[1], window_size=window_size, drop_path=drop_path_fuser)
        self.decoder_surface = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=4, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.decoder_z = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=pressure_level, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.decoder_r = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=pressure_level, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.decoder_u = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=pressure_level, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.decoder_v = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=pressure_level, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)
        self.decoder_t = DecoderLayer(img_size=img_size, patch_size=patch_size, out_chans=pressure_level, dim=embed_dim, output_resolution=resolution[0], middle_resolution=resolution[1], depth=2, depth_middle=6, num_heads=num_heads[:2], window_size=window_size[1:], drop_path=drop_path)

    def prepare_input(self, surface, z, r, u, v, t):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        return torch.concat([surface, z, r, u, v, t], dim=1)

    def forward(self, x):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        surface = x[:, :4, :, :]
        z = x[:, 4:41, :, :]
        r = x[:, 41:78, :, :]
        u = x[:, 78:115, :, :]
        v = x[:, 115:152, :, :]
        t = x[:, 152:189, :, :]
        surface, skip_surface = self.encoder_surface(surface)
        z, skip_z = self.encoder_z(z)
        r, skip_r = self.encoder_r(r)
        u, skip_u = self.encoder_u(u)
        v, skip_v = self.encoder_v(v)
        t, skip_t = self.encoder_t(t)
        x = torch.concat([surface.unsqueeze(1), z.unsqueeze(1), r.unsqueeze(1), u.unsqueeze(1), v.unsqueeze(1), t.unsqueeze(1)], dim=1)
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.fuser(x)
        x = x.reshape(B, PL, L_SIZE, C)
        surface, z, r, u, v, t = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :], x[:, 4, :, :], x[:, 5, :, :]
        surface = self.decoder_surface(surface, skip_surface)
        z = self.decoder_z(z, skip_z)
        r = self.decoder_r(r, skip_r)
        u = self.decoder_u(u, skip_u)
        v = self.decoder_v(v, skip_v)
        t = self.decoder_t(t, skip_t)
        return surface, z, r, u, v, t


class BaseModel(Module):
    """Base model class."""

    def data_dict_to_input(self, data_dict, **kwargs) ->Any:
        """Convert data dictionary to appropriate input for the model."""
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs) ->Dict:
        """Compute the loss dictionary for the model."""
        raise NotImplementedError

    @torch.no_grad()
    def eval_dict(self, data_dict, **kwargs) ->Dict:
        """Compute the evaluation dictionary for the model."""
        raise NotImplementedError

    def image_pointcloud_dict(self, data_dict, datamodule) ->Tuple[Dict, Dict]:
        """Compute the image dict and pointcloud dict for the model."""
        raise NotImplementedError


class SinusoidalEncoding(nn.Module):
    """SinusoidalEncoding."""

    def __init__(self, num_channels: 'int', data_range: 'float'=2.0):
        super().__init__()
        assert num_channels % 2 == 0, f'num_channels must be even for sin/cos, got {num_channels}'
        self.num_channels = num_channels
        self.data_range = data_range

    def forward(self, x):
        freqs = 2 ** torch.arange(start=0, end=self.num_channels // 2, device=x.device)
        freqs = 2 * np.pi / self.data_range * freqs
        x = x.unsqueeze(-1)
        freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
        return x


class ResidualLinearBlock(nn.Module):
    """MLPBlock."""

    def __init__(self, in_channels: 'int', out_channels: 'int', hidden_channels: 'int'=None, activation: 'type[nn.Module]'=nn.GELU):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.blocks = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels), activation(), nn.Linear(hidden_channels, out_channels), nn.LayerNorm(out_channels))
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.blocks(x)
        out = self.activation(out + self.shortcut(x))
        return out


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class MLPBlock(nn.Module):
    """MLPBlock."""

    def __init__(self, in_channels: 'int', hidden_channels: 'int'=None, out_channels: 'int'=None, activation: 'type[nn.Module]'=nn.GELU):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        out = self.activation(out + self.shortcut(x))
        return out


class PointFeatures:
    """
    PointFeatures class represents the features defined on a set of points in 3D space.
    The vertices are the set of 3D coordinates that define the points and the features
    are defined at each point.

    The point features have BxNx3 and BxNxC shape where B is the batch size, N is the
    number of points, and C is the number of channels in the features.
    """
    _shape_hint = None
    vertices: "Float[Tensor, 'B N 3']"
    features: "Float[Tensor, 'B N C']"
    num_channels: 'int' = None
    num_points: 'int' = None

    def __class_getitem__(cls, item: 'str'):


        class _PointFeaturesSubclass(cls):
            _shape_hint = tuple(item.split())
        return _PointFeaturesSubclass

    def __init__(self, vertices, features):
        self.vertices = vertices
        self.features = features
        self.check()
        self.batch_size = len(self.vertices)
        self.num_points = self.vertices.shape[1]
        self.num_channels = self.features.shape[-1]

    @property
    def device(self):
        return self.vertices.device

    def check(self):
        assert self.vertices.ndim == 3
        assert self.features.ndim == 3
        assert self.vertices.shape[0] == self.features.shape[0]
        assert self.vertices.shape[1] == self.features.shape[1]
        assert self.vertices.shape[2] == 3

    def to(self, device):
        self.vertices = self.vertices
        self.features = self.features
        return self

    def expand_batch_size(self, batch_size: 'int'):
        if batch_size == 1:
            return self
        self.vertices = self.vertices.expand(batch_size, -1, -1).contiguous()
        self.features = self.features.expand(batch_size, -1, -1).contiguous()
        self.batch_size = batch_size
        return self

    def voxel_down_sample(self, voxel_size: 'float'):
        down_vertices = []
        down_features = []
        for vert, feat in zip(self.vertices, self.features):
            assert len(vert.shape) == 2
            assert vert.shape[1] == 3
            int_coords = torch.floor(vert / voxel_size).int()
            _, unique_indices = np.unique(int_coords.cpu().numpy(), axis=0, return_index=True)
            unique_indices = torch.from_numpy(unique_indices)
            down_vertices.append(vert[unique_indices])
            down_features.append(feat[unique_indices])
        min_len = min([len(vert) for vert in down_vertices])
        down_vertices = torch.stack([vert[:min_len] for vert in down_vertices], dim=0)
        down_features = torch.stack([feat[:min_len] for feat in down_features], dim=0)
        return PointFeatures(down_vertices, down_features)

    def contiguous(self):
        self.vertices = self.vertices.contiguous()
        self.features = self.features.contiguous()
        return self

    def __add__(self, other):
        assert self.batch_size == other.batch_size
        assert self.num_channels == other.num_channels
        return PointFeatures(self.vertices, self.features + other.features)

    def __mul__(self, other):
        assert self.batch_size == other.batch_size
        assert self.num_channels == other.num_channels
        return PointFeatures(self.vertices, self.features * other.features)

    def __len__(self):
        return self.batch_size

    def __repr__(self) ->str:
        return f'PointFeatures(vertices={self.vertices.shape}, features={self.features.shape})'


class VerticesToPointFeatures(nn.Module):
    """
    VerticesToPointFeatures module converts the 3D vertices (XYZ coordinates) to point features.

    The module applies sinusoidal encoding to the vertices and optionally applies
    an MLP to the encoded vertices.
    """

    def __init__(self, embed_dim: 'int', out_features: 'Optional[int]'=32, use_mlp: 'Optional[bool]'=True, pos_embed_range: 'Optional[float]'=2.0) ->None:
        super().__init__()
        self.pos_embed = SinusoidalEncoding(embed_dim, pos_embed_range)
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(3 * embed_dim, out_features, [])

    def forward(self, vertices: "Float[Tensor, 'B N 3']") ->PointFeatures:
        assert vertices.ndim == 3, f'Expected 3D vertices of shape BxNx3, got {vertices.shape}'
        vert_embed = self.pos_embed(vertices)
        if self.use_mlp:
            vert_embed = self.mlp(vert_embed)
        return PointFeatures(vertices, vert_embed)


class GridFeaturesMemoryFormat(enum.Enum):
    """Memory format used for GridFeatures class.

    The memory format defines how the grid features are stored in memory.

    b_x_y_z_c: Batch, X, Y, Z, Channels (3D Grid)
    b_c_x_y_z: Batch, Channels, X, Y, Z (3D Grid)
    b_zc_x_y: Batch, Z * Channels, X, Y (2D Grid)
    b_xc_y_z: Batch, X * Channels, Y, Z (2D Grid)
    b_yc_x_z: Batch, Y * Channels, X, Z (2D Grid)
    """
    b_x_y_z_c = enum.auto()
    b_c_x_y_z = enum.auto()
    b_zc_x_y = enum.auto()
    b_xc_y_z = enum.auto()
    b_yc_x_z = enum.auto()


def convert_from_b_x_y_z_c(tensor, to_memory_format):
    B, H, W, D, C = tensor.shape
    if to_memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
        return tensor.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)
    elif to_memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
        return tensor.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)
    elif to_memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
        return tensor.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)
    elif to_memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        raise ValueError(f'Unsupported memory format: {to_memory_format}')


def convert_to_b_x_y_z_c(tensor, from_memory_format, num_channels):
    if from_memory_format == GridFeaturesMemoryFormat.b_zc_x_y:
        B, D_C, H, W = tensor.shape
        D, rem = divmod(D_C, num_channels)
        assert rem == 0, 'Number of channels does not match.'
        return tensor.reshape(B, D, num_channels, H, W).permute(0, 3, 4, 1, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_xc_y_z:
        B, H_C, W, D = tensor.shape
        H, rem = divmod(H_C, num_channels)
        assert rem == 0, 'Number of channels does not match.'
        return tensor.reshape(B, H, num_channels, W, D).permute(0, 1, 3, 4, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_yc_x_z:
        B, W_C, H, D = tensor.shape
        W, rem = divmod(W_C, num_channels)
        assert rem == 0, 'Number of channels does not match.'
        return tensor.reshape(B, W, num_channels, H, D).permute(0, 3, 1, 4, 2)
    elif from_memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f'Unsupported memory format: {from_memory_format}')


def grid_init(bb_max, bb_min, resolution):
    """grid_init."""
    grid = torch.meshgrid(torch.linspace(bb_min[0], bb_max[0], resolution[0]), torch.linspace(bb_min[1], bb_max[1], resolution[1]), torch.linspace(bb_min[2], bb_max[2], resolution[2]))
    grid = torch.stack(grid, dim=-1)
    return grid


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm2d."""

    def forward(self, x: 'Tensor') ->Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


@torch.no_grad()
def _chunked_knn_search(ref_positions: "Int[Tensor, 'N 3']", query_positions: "Int[Tensor, 'M 3']", k: 'int', chunk_size: 'int'=4096):
    """Divide the out_positions into chunks and perform knn search."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert chunk_size > 0
    neighbors_index = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i:i + chunk_size]
        chunk_neighbors_index = _knn_search(ref_positions, chunk_out_positions, k)
        neighbors_index.append(chunk_neighbors_index)
    return torch.concatenate(neighbors_index, dim=0)


class NeighborSearchReturn:
    """
    Wrapper for the output of a neighbor search operation.
    """
    _neighbors_index: "Int[Tensor, 'N']"
    _neighbors_row_splits: "Int[Tensor, 'M + 1']"

    def __init__(self, *args):
        if len(args) == 2:
            self._neighbors_index = args[0].long()
            self._neighbors_row_splits = args[1].long()
        elif len(args) == 1:
            self._neighbors_index = args[0].neighbors_index.long()
            self._neighbors_row_splits = args[0].neighbors_row_splits.long()
        else:
            raise ValueError('NeighborSearchReturn must be initialized with 1 or 2 arguments')

    @property
    def neighbors_index(self):
        return self._neighbors_index

    @property
    def neighbors_row_splits(self):
        return self._neighbors_row_splits

    def to(self, device: 'Union[str, int, torch.device]'):
        self._neighbors_index
        self._neighbors_row_splits
        return self


def _radius_search_warp(points: 'wp.array(dtype=wp.vec3)', queries: 'wp.array(dtype=wp.vec3)', radius: 'float', grid_dim: 'Union[int, Tuple[int, int, int]]'=(128, 128, 128), device: 'str'='cuda'):
    if isinstance(grid_dim, int):
        grid_dim = grid_dim, grid_dim, grid_dim
    result_count = wp.zeros(shape=len(queries), dtype=wp.int32)
    grid = wp.HashGrid(dim_x=grid_dim[0], dim_y=grid_dim[1], dim_z=grid_dim[2], device=device)
    grid.build(points=points, radius=2 * radius)
    wp.launch(kernel=_radius_search_count, dim=len(queries), inputs=[grid.id, points, queries, result_count, radius], device=device)
    torch_offset = torch.zeros(len(result_count) + 1, device=device, dtype=torch.int32)
    result_count_torch = wp.to_torch(result_count)
    torch.cumsum(result_count_torch, dim=0, out=torch_offset[1:])
    total_count = torch_offset[-1].item()
    assert total_count < 2 ** 31 - 1, f'Total result count is too large: {total_count} > 2**31 - 1'
    result_point_idx = wp.zeros(shape=(total_count,), dtype=wp.int32)
    result_point_dist = wp.zeros(shape=(total_count,), dtype=wp.float32)
    wp.launch(kernel=_radius_search_query, dim=len(queries), inputs=[grid.id, points, queries, wp.from_torch(torch_offset), result_point_idx, result_point_dist, radius], device=device)
    return result_point_idx, result_point_dist, torch_offset


@torch.no_grad()
def batched_neighbor_radius_search(inp_positions: "Float[Tensor, 'B N 3']", out_positions: "Float[Tensor, 'B M 3']", radius: 'float', search_method: "Literal['warp']"='warp') ->NeighborSearchReturn:
    """
    inp_positions: [B,N,3]
    out_positions: [B,M,3]
    radius: float
    search_method: Literal["warp", "open3d"]
    """
    assert inp_positions.shape[0] == out_positions.shape[0], f'Batch size mismatch, {inp_positions.shape[0]} != {out_positions.shape[0]}'
    if search_method == 'warp':
        neighbor_index, neighbor_dist, neighbor_offset = batched_radius_search_warp(inp_positions, out_positions, radius)
    else:
        raise ValueError(f'search_method {search_method} not supported.')
    return NeighborSearchReturn(neighbor_index, neighbor_offset)


REDUCTIONS = ['min', 'max', 'mean', 'sum', 'var', 'std']


class PointFeatureConv(nn.Module):
    """PointFeatureConv."""

    def __init__(self, radius: 'float', edge_transform_mlp: 'Optional[nn.Module]'=None, out_transform_mlp: 'Optional[nn.Module]'=None, in_channels: 'int'=8, out_channels: 'int'=32, hidden_dim: 'Optional[int]'=None, channel_multiplier: 'int'=2, use_rel_pos: 'bool'=True, use_rel_pos_encode: 'bool'=False, pos_encode_dim: 'int'=32, pos_encode_range: 'float'=4, reductions: 'List[REDUCTION_TYPES]'=['mean'], downsample_voxel_size: 'Optional[float]'=None, out_point_feature_type: "Literal['provided', 'downsample', 'same']"='same', provided_in_channels: 'Optional[int]'=None, neighbor_search_vertices_scaler: "Optional[Float[Tensor, '3']]"=None, neighbor_search_type: "Literal['radius', 'knn']"='radius', radius_search_method: "Literal['open3d', 'warp']"='warp', knn_k: 'Optional[int]'=None):
        """If use_relative_position_encoding is True, the positional encoding vertex coordinate
        difference is added to the edge features.

        downsample_voxel_size: If not None, the input point cloud will be downsampled.

        out_point_feature_type: If "upsample", the output point features will be upsampled to the input point cloud size.

        use_rel_pos: If True, the relative position of the neighbor points will be used as the edge features.
        use_rel_pos_encode: If True, the encoding relative position of the neighbor points will be used as the edge features.

        if neighbor_search_vertices_scaler is not None, find neighbors using the
        scaled vertices. This allows finding neighbors with an axis aligned
        ellipsoidal neighborhood.
        """
        super().__init__()
        assert isinstance(reductions, (tuple, list)) and len(reductions) > 0, f'reductions must be a list or tuple of length > 0, got {reductions}'
        if out_point_feature_type == 'provided':
            assert downsample_voxel_size is None, 'downsample_voxel_size is only used for downsample'
            assert provided_in_channels is not None, 'provided_in_channels must be provided for provided type'
        elif out_point_feature_type == 'downsample':
            assert downsample_voxel_size is not None, 'downsample_voxel_size must be provided for downsample'
            assert provided_in_channels is None, 'provided_in_channels must be None for downsample type'
        elif out_point_feature_type == 'same':
            assert downsample_voxel_size is None, 'downsample_voxel_size is only used for downsample'
            assert provided_in_channels is None, 'provided_in_channels must be None for same type'
        if downsample_voxel_size is not None and downsample_voxel_size > radius:
            raise ValueError(f'downsample_voxel_size {downsample_voxel_size} must be <= radius {radius}')
        self.reductions = reductions
        self.downsample_voxel_size = downsample_voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_encode = use_rel_pos_encode
        self.out_point_feature_type = out_point_feature_type
        self.neighbor_search_vertices_scaler = neighbor_search_vertices_scaler
        self.neighbor_search_type = neighbor_search_type
        self.radius_search_method = radius_search_method
        if neighbor_search_type == 'radius':
            self.radius_or_k = radius
        elif neighbor_search_type == 'knn':
            assert knn_k is not None
            self.radius_or_k = knn_k
        else:
            raise ValueError(f'neighbor_search_type must be radius or knn, got {neighbor_search_type}')
        self.positional_encoding = SinusoidalEncoding(pos_encode_dim, data_range=pos_encode_range)
        if provided_in_channels is None:
            provided_in_channels = in_channels
        if hidden_dim is None:
            hidden_dim = channel_multiplier * out_channels
        if edge_transform_mlp is None:
            edge_in_channels = in_channels + provided_in_channels
            if use_rel_pos_encode:
                edge_in_channels += pos_encode_dim * 3
            elif use_rel_pos:
                edge_in_channels += 3
            edge_transform_mlp = MLPBlock(in_channels=edge_in_channels, hidden_channels=hidden_dim, out_channels=out_channels)
        self.edge_transform_mlp = edge_transform_mlp
        if out_transform_mlp is None:
            out_transform_mlp = MLPBlock(in_channels=out_channels * len(reductions), hidden_channels=hidden_dim, out_channels=out_channels)
        self.out_transform_mlp = out_transform_mlp

    def __repr__(self):
        out_str = f'{self.__class__.__name__}(in_channels={self.in_channels} out_channels={self.out_channels} search_type={self.neighbor_search_type} reductions={self.reductions}'
        if self.downsample_voxel_size is not None:
            out_str += f' down_voxel_size={self.downsample_voxel_size}'
        if self.use_rel_pos_encode:
            out_str += f' rel_pos_encode={self.use_rel_pos_encode}'
        out_str += ')'
        return out_str

    def forward(self, in_point_features: "PointFeatures['B N C1']", out_point_features: "Optional[PointFeatures['B M C1']]"=None, neighbor_search_vertices_scaler: "Optional[Float[Tensor, '3']]"=None) ->PointFeatures['B M C2']:
        """When out_point_features is None, the output will be generated on the
        in_point_features.vertices."""
        if self.out_point_feature_type == 'provided':
            assert out_point_features is not None, 'out_point_features must be provided for the provided type'
        elif self.out_point_feature_type == 'downsample':
            assert out_point_features is None
            out_point_features = in_point_features.voxel_down_sample(self.downsample_voxel_size)
        elif self.out_point_feature_type == 'same':
            assert out_point_features is None
            out_point_features = in_point_features
        in_num_channels = in_point_features.num_channels
        out_num_channels = out_point_features.num_channels
        assert in_num_channels + out_num_channels + self.use_rel_pos_encode * self.positional_encoding.num_channels * 3 + (not self.use_rel_pos_encode) * self.use_rel_pos * 3 == self.edge_transform_mlp.in_channels, f'input features shape {in_point_features.features.shape} and {out_point_features.features.shape} does not match the edge_transform_mlp input features {self.edge_transform_mlp.in_channels}'
        in_vertices = in_point_features.vertices
        out_vertices = out_point_features.vertices
        if self.neighbor_search_vertices_scaler is not None:
            neighbor_search_vertices_scaler = self.neighbor_search_vertices_scaler
        if neighbor_search_vertices_scaler is not None:
            in_vertices = in_vertices * neighbor_search_vertices_scaler
            out_vertices = out_vertices * neighbor_search_vertices_scaler
        if self.neighbor_search_type == 'knn':
            device = in_vertices.device
            neighbors_index = batched_neighbor_knn_search(in_vertices, out_vertices, self.radius_or_k)
            neighbors_index = neighbors_index.long().view(-1)
            neighbors_row_splits = torch.arange(0, out_vertices.shape[0] * out_vertices.shape[1] + 1, device=device) * self.radius_or_k
            rep_in_features = in_point_features.features.view(-1, in_num_channels)[neighbors_index]
            num_reps = self.radius_or_k
        elif self.neighbor_search_type == 'radius':
            neighbors = batched_neighbor_radius_search(in_vertices, out_vertices, radius=self.radius_or_k, search_method=self.radius_search_method)
            neighbors_index = neighbors.neighbors_index.long()
            rep_in_features = in_point_features.features.view(-1, in_num_channels)[neighbors_index]
            neighbors_row_splits = neighbors.neighbors_row_splits
            num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        else:
            raise ValueError(f'neighbor_search_type must be radius or knn, got {self.neighbor_search_type}')
        self_features = torch.repeat_interleave(out_point_features.features.view(-1, out_num_channels).contiguous(), num_reps, dim=0)
        edge_features = [rep_in_features, self_features]
        if self.use_rel_pos or self.use_rel_pos_encode:
            in_rep_vertices = in_point_features.vertices.view(-1, 3)[neighbors_index]
            self_vertices = torch.repeat_interleave(out_point_features.vertices.view(-1, 3).contiguous(), num_reps, dim=0)
            if self.use_rel_pos_encode:
                edge_features.append(self.positional_encoding(in_rep_vertices.view(-1, 3) - self_vertices.view(-1, 3)))
            elif self.use_rel_pos:
                edge_features.append(in_rep_vertices - self_vertices)
        edge_features = torch.cat(edge_features, dim=1)
        edge_features = self.edge_transform_mlp(edge_features)
        out_features = [row_reduction(edge_features, neighbors_row_splits, reduction=reduction) for reduction in self.reductions]
        out_features = torch.cat(out_features, dim=-1)
        out_features = self.out_transform_mlp(out_features)
        out_features = out_features.view(out_point_features.batch_size, out_point_features.num_points, out_features.shape[-1])
        return PointFeatures(out_point_features.vertices, out_features)


class GridFeatureToPointGraphConv(nn.Module):
    """GridFeatureToPointGraphConv."""

    def __init__(self, grid_in_channels: 'int', point_in_channels: 'int', out_channels: 'int', aabb_max: 'Tuple[float, float, float]', aabb_min: 'Tuple[float, float, float]', hidden_dim: 'Optional[int]'=None, use_rel_pos: 'bool'=True, use_rel_pos_embed: 'bool'=False, pos_embed_dim: 'int'=32, neighbor_search_type: "Literal['radius', 'knn']"='radius', knn_k: 'int'=16, reductions: 'List[REDUCTION_TYPES]'=['mean']) ->None:
        super().__init__()
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min
        self.conv = PointFeatureConv(radius=np.sqrt(3), in_channels=grid_in_channels, out_channels=out_channels, provided_in_channels=point_in_channels, hidden_dim=hidden_dim, use_rel_pos=use_rel_pos, use_rel_pos_encode=use_rel_pos_embed, pos_encode_dim=pos_embed_dim, out_point_feature_type='provided', neighbor_search_type=neighbor_search_type, knn_k=knn_k, reductions=reductions)

    def forward(self, grid_features: 'GridFeatures', point_features: 'PointFeatures') ->PointFeatures:
        resolution = grid_features.resolution
        vertices_scaler = torch.FloatTensor([resolution[0] / (self.aabb_max[0] - self.aabb_min[0]), resolution[1] / (self.aabb_max[1] - self.aabb_min[1]), resolution[2] / (self.aabb_max[2] - self.aabb_min[2])])
        out_point_features = self.conv(grid_features.point_features.contiguous(), point_features, neighbor_search_vertices_scaler=vertices_scaler)
        return out_point_features


class PointFeatureCat(nn.Module):
    """PointFeatureCat."""

    def forward(self, point_features: "PointFeatures['N C1']", point_features2: "PointFeatures['N C2']") ->PointFeatures['N C3']:
        return PointFeatures(point_features.vertices, torch.cat([point_features.features, point_features2.features], dim=1))


class GridFeatureToPointInterp(nn.Module):
    """GridFeatureToPointInterp."""

    def __init__(self, aabb_max: 'Tuple[float, float, float]', aabb_min: 'Tuple[float, float, float]', cat_in_point_features: 'bool'=True) ->None:
        super().__init__()
        self.aabb_max = torch.Tensor(aabb_max)
        self.aabb_min = torch.Tensor(aabb_min)
        self.cat_in_point_features = cat_in_point_features
        self.cat = PointFeatureCat()

    def to(self, *args, **kwargs):
        self.aabb_max = self.aabb_max
        self.aabb_min = self.aabb_min
        return super()

    def forward(self, grid_features: 'GridFeatures', point_features: 'PointFeatures') ->PointFeatures:
        grid_features
        xyz = point_features.vertices
        self
        normalized_xyz = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min) * 2 - 1
        normalized_xyz = normalized_xyz.view(1, 1, 1, -1, 3)
        batch_grid_features = grid_features.batch_features
        batch_point_features = F.grid_sample(batch_grid_features, normalized_xyz, align_corners=True).squeeze().permute(1, 0)
        out_point_features = PointFeatures(point_features.vertices, batch_point_features)
        if self.cat_in_point_features:
            out_point_features = self.cat(point_features, out_point_features)
        return out_point_features


class PointFeatureTransform(nn.Module):
    """PointFeatureTransform."""

    def __init__(self, feature_transform: 'nn.Module'):
        super().__init__()
        self.feature_transform = feature_transform

    def forward(self, point_features: "PointFeatures['N C1']") ->PointFeatures['N C2']:
        return PointFeatures(point_features.vertices, self.feature_transform(point_features.features))


class GridFeatureToPoint(nn.Module):
    """GridFeatureToPoint."""

    def __init__(self, grid_in_channels: 'int', point_in_channels: 'int', out_channels: 'int', aabb_max: 'Tuple[float, float, float]', aabb_min: 'Tuple[float, float, float]', hidden_dim: 'Optional[int]'=None, use_rel_pos: 'bool'=True, use_rel_pos_embed: 'bool'=False, pos_embed_dim: 'int'=32, sample_method: "Literal['graphconv', 'interp']"='graphconv', neighbor_search_type: "Literal['radius', 'knn']"='radius', knn_k: 'int'=16, reductions: 'List[REDUCTION_TYPES]'=['mean']) ->None:
        super().__init__()
        self.sample_method = sample_method
        if sample_method == 'graphconv':
            self.conv = GridFeatureToPointGraphConv(grid_in_channels, point_in_channels, out_channels, aabb_max, aabb_min, hidden_dim=hidden_dim, use_rel_pos=use_rel_pos, use_rel_pos_embed=use_rel_pos_embed, pos_embed_dim=pos_embed_dim, neighbor_search_type=neighbor_search_type, knn_k=knn_k, reductions=reductions)
        elif sample_method == 'interp':
            self.conv = GridFeatureToPointInterp(aabb_max, aabb_min, cat_in_point_features=True)
            self.transform = PointFeatureTransform(nn.Sequential(nn.Linear(grid_in_channels + point_in_channels, out_channels), nn.LayerNorm(out_channels)))
        else:
            raise NotImplementedError

    def forward(self, grid_features: 'GridFeatures', point_features: 'PointFeatures') ->PointFeatures:
        out_point_features = self.conv(grid_features, point_features)
        if self.sample_method == 'interp':
            out_point_features = self.transform(out_point_features)
        return out_point_features


class GridFeatureGroupToPoint(nn.Module):
    """GridFeatureGroupToPoint."""

    def __init__(self, grid_in_channels: 'int', point_in_channels: 'int', out_channels: 'int', grid_feature_group_size: 'int', aabb_max: 'Tuple[float, float, float]', aabb_min: 'Tuple[float, float, float]', use_rel_pos: 'bool'=True, use_rel_pos_embed: 'bool'=False, pos_embed_dim: 'int'=32, sample_method: "Literal['graphconv', 'interp']"='graphconv', neighbor_search_type: "Literal['radius', 'knn']"='radius', knn_k: 'int'=16, reductions: 'List[REDUCTION_TYPES]'=['mean']) ->None:
        super().__init__()
        self.conv_list = nn.ModuleList()
        assert out_channels % 2 == 0
        for i in range(grid_feature_group_size):
            self.conv_list.append(GridFeatureToPoint(grid_in_channels=grid_in_channels, point_in_channels=point_in_channels, out_channels=out_channels // 2, aabb_max=aabb_max, aabb_min=aabb_min, use_rel_pos=use_rel_pos, use_rel_pos_embed=use_rel_pos_embed, pos_embed_dim=pos_embed_dim, sample_method=sample_method, neighbor_search_type=neighbor_search_type, knn_k=knn_k, reductions=reductions))

    def forward(self, grid_features_group: 'GridFeatureGroup', point_features: 'PointFeatures') ->PointFeatures:
        assert len(grid_features_group) == len(self.conv_list)
        out_point_features: 'PointFeatures' = self.conv_list[0](grid_features_group[0], point_features)
        out_point_features_add: 'PointFeatures' = out_point_features
        out_point_features_mul: 'PointFeatures' = out_point_features
        for i in range(1, len(grid_features_group)):
            curr = self.conv_list[i](grid_features_group[i], point_features)
            out_point_features_add += curr
            out_point_features_mul *= curr
        out_point_features = PointFeatures(vertices=point_features.vertices, features=torch.cat((out_point_features_add.features, out_point_features_mul.features), dim=-1))
        return out_point_features


memory_format_to_axis_index = {GridFeaturesMemoryFormat.b_xc_y_z: 0, GridFeaturesMemoryFormat.b_yc_x_z: 1, GridFeaturesMemoryFormat.b_zc_x_y: 2, GridFeaturesMemoryFormat.b_x_y_z_c: -1}


class FIGConvUNet(BaseModel):
    """Factorized Implicit Global Convolutional U-Net.

    The FIGConvUNet is a U-Net architecture that uses factorized implicit global
    convolutional layers to create U-shaped architecture. The advantage of using
    FIGConvolution is that it can handle high resolution 3D data efficiently
    using a set of factorized grids.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', hidden_channels: 'List[int]', num_levels: 'int'=3, num_down_blocks: 'Union[int, List[int]]'=1, num_up_blocks: 'Union[int, List[int]]'=1, mlp_channels: 'List[int]'=[512, 512], aabb_max: 'Tuple[float, float, float]'=(1.0, 1.0, 1.0), aabb_min: 'Tuple[float, float, float]'=(0.0, 0.0, 0.0), voxel_size: 'Optional[float]'=None, resolution_memory_format_pairs: 'List[Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]]'=[(GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)), (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)), (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2))], use_rel_pos: 'bool'=True, use_rel_pos_embed: 'bool'=True, pos_encode_dim: 'int'=32, communication_types: "List[Literal['mul', 'sum']]"=['sum'], to_point_sample_method: "Literal['graphconv', 'interp']"='graphconv', neighbor_search_type: "Literal['knn', 'radius']"='radius', knn_k: 'int'=16, reductions: 'List[REDUCTION_TYPES]'=['mean'], drag_loss_weight: 'Optional[float]'=None, pooling_type: "Literal['attention', 'max', 'mean']"='max', pooling_layers: 'List[int]'=None):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        compressed_spatial_dims = []
        self.grid_feature_group_size = len(resolution_memory_format_pairs)
        self.point_feature_to_grids = nn.ModuleList()
        self.aabb_length = torch.tensor(aabb_max) - torch.tensor(aabb_min)
        self.min_voxel_edge_length = torch.tensor([np.inf, np.inf, np.inf])
        for mem_fmt, res in resolution_memory_format_pairs:
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])
            to_grid = nn.Sequential(PointFeatureToGrid(in_channels=in_channels, out_channels=hidden_channels[0], aabb_max=aabb_max, aabb_min=aabb_min, voxel_size=voxel_size, resolution=res, use_rel_pos=use_rel_pos, use_rel_pos_encode=use_rel_pos_embed, pos_encode_dim=pos_encode_dim, reductions=reductions, neighbor_search_type=neighbor_search_type, knn_k=knn_k), GridFeatureMemoryFormatConverter(memory_format=mem_fmt))
            self.point_feature_to_grids.append(to_grid)
            voxel_size = self.aabb_length / torch.tensor(res)
            self.min_voxel_edge_length = torch.min(self.min_voxel_edge_length, voxel_size)
        self.compressed_spatial_dims = compressed_spatial_dims
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)
        for level in range(num_levels):
            down_block = [GridFeatureConv2DBlocksAndIntraCommunication(in_channels=hidden_channels[level], out_channels=hidden_channels[level + 1], kernel_size=kernel_size, stride=2, compressed_spatial_dims=compressed_spatial_dims, communication_types=communication_types)]
            for _ in range(1, num_down_blocks[level]):
                down_block.append(GridFeatureConv2DBlocksAndIntraCommunication(in_channels=hidden_channels[level + 1], out_channels=hidden_channels[level + 1], kernel_size=kernel_size, stride=1, compressed_spatial_dims=compressed_spatial_dims, communication_types=communication_types))
            down_block = nn.Sequential(*down_block)
            self.down_blocks.append(down_block)
            up_block = [GridFeatureConv2DBlocksAndIntraCommunication(in_channels=hidden_channels[level + 1], out_channels=hidden_channels[level], kernel_size=kernel_size, up_stride=2, compressed_spatial_dims=compressed_spatial_dims, communication_types=communication_types)]
            for _ in range(1, num_up_blocks[level]):
                up_block.append(GridFeatureConv2DBlocksAndIntraCommunication(in_channels=hidden_channels[level], out_channels=hidden_channels[level], kernel_size=kernel_size, up_stride=1, compressed_spatial_dims=compressed_spatial_dims, communication_types=communication_types))
            up_block = nn.Sequential(*up_block)
            self.up_blocks.append(up_block)
        self.convert_to_orig = GridFeatureMemoryFormatConverter(memory_format=GridFeaturesMemoryFormat.b_x_y_z_c)
        if pooling_layers is None:
            pooling_layers = [num_levels]
        else:
            assert isinstance(pooling_layers, list), f'pooling_layers must be a list, got {type(pooling_layers)}.'
            for layer in pooling_layers:
                assert layer <= num_levels, f'pooling_layer {layer} is greater than num_levels {num_levels}.'
        self.pooling_layers = pooling_layers
        grid_pools = [GridFeatureGroupPool(in_channels=hidden_channels[layer], out_channels=mlp_channels[0], compressed_spatial_dims=self.compressed_spatial_dims, pooling_type=pooling_type) for layer in pooling_layers]
        self.grid_pools = nn.ModuleList(grid_pools)
        self.mlp = MLP(mlp_channels[0] * len(self.compressed_spatial_dims) * len(pooling_layers), mlp_channels[-1], mlp_channels, use_residual=True, activation=nn.GELU)
        self.mlp_projection = nn.Linear(mlp_channels[-1], 1)
        self.to_point = GridFeatureGroupToPoint(grid_in_channels=hidden_channels[0], point_in_channels=in_channels, out_channels=hidden_channels[0] * 2, grid_feature_group_size=self.grid_feature_group_size, aabb_max=aabb_max, aabb_min=aabb_min, use_rel_pos=use_rel_pos, use_rel_pos_embed=use_rel_pos_embed, pos_embed_dim=pos_encode_dim, sample_method=to_point_sample_method, neighbor_search_type=neighbor_search_type, knn_k=knn_k, reductions=reductions)
        self.projection = PointFeatureTransform(nn.Sequential(nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2), nn.LayerNorm(hidden_channels[0] * 2), nn.GELU(), nn.Linear(hidden_channels[0] * 2, out_channels)))
        self.pad_to_match = GridFeatureGroupPadToMatch()
        vertex_to_point_features = VerticesToPointFeatures(embed_dim=pos_encode_dim, out_features=hidden_channels[0], use_mlp=True, pos_embed_range=aabb_max[0] - aabb_min[0])
        self.vertex_to_point_features = vertex_to_point_features
        if drag_loss_weight is not None:
            self.drag_loss_weight = drag_loss_weight

    def _grid_forward(self, point_features: 'PointFeatures'):
        grid_feature_group = GridFeatureGroup([to_grid(point_features) for to_grid in self.point_feature_to_grids])
        down_grid_feature_groups = [grid_feature_group]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_feature_groups[-1])
            down_grid_feature_groups.append(out_features)
        pooled_feats = []
        for grid_pool, layer in zip(self.grid_pools, self.pooling_layers):
            pooled_feats.append(grid_pool(down_grid_feature_groups[layer]))
        if len(pooled_feats) > 1:
            pooled_feats = torch.cat(pooled_feats, dim=-1)
        else:
            pooled_feats = pooled_feats[0]
        drag_pred = self.mlp_projection(self.mlp(pooled_feats))
        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](down_grid_feature_groups[level + 1])
            padded_down_features = self.pad_to_match(up_grid_features, down_grid_feature_groups[level])
            up_grid_features = up_grid_features + padded_down_features
            down_grid_feature_groups[level] = up_grid_features
        grid_features = self.convert_to_orig(down_grid_feature_groups[0])
        return grid_features, drag_pred

    def forward(self, vertices: "Float[Tensor, 'B N 3']", features: "Optional[Float[Tensor, 'B N C']]"=None) ->Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)
        grid_features, drag_pred = self._grid_forward(point_features)
        out_point_features = self.to_point(grid_features, point_features)
        out_point_features = self.projection(out_point_features)
        return out_point_features.features, drag_pred


class PointFeatureMLP(PointFeatureTransform):
    """PointFeatureMLP."""

    def __init__(self, in_channels: 'int', out_channels: 'int', hidden_channels: 'Optional[int]'=None, multiplier: 'int'=2, nonlinearity: 'nn.Module'=nn.GELU):
        if hidden_channels is None:
            hidden_channels = multiplier * out_channels
        PointFeatureTransform.__init__(self, nn.Sequential(nn.Linear(in_channels, hidden_channels), nonlinearity(), nn.Linear(hidden_channels, out_channels)))


class PointFeatureConvBlock(nn.Module):
    """ConvBlock has two convolutions with a residual connection."""

    def __init__(self, in_channels: 'int', out_channels: 'int', radius: 'float', use_rel_pos: 'bool'=True, use_rel_pos_encode: 'bool'=False, pos_encode_dim: 'int'=32, reductions: 'List[REDUCTION_TYPES]'=['mean'], downsample_voxel_size: 'Optional[float]'=None, pos_encode_range: 'float'=4, out_point_feature_type: "Literal['provided', 'downsample', 'same']"='same', provided_in_channels: 'Optional[int]'=None, neighbor_search_type: "Literal['radius', 'knn']"='radius', knn_k: 'Optional[int]'=None):
        super().__init__()
        self.downsample_voxel_size = downsample_voxel_size
        self.out_point_feature_type = out_point_feature_type
        self.conv1 = PointFeatureConv(in_channels=in_channels, out_channels=out_channels, radius=radius, use_rel_pos=use_rel_pos, use_rel_pos_encode=use_rel_pos_encode, pos_encode_dim=pos_encode_dim, reductions=reductions, downsample_voxel_size=downsample_voxel_size, pos_encode_range=pos_encode_range, out_point_feature_type=out_point_feature_type, provided_in_channels=provided_in_channels, neighbor_search_type=neighbor_search_type, knn_k=knn_k)
        self.conv2 = PointFeatureConv(in_channels=out_channels, out_channels=out_channels, radius=radius, use_rel_pos=use_rel_pos, pos_encode_range=pos_encode_range, reductions=reductions, out_point_feature_type='same', neighbor_search_type=neighbor_search_type, knn_k=knn_k)
        self.norm1 = PointFeatureTransform(nn.LayerNorm(out_channels))
        self.norm2 = PointFeatureTransform(nn.LayerNorm(out_channels))
        if out_point_feature_type == 'provided':
            self.shortcut = PointFeatureMLP(in_channels=provided_in_channels, out_channels=out_channels)
        elif in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = PointFeatureMLP(in_channels, out_channels)
        self.nonlinear = PointFeatureTransform(nn.GELU())

    def forward(self, in_point_features: "PointFeatures['B N C1']", out_point_features: "Optional[PointFeatures['B M C2']]"=None) ->PointFeatures['B N C2']:
        if self.out_point_feature_type == 'provided':
            assert out_point_features is not None, 'out_point_features must be provided for the provided type'
            out = self.conv1(in_point_features, out_point_features)
        elif self.out_point_feature_type == 'downsample':
            assert out_point_features is None
            out_point_features = in_point_features.voxel_down_sample(self.downsample_voxel_size)
            out = self.conv1(in_point_features)
        elif self.out_point_feature_type == 'same':
            assert out_point_features is None
            out_point_features = in_point_features
            out = self.conv1(in_point_features)
        out = self.nonlinear(self.norm1(out))
        out = self.norm2(self.conv2(out))
        return self.nonlinear(out + self.shortcut(out_point_features))


class FNO1DEncoder(nn.Module):
    """1D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(layers.SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0]))
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


class FNO2DEncoder(nn.Module):
    """2D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]
        self.pad = padding[:2]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(layers.SpectralConv2d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]))
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if x.dim() != 4:
            raise ValueError('Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO')
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))


class FNO3DEncoder(nn.Module):
    """3D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]
        self.pad = padding[:3]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(layers.SpectralConv3d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], num_fno_modes[2]))
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1], :self.ipad[2]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


class FNO4DEncoder(nn.Module):
    """4D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(self, in_channels: 'int'=1, num_fno_layers: 'int'=4, fno_layer_size: 'int'=32, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'Union[int, List[int]]'=8, padding_type: 'str'='constant', activation_fn: 'nn.Module'=nn.GELU(), coord_features: 'bool'=True) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn
        if self.coord_features:
            self.in_channels = self.in_channels + 4
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        padding = padding + [0, 0, 0, 0]
        self.pad = padding[:4]
        self.ipad = [(-pad if pad > 0 else None) for pad in self.pad]
        self.padding_type = padding_type
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes, num_fno_modes]
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) ->None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(layers.ConvNdFCLayer(self.in_channels, int(self.fno_width / 2)))
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(layers.ConvNdFCLayer(int(self.fno_width / 2), self.fno_width))

    def build_fno(self, num_fno_modes: 'List[int]') ->None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(layers.SpectralConv4d(self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], num_fno_modes[2], num_fno_modes[3]))
            self.conv_layers.append(layers.ConvNdKernel1Layer(self.fno_width, self.fno_width))

    def forward(self, x: 'Tensor') ->Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[3], 0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)
        x = x[..., :self.ipad[0], :self.ipad[1], :self.ipad[2], :self.ipad[3]]
        return x

    def meshgrid(self, shape: 'List[int]', device: 'torch.device') ->Tensor:
        """Creates 4D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z, size_t = shape[0], shape[2], shape[3], shape[4], shape[5]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(grid_x, grid_y, grid_z, grid_t, indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)

    def grid_to_points(self, value: 'Tensor') ->Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 5, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: 'Tensor', shape: 'List[int]') ->Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], shape[5], value.size(-1))
        return torch.permute(output, (0, 5, 1, 2, 3, 4))


class FNO(Module):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True

    Example
    -------
    >>> # define the 2d FNO model
    >>> model = modulus.models.fno.FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Li, Zongyi, et al. "Fourier neural operator for parametric
    partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', decoder_layers: 'int'=1, decoder_layer_size: 'int'=32, decoder_activation_fn: 'str'='silu', dimension: 'int'=2, latent_channels: 'int'=32, num_fno_layers: 'int'=4, num_fno_modes: 'Union[int, List[int]]'=16, padding: 'int'=8, padding_type: 'str'='constant', activation_fn: 'str'='gelu', coord_features: 'bool'=True) ->None:
        super().__init__(meta=MetaData())
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension
        self.decoder_net = FullyConnected(in_features=latent_channels, layer_size=decoder_layer_size, out_features=out_channels, num_layers=decoder_layers, activation_fn=decoder_activation_fn)
        FNOModel = self.getFNOEncoder()
        self.spec_encoder = FNOModel(in_channels, num_fno_layers=self.num_fno_layers, fno_layer_size=latent_channels, num_fno_modes=self.num_fno_modes, padding=self.padding, padding_type=self.padding_type, activation_fn=self.activation_fn, coord_features=self.coord_features)

    def getFNOEncoder(self):
        if self.dimension == 1:
            return FNO1DEncoder
        elif self.dimension == 2:
            return FNO2DEncoder
        elif self.dimension == 3:
            return FNO3DEncoder
        elif self.dimension == 4:
            return FNO4DEncoder
        else:
            raise NotImplementedError('Invalid dimensionality. Only 1D, 2D, 3D and 4D FNO implemented')

    def forward(self, x: 'Tensor') ->Tensor:
        y_latent = self.spec_encoder(x)
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)
        y = self.decoder_net(y_latent)
        y = self.spec_encoder.points_to_grid(y, y_shape)
        return y


@functools.lru_cache(maxsize=None)
def silu_backward_for(fd: 'FusionDefinition', dtype: 'torch.dtype', dim: 'int', size: 'torch.Size', stride: 'Tuple[int, ...]'):
    """
    nvfuser frontend implmentation of SiLU backward as a fused kernel and with
    activations recomputation

    Parameters
    ----------
    fd : FusionDefition
        nvFuser's FusionDefition class
    dtype : torch.dtype
        Data type to use for the implementation
    dim : int
        Dimension of the input tensor
    size : torch.Size
        Size of the input tensor
    stride : Tuple[int, ...]
        Stride of the input tensor
    """
    try:
        dtype = _torch_dtype_to_nvfuser[dtype]
    except KeyError:
        raise TypeError('Unsupported dtype')
    x = fd.define_tensor(shape=[-1] * dim, contiguity=nvfuser.compute_contiguity(size, stride), dtype=dtype)
    one = fd.define_constant(1.0)
    y = fd.ops.sigmoid(x)
    grad_input = fd.ops.mul(y, fd.ops.add(one, fd.ops.mul(x, fd.ops.sub(one, y))))
    grad_input = fd.ops.cast(grad_input, dtype)
    fd.add_output(grad_input)


class CustomSiLuLinearAutogradFunction(torch.autograd.Function):
    """Custom SiLU + Linear autograd function"""

    @staticmethod
    def forward(ctx, features: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor') ->torch.Tensor:
        out = F.silu(features)
        out = F.linear(out, weight, bias)
        ctx.save_for_backward(features, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: 'torch.Tensor') ->Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """backward pass of the SiLU + Linear function"""
        need_dgrad, need_wgrad, need_bgrad = ctx.needs_input_grad
        features, weight = ctx.saved_tensors
        grad_features = None
        grad_weight = None
        grad_bias = None
        if need_bgrad:
            grad_bias = grad_output.sum(dim=0)
        if need_wgrad:
            out = F.silu(features)
            grad_weight = grad_output.T @ out
        if need_dgrad:
            grad_features = grad_output @ weight
            with FusionDefinition() as fd:
                silu_backward_for(fd, features.dtype, features.dim(), features.size(), features.stride())
            grad_silu = fd.execute([features])[0]
            grad_features = grad_features * grad_silu
        return grad_features, grad_weight, grad_bias


class MeshGraphMLP(nn.Module):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer.

    Parameters
    ----------
    input_dim : int
        dimensionality of the input features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : Union[int, None], optional
        number of hidden layers, by default 1
        if None is provided, the MLP will collapse to a Identity function
    activation_fn : nn.Module, optional
        , by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing recompute_activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, input_dim: 'int', output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'Union[int, None]'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', recompute_activation: 'bool'=False):
        super().__init__()
        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type not in ['LayerNorm', 'TELayerNorm']:
                    raise ValueError(f'Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm.')
                if norm_type == 'TELayerNorm' and te_imported:
                    norm_layer = te.LayerNorm
                elif norm_type == 'TELayerNorm' and not te_imported:
                    raise ValueError('TELayerNorm requires transformer-engine to be installed.')
                else:
                    norm_layer = getattr(nn, norm_type)
                layers.append(norm_layer(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()
        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def default_forward(self, x: 'Tensor') ->Tensor:
        """default forward pass of the MLP"""
        return self.model(x)

    @torch.jit.ignore()
    def custom_silu_linear_forward(self, x: 'Tensor') ->Tensor:
        """forward pass of the MLP where SiLU is recomputed in backward"""
        lin = self.model[0]
        hidden = lin(x)
        for i in range(1, self.hidden_layers + 1):
            lin = self.model[2 * i]
            hidden = CustomSiLuLinearAutogradFunction.apply(hidden, lin.weight, lin.bias)
        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers + 1]
            hidden = norm(hidden)
        return hidden

    def forward(self, x: 'Tensor') ->Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(x)
        return self.default_forward(x)


def broadcast(src: 'torch.Tensor', ref: 'torch.Tensor', dim: 'int') ->torch.Tensor:
    """helper function for scatter_reduce"""
    size = (1,) * dim + (-1,) + (1,) * (ref.dim() - dim - 1)
    return src.view(size).expand_as(ref)


def scatter_sum(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class GraphMessagePassing(nn.Module):
    """Graph Message Passing (GMP) block."""

    def __init__(self, latent_dim, hidden_layer, pos_dim):
        """
        Initialize the GMP block.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        hidden_layer : int
            Number of hidden layers.
        pos_dim : int
            Dimension of the positional encoding.
        """
        super().__init__()
        self.mlp_node = MeshGraphMLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer)
        edge_info_in_len = 2 * latent_dim + pos_dim + 1
        self.mlp_edge = MeshGraphMLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer)
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        """
        Forward pass for GMP block.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        pos : torch.Tensor
            Node positional information of shape [B, N, pos_dim] or [N, pos_dim].

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        i, j = g[0], g[1]
        if len(x.shape) == 3:
            B, _, _ = x.shape
            x_i, x_j = x[:, i], x[:, j]
        elif len(x.shape) == 2:
            x_i, x_j = x[i], x[j]
        else:
            raise ValueError(f'Only implemented for dim 2 and 3, got {x.shape}')
        if len(pos.shape) == 3:
            pi, pj = pos[:, i], pos[:, j]
        elif len(pos.shape) == 2:
            pi, pj = pos[i], pos[j]
        else:
            raise ValueError(f'Only implemented for dim 2 and 3, got {x.shape}')
        dir = pi - pj
        norm = torch.norm(dir, dim=-1, keepdim=True)
        fiber = torch.cat([dir, norm], dim=-1)
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(B, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge(tmp)
        aggr_out = scatter_sum(edge_embedding, j, dim=-2, dim_size=x.shape[-2])
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node(tmp) + x


class Unpool(nn.Module):
    """Unpooling layer for graph neural networks."""

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        """
        Forward pass for the unpooling layer.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape [N, C] or [B, N, C].
        pre_node_num : int
            Number of nodes in the previous upper layer.
        idx : torch.Tensor
            Relative indices (in the previous upper layer) for unpooling of shape [N] or [B, N].

        Returns
        -------
        torch.Tensor
            Unpooled node features of shape [pre_node_num, C] or [B, pre_node_num, C].
        """
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h
        return new_h


def degree(index: 'torch.Tensor', num_nodes: 'Optional[int]'=None, dtype: 'Optional[torch.dtype]'=None) ->torch.Tensor:
    """Computes the (unweighted) degree of a given one-dimensional index tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = torch.max(index) + 1
    N = int(N)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


class WeightedEdgeConv(nn.Module):
    """Weighted Edge Convolution layer for transition between layers."""

    def __init__(self, *args):
        super(WeightedEdgeConv, self).__init__()

    def forward(self, x, g, ew, aggragating=True):
        """
        Forward pass for WeightedEdgeConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        ew : torch.Tensor
            Edge weights for convolution of shape [E].
        aggragating : bool, optional
            If True, aggregate messages (used in down pass); if False, return messages (used in up pass).

        Returns
        -------
        torch.Tensor
            Aggregated or scattered node features.
        """
        i, j = g[0], g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError('Only implemented for dim 2 and 3')
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter_sum(weighted_info, target_index, dim=-2, dim_size=x.shape[-2])
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        """
        Calculate the edge weights for later use in forward.

        Parameters
        ----------
        w : torch.Tensor
            Node weights of shape [N, 1].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].

        Returns
        -------
        tuple
            Edge weights for convolution and aggregated node weights (used for iteratively calculating this in the next layer).
        """
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i, j = g[0], g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class BistrideGraphMessagePassing(nn.Module):
    """Bistride Graph Message Passing (BSGMP) network for hierarchical graph processing."""

    def __init__(self, unet_depth, latent_dim, hidden_layer, pos_dim):
        """
        Initializes the BSGMP network.

        Parameters
        ----------
        unet_depth : int
            UNet depth in the network, excluding top level.
        latent_dim : int
            Latent dimension for the graph nodes and edges.
        hidden_layer : int
            Number of hidden layers in the MLPs.
        pos_dim : int
            Dimension of the physical position (in Euclidean space).
        """
        super().__init__()
        self.bottom_gmp = GraphMessagePassing(latent_dim, hidden_layer, pos_dim)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.unet_depth = unet_depth
        self.edge_conv = WeightedEdgeConv()
        for _ in range(self.unet_depth):
            self.down_gmps.append(GraphMessagePassing(latent_dim, hidden_layer, pos_dim))
            self.up_gmps.append(GraphMessagePassing(latent_dim, hidden_layer, pos_dim))
            self.unpools.append(Unpool())

    def forward(self, h, m_ids, m_gs, pos):
        """
        Forward pass for the BSGMP network.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape [B, N, F] or [N, F].
        m_ids : list of torch.Tensor
            Indices for pooling/unpooling nodes at each level.
        m_gs : list of torch.Tensor
            Graph connectivity (edges) at each level.
        pos : torch.Tensor
            Node positional information of shape [B, N, D] or [N, D].

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        down_outs = []
        down_ps = []
        cts = []
        w = pos.new_ones((pos.shape[-2], 1))
        for i in range(self.unet_depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)
            if len(h.shape) == 3:
                h = h[:, m_ids[i]]
            elif len(h.shape) == 2:
                h = h[m_ids[i]]
            if len(pos.shape) == 3:
                pos = pos[:, m_ids[i]]
            elif len(pos.shape) == 2:
                pos = pos[m_ids[i]]
            w = w[m_ids[i]]
        h = self.bottom_gmp(h, m_gs[self.unet_depth], pos)
        for i in range(self.unet_depth):
            depth_idx = self.unet_depth - i - 1
            g, idx = m_gs[depth_idx], m_ids[depth_idx]
            h = self.unpools[i](h, down_outs[depth_idx].shape[-2], idx)
            h = self.edge_conv(h, g, cts[depth_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[depth_idx])
            h = h.add(down_outs[depth_idx])
        return h


class GraphCastEncoderEmbedder(nn.Module):
    """GraphCast feature embedder for gird node features, multimesh node features,
    grid2mesh edge features, and multimesh edge features.

    Parameters
    ----------
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim : int, optional
        Dimensionality of the embedded features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, input_dim_grid_nodes: 'int'=474, input_dim_mesh_nodes: 'int'=3, input_dim_edges: 'int'=4, output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', recompute_activation: 'bool'=False):
        super().__init__()
        self.grid_node_mlp = MeshGraphMLP(input_dim=input_dim_grid_nodes, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.mesh_node_mlp = MeshGraphMLP(input_dim=input_dim_mesh_nodes, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.grid2mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    def forward(self, grid_nfeat: 'Tensor', mesh_nfeat: 'Tensor', g2m_efeat: 'Tensor', mesh_efeat: 'Tensor') ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        grid_nfeat = self.grid_node_mlp(grid_nfeat)
        mesh_nfeat = self.mesh_node_mlp(mesh_nfeat)
        g2m_efeat = self.grid2mesh_edge_mlp(g2m_efeat)
        mesh_efeat = self.mesh_edge_mlp(mesh_efeat)
        return grid_nfeat, mesh_nfeat, g2m_efeat, mesh_efeat


class GraphCastDecoderEmbedder(nn.Module):
    """GraphCast feature embedder for mesh2grid edge features

    Parameters
    ----------
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim : int, optional
        Dimensionality of the embedded features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, input_dim_edges: 'int'=4, output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', recompute_activation: 'bool'=False):
        super().__init__()
        self.mesh2grid_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    def forward(self, m2g_efeat: 'Tensor') ->Tensor:
        m2g_efeat = self.mesh2grid_edge_mlp(m2g_efeat)
        return m2g_efeat


def all_gather_v_bwd_wrapper(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, use_fp32: 'bool'=True, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    tmp = [torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device) for _ in range(comm_size)]
    scatter_list = list(torch.split(tensor, sizes, dim=dim))
    scatter_list = [t.contiguous() for t in scatter_list]
    dist.all_to_all(tmp, scatter_list, group=group)
    stack_dim = tensor.dim()
    tmp = torch.stack(tmp, dim=stack_dim)
    if use_fp32 and tmp.dtype.itemsize < 4 and tmp.dtype.is_floating_point:
        output = tmp.sum(dim=stack_dim, dtype=torch.float32)
        output = output
    else:
        output = tmp.sum(dim=stack_dim)
    return output


class AllGatherVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed AllGatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank. Its indended to be used in
    tensor-parallel settings on tensors which require gradients
    to be passed through.
    The backward pass performs an AllReduceV operation where
    each rank gathers its corresponding chunk of a global tensor
    from each other rank and sums up these individual gradients.
    """

    @staticmethod
    def forward(ctx, tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, use_fp32: 'bool'=True, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
        """forward pass of the Distributed AllGatherV primitive"""
        gathered_tensor = all_gather_v_wrapper(tensor, sizes, dim=dim, group=group)
        ctx.sizes = sizes
        ctx.group = group
        ctx.dim = dim
        ctx.use_fp32 = use_fp32
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor'):
        """backward pass of the of the Distributed AllGatherV primitive"""
        grad_tensor = all_gather_v_bwd_wrapper(grad_output, ctx.sizes, dim=ctx.dim, use_fp32=ctx.use_fp32, group=ctx.group)
        if not ctx.needs_input_grad[0]:
            grad_tensor = None
        return grad_tensor, None, None, None, None


def all_gather_v(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, use_fp32: 'bool'=True, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Autograd Wrapper for a distributed AllGatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank. Its indended to be used in
    tensor-parallel settings on tensors which require gradients
    to be passed through.
    The backward pass performs an AllReduceV operation where
    each rank gathers its corresponding chunk of a global tensor
    from each other rank and sums up these individual gradients.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        boolean flag to indicate whether to use FP32 precision for the
        reduction in the backward pass, by default True
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """
    return AllGatherVAutograd.apply(tensor, sizes, dim, use_fp32, group)


def gather_v_wrapper(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, dst: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements a distributed GatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the
        operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not 0 <= dst < comm_size:
        raise ValueError()
    if tensor.size(dim) != sizes[rank]:
        raise ValueError()
    if comm_size == 1:
        return tensor
    tensor_shape = list(tensor.shape)
    x_recv = [None] * comm_size
    x_send = [None] * comm_size
    for r in range(comm_size):
        if rank == dst:
            tensor_shape[dim] = sizes[r]
        else:
            tensor_shape[dim] = 0
        x_recv[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
        if r == dst:
            x_send[r] = tensor
        else:
            tensor_shape[dim] = 0
            x_send[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all(x_recv, x_send, group=group)
    if rank != dst:
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r]
            x_recv[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    output = torch.cat(x_recv, dim=dim)
    return output


def scatter_v_wrapper(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, src: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements a distributed ScatterV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dist.get_rank(group=group) == 0 and dim >= tensor.dim():
        raise ValueError()
    if not 0 <= src < comm_size:
        raise ValueError()
    tensor_shape = list(tensor.shape)
    x_send = [None] * comm_size
    x_recv = [None] * comm_size
    if rank == src:
        scatter_list = torch.split(tensor, sizes, dim=dim)
        scatter_list = [t.contiguous() for t in scatter_list]
        x_send = scatter_list
    else:
        for r in range(comm_size):
            tensor_shape[dim] = 0
            x_send[r] = torch.empty(tensor_shape, device=tensor.device, dtype=tensor.dtype)
    for r in range(comm_size):
        if r == src:
            tensor_shape[dim] = sizes[rank]
        else:
            tensor_shape[dim] = 0
        x_recv[r] = torch.empty(tensor_shape, device=tensor.device, dtype=tensor.dtype)
    dist.all_to_all(x_recv, x_send, group=group)
    return x_recv[src]


class GatherVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed GatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to a straightforward
    ScatterV primitive distributing the global gradient from the
    specified destination rank to all the other ranks.
    """

    @staticmethod
    def forward(ctx, tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, dst: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
        """forward pass of the distributed GatherV primitive"""
        gathered_tensor = gather_v_wrapper(tensor, sizes, dim=dim, dst=dst, group=group)
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.dst = dst
        ctx.group = group
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        """backward pass of the Distributed GatherV primitive"""
        grad_tensor = scatter_v_wrapper(grad_output, ctx.sizes, dim=ctx.dim, src=ctx.dst, group=ctx.group)
        if not ctx.needs_input_grad[0]:
            grad_tensor = None
        return grad_tensor, None, None, None, None


def gather_v(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, dst: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Autograd Wrapper for a distributed GatherV primitive.
    It is based on the idea of a single global tensor which is distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to a straightforward
    ScatterV primitive distributing the global gradient from the
    specified destination rank to all the other ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """
    return GatherVAutograd.apply(tensor, sizes, dim, dst, group)


def indexed_all_to_all_v_wrapper(tensor: 'torch.Tensor', indices: 'List[torch.Tensor]', sizes: 'List[List[int]]', dim: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()
    x_send = [tensor[idx] for idx in indices]
    x_recv = [None] * comm_size
    tensor_shape = list(tensor.shape)
    for r in range(comm_size):
        tensor_shape[dim] = sizes[r][rank]
        x_recv[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all(x_recv, x_send, group=group)
    tensor_to_recv = torch.cat(x_recv, dim=dim)
    return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(tensor: 'torch.Tensor', indices: 'List[torch.Tensor]', sizes: 'List[List[int]]', tensor_size_along_dim: 'int', use_fp32: 'bool'=True, dim: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        result of primitive corresponding to indexed global tensor
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()
    tensor_shape = list(tensor.shape)
    recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    send_sizes = [sizes[rank][i] for i in range(comm_size)]
    x_send = torch.split(tensor, recv_sizes, dim=dim)
    x_send = [t.contiguous() for t in x_send]
    x_recv = [None] * comm_size
    for r in range(comm_size):
        tensor_shape[dim] = send_sizes[r]
        x_recv[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all(x_recv, x_send, group=group)
    tensor_to_recv = torch.cat(x_recv, dim=dim)
    indices = torch.cat(indices, dim=0)
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32 and tensor.dtype.itemsize < 4 and tensor.dtype.is_floating_point:
        out = torch.zeros(tensor_shape, dtype=torch.float32, device=tensor.device)
        tensor_to_recv = tensor_to_recv
    else:
        out = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    out.index_add_(source=tensor_to_recv, index=indices, dim=dim)
    if out.dtype != tensor.dtype:
        out = out
    return out


class IndexedAllToAllVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for an Indexed AllToAllV primitive. It is based on the
    idea of a single global tensor which is distributed along a
    specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive. It is intended to be used in tensor-parallel settings
    on tensors which require gradients to be passed through.
    The backward pass more or less corresponds to the same operation as in the forward
    pass but with reversed roles and does an additional reduction of gathered gradients
    so that each rank finally will compute the overall gradient on its local tensor partition.
    """

    @staticmethod
    def forward(ctx, tensor: 'torch.Tensor', indices: 'List[torch.Tensor]', sizes: 'List[List[int]]', use_fp32: 'bool'=True, dim: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
        """forward pass of the Distributed IndexedAlltoAllV primitive"""
        tensor_to_recv = indexed_all_to_all_v_wrapper(tensor, indices, sizes, dim=dim, group=group)
        ctx.sizes = sizes
        ctx.use_fp32 = use_fp32
        ctx.group = group
        ctx.tensor_size_along_dim = tensor.size(dim)
        ctx.indices = indices
        ctx.dim = dim
        return tensor_to_recv

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        """backward pass of the Distributed IndexedAlltoAllV primitive"""
        grad_tensor = indexed_all_to_all_v_wrapper_bwd(grad_output, ctx.indices, ctx.sizes, tensor_size_along_dim=ctx.tensor_size_along_dim, use_fp32=ctx.use_fp32, dim=ctx.dim, group=ctx.group)
        if not ctx.needs_input_grad[0]:
            grad_tensor = None
        return grad_tensor, None, None, None, None, None, None


def indexed_all_to_all_v(tensor: 'torch.Tensor', indices: 'List[torch.Tensor]', sizes: 'List[List[int]]', use_fp32: 'bool'=True, dim: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Autograd Wrapper for an Indexed AllToAllV primitive. It is based on the
    idea of a single global tensor which is distributed along a
    specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive. It is intended to be used in tensor-parallel settings
    on tensors which require gradients to be passed through.
    The backward pass more or less corresponds to the same operation as in the forward
    pass but with reversed roles and does an additional reduction of gathered gradients
    so that each rank finally will compute the overall gradient on its local tensor partition.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    use_fp32 : bool, optional
        flag to specify whether to use FP32 precision in the reduction
        in the backward pass, by default True
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """
    return IndexedAllToAllVAutograd.apply(tensor, indices, sizes, use_fp32, dim, group)


class ScatterVAutograd(torch.autograd.Function):
    """
    Autograd Wrapper for Distributed ScatterV. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to an GatherV primitive
    gathering local gradients from all the other ranks into a single
    global gradient on the specified source rank.
    """

    @staticmethod
    def forward(ctx, tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, src: 'int'=0, group=Optional[dist.ProcessGroup]) ->torch.Tensor:
        """forward pass of the Distributed ScatterV primitive"""
        scattered_tensor = scatter_v_wrapper(tensor, sizes, dim=dim, src=src, group=group)
        ctx.tensor = tensor
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.src = src
        ctx.group = group
        return scattered_tensor

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        """backward pass of the Distributed ScatterV primitive"""
        grad_tensor = gather_v_wrapper(grad_output, ctx.sizes, dim=ctx.dim, dst=ctx.src, group=ctx.group)
        if not ctx.needs_input_grad[0]:
            grad_tensor = None
        return grad_tensor, None, None, None, None


def scatter_v(tensor: 'torch.Tensor', sizes: 'List[int]', dim: 'int'=0, src: 'int'=0, group: 'Optional[dist.ProcessGroup]'=None) ->torch.Tensor:
    """
    Autograd Wrapper for Distributed ScatterV. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank. It is intended to be used in
    tensor-parallel settings on tensors which require gradients to
    be passed through.
    The backward pass corresponds to an GatherV primitive
    gathering local gradients from all the other ranks into a single
    global gradient on the specified source rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """
    return ScatterVAutograd.apply(tensor, sizes, dim, src, group)


class DistributedGraph:

    def __init__(self, global_offsets: 'torch.Tensor', global_indices: 'torch.Tensor', partition_size: 'int', graph_partition_group_name: 'str'=None, graph_partition: 'Optional[GraphPartition]'=None):
        """
        Utility Class representing a distributed graph based on a given
        partitioning of a CSC graph structure. By default, a naive node-wise
        partitioning scheme is applied, see ``partition_graph_nodewise`` for
        details on that. This class then wraps necessary communication primitives
        to access all relevant feature buffers related to the graph.

        Parameters
        ----------
        global_offsets : torch.Tensor
            CSC offsets, can live on the CPU
        global_indices : torch.Tensor
            CSC indices, can live on the CPU
        partition_size : int
            Number of process groups across which graphs are distributed, expected to
            be larger than 1, i.e. an actual partition distributed among multiple ranks.
        partition_group_name : str, default=None
            Name of process group across which graphs are distributed. Passing no process
            group name leads to a parallelism across the default process group.
            Otherwise, the group size of a process group is expected to match partition_size.
        graph_partition : GraphPartition, optional
            Optional graph_partition, if passed as None, the naive
            node-wise partitioning scheme will be applied to global_offsets and global_indices,
            otherwise, these will be ignored and the passed partition will be used instead.
        """
        dist_manager = DistributedManager()
        self.device = dist_manager.device
        self.partition_rank = dist_manager.group_rank(name=graph_partition_group_name)
        self.partition_size = dist_manager.group_size(name=graph_partition_group_name)
        error_msg = f'Passed partition_size does not correspond to size of process_group, got {partition_size} and {self.partition_size} respectively.'
        if self.partition_size != partition_size:
            raise AssertionError(error_msg)
        self.process_group = dist_manager.group(name=graph_partition_group_name)
        if graph_partition is None:
            self.graph_partition = partition_graph_nodewise(global_offsets, global_indices, self.partition_size, self.partition_rank, self.device)
        else:
            error_msg = f'Passed graph_partition.partition_size does not correspond to size of process_group, got {graph_partition.partition_size} and {self.partition_size} respectively.'
            if graph_partition.partition_size != self.partition_size:
                raise AssertionError(error_msg)
            error_msg = f'Passed graph_partition.device does not correspond to device of this rank, got {graph_partition.device} and {self.device} respectively.'
            if graph_partition.device != self.device:
                raise AssertionError(error_msg)
            self.graph_partition = graph_partition
        send_sizes = self.graph_partition.sizes[self.graph_partition.partition_rank]
        recv_sizes = [p[self.graph_partition.partition_rank] for p in self.graph_partition.sizes]
        msg = f'GraphPartition(rank={self.graph_partition.partition_rank}, '
        msg += f'num_local_src_nodes={self.graph_partition.num_local_src_nodes}, '
        msg += f'num_local_dst_nodes={self.graph_partition.num_local_dst_nodes}, '
        msg += f'num_partitioned_src_nodes={self.graph_partition.num_src_nodes_in_each_partition[self.graph_partition.partition_rank]}, '
        msg += f'num_partitioned_dst_nodes={self.graph_partition.num_dst_nodes_in_each_partition[self.graph_partition.partition_rank]}, '
        msg += f'send_sizes={send_sizes}, recv_sizes={recv_sizes})'
        None
        dist.barrier(self.process_group)

    def get_src_node_features_in_partition(self, global_node_features: 'torch.Tensor', scatter_features: 'bool'=False, src_rank: 'int'=0) ->torch.Tensor:
        if scatter_features:
            global_node_features = global_node_features[self.graph_partition.map_concatenated_local_src_ids_to_global]
            return scatter_v(global_node_features, self.graph_partition.num_src_nodes_in_each_partition, dim=0, src=src_rank, group=self.process_group)
        return global_node_features[self.graph_partition.map_partitioned_src_ids_to_global, :]

    def get_src_node_features_in_local_graph(self, partitioned_src_node_features: 'torch.Tensor') ->torch.Tensor:
        return indexed_all_to_all_v(partitioned_src_node_features, indices=self.graph_partition.scatter_indices, sizes=self.graph_partition.sizes, use_fp32=True, dim=0, group=self.process_group)

    def get_dst_node_features_in_partition(self, global_node_features: 'torch.Tensor', scatter_features: 'bool'=False, src_rank: 'int'=0) ->torch.Tensor:
        if scatter_features:
            global_node_features = global_node_features[self.graph_partition.map_concatenated_local_dst_ids_to_global]
            return scatter_v(global_node_features, self.graph_partition.num_dst_nodes_in_each_partition, dim=0, src=src_rank, group=self.process_group)
        return global_node_features[self.graph_partition.map_partitioned_dst_ids_to_global, :]

    def get_dst_node_features_in_local_graph(self, partitioned_dst_node_features: 'torch.Tensor') ->torch.Tensor:
        return partitioned_dst_node_features

    def get_edge_features_in_partition(self, global_edge_features: 'torch.Tensor', scatter_features: 'bool'=False, src_rank: 'int'=0) ->torch.Tensor:
        if scatter_features:
            global_edge_features = global_edge_features[self.graph_partition.map_concatenated_local_edge_ids_to_global]
            return scatter_v(global_edge_features, self.graph_partition.num_indices_in_each_partition, dim=0, src=src_rank, group=self.process_group)
        return global_edge_features[self.graph_partition.map_partitioned_edge_ids_to_global, :]

    def get_edge_features_in_local_graph(self, partitioned_edge_features: 'torch.Tensor') ->torch.Tensor:
        return partitioned_edge_features

    def get_global_src_node_features(self, partitioned_node_features: 'torch.Tensor', get_on_all_ranks: 'bool'=True, dst_rank: 'int'=0) ->torch.Tensor:
        error_msg = f'Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively.'
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)
        if not get_on_all_ranks:
            global_node_feat = gather_v(partitioned_node_features, self.graph_partition.num_src_nodes_in_each_partition, dim=0, dst=dst_rank, group=self.process_group)
            if self.graph_partition.partition_rank == dst_rank:
                global_node_feat = global_node_feat[self.graph_partition.map_global_src_ids_to_concatenated_local]
            return global_node_feat
        global_node_feat = all_gather_v(partitioned_node_features, self.graph_partition.num_src_nodes_in_each_partition, dim=0, use_fp32=True, group=self.process_group)
        global_node_feat = global_node_feat[self.graph_partition.map_global_src_ids_to_concatenated_local]
        return global_node_feat

    def get_global_dst_node_features(self, partitioned_node_features: 'torch.Tensor', get_on_all_ranks: 'bool'=True, dst_rank: 'int'=0) ->torch.Tensor:
        error_msg = f'Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively.'
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)
        if not get_on_all_ranks:
            global_node_feat = gather_v(partitioned_node_features, self.graph_partition.num_dst_nodes_in_each_partition, dim=0, dst=dst_rank, group=self.process_group)
            if self.graph_partition.partition_rank == dst_rank:
                global_node_feat = global_node_feat[self.graph_partition.map_global_dst_ids_to_concatenated_local]
            return global_node_feat
        global_node_feat = all_gather_v(partitioned_node_features, self.graph_partition.num_dst_nodes_in_each_partition, dim=0, use_fp32=True, group=self.process_group)
        global_node_feat = global_node_feat[self.graph_partition.map_global_dst_ids_to_concatenated_local]
        return global_node_feat

    def get_global_edge_features(self, partitioned_edge_features: 'torch.Tensor', get_on_all_ranks: 'bool'=True, dst_rank: 'int'=0) ->torch.Tensor:
        error_msg = f'Passed partitioned_edge_features.device does not correspond to device of this rank, got {partitioned_edge_features.device} and {self.device} respectively.'
        if partitioned_edge_features.device != self.device:
            raise AssertionError(error_msg)
        if not get_on_all_ranks:
            global_edge_feat = gather_v(partitioned_edge_features, self.graph_partition.num_indices_in_each_partition, dim=0, dst=dst_rank, group=self.process_group)
            if self.graph_partition.partition_rank == dst_rank:
                global_edge_feat = global_edge_feat[self.graph_partition.map_global_edge_ids_to_concatenated_local]
            return global_edge_feat
        global_edge_feat = all_gather_v(partitioned_edge_features, self.graph_partition.num_indices_in_each_partition, dim=0, use_fp32=True, group=self.process_group)
        global_edge_feat = global_edge_feat[self.graph_partition.map_global_edge_ids_to_concatenated_local]
        return global_edge_feat


def concat_message_function(edges: 'Tensor') ->Dict[str, Tensor]:
    """Concatenates source node, destination node, and edge features.

    Parameters
    ----------
    edges : Tensor
        Edges.

    Returns
    -------
    Dict[Tensor]
        Concatenated source node, destination node, and edge features.
    """
    cat_feat = torch.cat((edges.data['x'], edges.src['x'], edges.dst['x']), dim=1)
    return {'cat_feat': cat_feat}


@torch.jit.ignore()
def concat_efeat_dgl(efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[torch.Tensor, torch.Tensor]]', graph: 'DGLGraph') ->Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor, Tensor]
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        with graph.local_scope():
            graph.srcdata['x'] = src_feat
            graph.dstdata['x'] = dst_feat
            graph.edata['x'] = efeat
            graph.apply_edges(concat_message_function)
            return graph.edata['cat_feat']
    with graph.local_scope():
        graph.ndata['x'] = nfeat
        graph.edata['x'] = efeat
        graph.apply_edges(concat_message_function)
        return graph.edata['cat_feat']


def concat_efeat(efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features.
    graph : DGLGraph | CuGraphCSC
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph.to_dgl_graph())
            elif graph.is_distributed:
                src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                dst_feat = nfeat
                efeat = update_efeat_bipartite_e2e(efeat, src_feat, dst_feat, bipartite_graph, 'concat')
            else:
                static_graph = graph.to_static_csc()
                efeat = update_efeat_static_e2e(efeat, nfeat, static_graph, mode='concat', use_source_emb=True, use_target_emb=True)
        else:
            efeat = concat_efeat_dgl(efeat, nfeat, graph)
    else:
        src_feat, dst_feat = nfeat
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph.to_dgl_graph())
            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                efeat = update_efeat_bipartite_e2e(efeat, src_feat, dst_feat, bipartite_graph, 'concat')
        else:
            efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph)
    return efeat


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer. It first
    concatenates the input edge features and the node features of the
    corresponding source and destination nodes of the corresponding edge
    to create new edge features. These then are transformed through the
    transformations mentioned above.

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    bias : bool, optional
        whether to use bias in the MLP, by default True
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, efeat_dim: 'int'=512, src_dim: 'int'=512, dst_dim: 'int'=512, output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=2, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', bias: 'bool'=True, recompute_activation: 'bool'=False):
        cat_dim = efeat_dim + src_dim + dst_dim
        super(MeshGraphEdgeMLPConcat, self).__init__(cat_dim, output_dim, hidden_dim, hidden_layers, activation_fn, norm_type, recompute_activation)

    def forward(self, efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        efeat = concat_efeat(efeat, nfeat, graph)
        efeat = self.model(efeat)
        return efeat


@torch.jit.script
def sum_efeat_dgl(efeat: 'Tensor', src_feat: 'Tensor', dst_feat: 'Tensor', src_idx: 'Tensor', dst_idx: 'Tensor') ->Tensor:
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]'):
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features (static setting) or tuple of node features of
        source and destination nodes (bipartite setting).
    graph : DGLGraph | CuGraphCSC
        The underlying graph.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
            elif graph.is_distributed:
                src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                dst_feat = nfeat
                bipartite_graph = graph.to_bipartite_csc()
                sum_efeat = update_efeat_bipartite_e2e(efeat, src_feat, dst_feat, bipartite_graph, mode='sum')
            else:
                static_graph = graph.to_static_csc()
                sum_efeat = update_efeat_bipartite_e2e(efeat, nfeat, static_graph, mode='sum')
        else:
            src_feat, dst_feat = nfeat, nfeat
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
    else:
        src_feat, dst_feat = nfeat
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                bipartite_graph = graph.to_bipartite_csc()
                sum_efeat = update_efeat_bipartite_e2e(efeat, src_feat, dst_feat, bipartite_graph, mode='sum')
        else:
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
    return sum_efeat


class MeshGraphEdgeMLPSum(nn.Module):
    """MLP layer which is commonly used in building blocks
    of models operating on the union of grids and meshes. It
    consists of a number of linear layers followed by an activation
    and a norm layer following the last linear layer. It transform
    edge features - which originally are intended to be a concatenation
    of previous edge features, and the node features of the corresponding
    source and destinationn nodes - by transorming these three features
    individually through separate linear transformations and then sums
    them for each edge accordingly. The result of this is transformed
    through the remaining linear layers and activation or norm functions.

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    bias : bool, optional
        whether to use bias in the MLP, by default True
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, efeat_dim: 'int', src_dim: 'int', dst_dim: 'int', output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', bias: 'bool'=True, recompute_activation: 'bool'=False):
        super().__init__()
        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(orig_weight, [efeat_dim, src_dim, dst_dim], dim=1)
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)
        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None
        layers = [activation_fn]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.norm_type = norm_type
        if norm_type is not None:
            if norm_type not in ['LayerNorm', 'TELayerNorm']:
                raise ValueError(f'Invalid norm type {norm_type}. Supported types are LayerNorm and TELayerNorm.')
            if norm_type == 'TELayerNorm' and te_imported:
                norm_layer = te.LayerNorm
            elif norm_type == 'TELayerNorm' and not te_imported:
                raise ValueError('TELayerNorm requires transformer-engine to be installed.')
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))
        self.model = nn.Sequential(*layers)
        if recompute_activation:
            if not isinstance(activation_fn, nn.SiLU):
                raise ValueError(activation_fn)
            self.recompute_activation = True
        else:
            self.recompute_activation = False

    def forward_truncated_sum(self, efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        """forward pass of the truncated MLP. This uses separate linear layers without
        bias. Bias is added to one MLP, as we sum afterwards. This adds the bias to the
         total sum, too. Having it in one F.linear should allow a fusion of the bias
         addition while avoiding adding the bias to the "edge-level" result.
        """
        if isinstance(nfeat, Tensor):
            src_feat, dst_feat = nfeat, nfeat
        else:
            src_feat, dst_feat = nfeat
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat(mlp_efeat, (mlp_src, mlp_dst), graph)
        return mlp_sum

    def default_forward(self, efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(efeat, nfeat, graph)
        return self.model(mlp_sum)

    def custom_silu_linear_forward(self, efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        """Forward pass of the truncated MLP with custom SiLU function."""
        mlp_sum = self.forward_truncated_sum(efeat, nfeat, graph)
        lin = self.model[1]
        hidden = CustomSiLuLinearAutogradFunction.apply(mlp_sum, lin.weight, lin.bias)
        for i in range(2, self.hidden_layers + 1):
            lin = self.model[2 * i - 1]
            hidden = CustomSiLuLinearAutogradFunction.apply(hidden, lin.weight, lin.bias)
        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers]
            hidden = norm(hidden)
        return hidden

    def forward(self, efeat: 'Tensor', nfeat: 'Union[Tensor, Tuple[Tensor]]', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        if self.recompute_activation:
            return self.custom_silu_linear_forward(efeat, nfeat, graph)
        return self.default_forward(efeat, nfeat, graph)


class MeshEdgeBlock(nn.Module):
    """Edge block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.

    Parameters
    ----------
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, input_dim_nodes: 'int'=512, input_dim_edges: 'int'=512, output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', do_concat_trick: 'bool'=False, recompute_activation: 'bool'=False):
        super().__init__()
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(efeat_dim=input_dim_edges, src_dim=input_dim_nodes, dst_dim=input_dim_nodes, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    @torch.jit.ignore()
    def forward(self, efeat: 'Tensor', nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        efeat_new = self.edge_mlp(efeat, nfeat, graph)
        efeat_new = efeat_new + efeat
        return efeat_new, nfeat


@torch.jit.ignore()
def agg_concat_dgl(efeat: 'Tensor', dst_nfeat: 'Tensor', graph: 'DGLGraph', aggregation: 'str') ->Tensor:
    """Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    with graph.local_scope():
        graph.edata['x'] = efeat
        if aggregation == 'sum':
            graph.update_all(fn.copy_e('x', 'm'), fn.sum('m', 'h_dest'))
        elif aggregation == 'mean':
            graph.update_all(fn.copy_e('x', 'm'), fn.mean('m', 'h_dest'))
        else:
            raise RuntimeError('Not a valid aggregation!')
        cat_feat = torch.cat((graph.dstdata['h_dest'], dst_nfeat), -1)
        return cat_feat


def aggregate_and_concat(efeat: 'Tensor', nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]', aggregation: 'str'):
    """
    Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    if isinstance(graph, CuGraphCSC):
        if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
            cat_feat = agg_concat_dgl(efeat, nfeat, graph.to_dgl_graph(), aggregation)
        else:
            static_graph = graph.to_static_csc()
            cat_feat = agg_concat_e2n(nfeat, efeat, static_graph, aggregation)
    else:
        cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)
    return cat_feat


class MeshGraphDecoder(nn.Module):
    """Decoder used e.g. in GraphCast
       which acts on the bipartite graph connecting a mesh
       (e.g. representing a latent space) to a mostly regular
       grid (e.g. representing the output domain).

    Parameters
    ----------
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    input_dim_src_nodes : int, optional
        Input dimensionality of the source node features, by default 512
    input_dim_dst_nodes : int, optional
        Input dimensionality of the destination node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim_dst_nodes : int, optional
        Output dimensionality of the destination node features, by default 512
    output_dim_edges : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, aggregation: 'str'='sum', input_dim_src_nodes: 'int'=512, input_dim_dst_nodes: 'int'=512, input_dim_edges: 'int'=512, output_dim_dst_nodes: 'int'=512, output_dim_edges: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', do_concat_trick: 'bool'=False, recompute_activation: 'bool'=False):
        super().__init__()
        self.aggregation = aggregation
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(efeat_dim=input_dim_edges, src_dim=input_dim_src_nodes, dst_dim=input_dim_dst_nodes, output_dim=output_dim_edges, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.node_mlp = MeshGraphMLP(input_dim=input_dim_dst_nodes + output_dim_edges, output_dim=output_dim_dst_nodes, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    @torch.jit.ignore()
    def forward(self, m2g_efeat: 'Tensor', grid_nfeat: 'Tensor', mesh_nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
        cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat


class MeshGraphEncoder(nn.Module):
    """Encoder used e.g. in GraphCast
       which acts on the bipartite graph connecting a mostly
       regular grid (e.g. representing the input domain) to a mesh
       (e.g. representing a latent space).

    Parameters
    ----------
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    input_dim_src_nodes : int, optional
        Input dimensionality of the source node features, by default 512
    input_dim_dst_nodes : int, optional
        Input dimensionality of the destination node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim_src_nodes : int, optional
        Output dimensionality of the source node features, by default 512
    output_dim_dst_nodes : int, optional
        Output dimensionality of the destination node features, by default 512
    output_dim_edges : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, aggregation: 'str'='sum', input_dim_src_nodes: 'int'=512, input_dim_dst_nodes: 'int'=512, input_dim_edges: 'int'=512, output_dim_src_nodes: 'int'=512, output_dim_dst_nodes: 'int'=512, output_dim_edges: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'int'=nn.SiLU(), norm_type: 'str'='LayerNorm', do_concat_trick: 'bool'=False, recompute_activation: 'bool'=False):
        super().__init__()
        self.aggregation = aggregation
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(efeat_dim=input_dim_edges, src_dim=input_dim_src_nodes, dst_dim=input_dim_dst_nodes, output_dim=output_dim_edges, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.src_node_mlp = MeshGraphMLP(input_dim=input_dim_src_nodes, output_dim=output_dim_src_nodes, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)
        self.dst_node_mlp = MeshGraphMLP(input_dim=input_dim_dst_nodes + output_dim_edges, output_dim=output_dim_dst_nodes, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    @torch.jit.ignore()
    def forward(self, g2m_efeat: 'Tensor', grid_nfeat: 'Tensor', mesh_nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tuple[Tensor, Tensor]:
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
        cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat


class MeshNodeBlock(nn.Module):
    """Node block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.

    Parameters
    ----------
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, aggregation: 'str'='sum', input_dim_nodes: 'int'=512, input_dim_edges: 'int'=512, output_dim: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', recompute_activation: 'bool'=False):
        super().__init__()
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(input_dim=input_dim_nodes + input_dim_edges, output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type, recompute_activation=recompute_activation)

    @torch.jit.ignore()
    def forward(self, efeat: 'Tensor', nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tuple[Tensor, Tensor]:
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat, nfeat_new


def azimuthal_angle(lon: 'Tensor') ->Tensor:
    """
    Gives the azimuthal angle of a point on the sphere

    Parameters
    ----------
    lon : Tensor
        Tensor of shape (N, ) containing the longitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the azimuthal angle
    """
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def rad2deg(rad):
    """Converts radians to degrees

    Parameters
    ----------
    rad :
        Tensor of shape (N, ) containing the radians

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the degrees
    """
    return rad * 180 / np.pi


def geospatial_rotation(invar: 'Tensor', theta: 'Tensor', axis: 'str', unit: 'str'='rad') ->Tensor:
    """Rotation using right hand rule

    Parameters
    ----------
    invar : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    theta : Tensor
        Tensor of shape (N, ) containing the rotation angle
    axis : str
        Axis of rotation
    unit : str, optional
        Unit of the theta, by default "rad"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing the rotated x, y, z coordinates
    """
    if unit == 'deg':
        invar = rad2deg(invar)
    elif unit == 'rad':
        pass
    else:
        raise ValueError('Not a valid unit')
    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    if axis == 'x':
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == 'y':
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == 'z':
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError('Invalid axis')
    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def polar_angle(lat: 'Tensor') ->Tensor:
    """
    Gives the polar angle of a point on the sphere

    Parameters
    ----------
    lat : Tensor
        Tensor of shape (N, ) containing the latitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the polar angle
    """
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def xyz2latlon(xyz: 'Tensor', radius: 'float'=1, unit: 'str'='deg') ->Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    xyz : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == 'deg':
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == 'rad':
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError('Not a valid unit')


def faces_to_edges(faces: 'np.ndarray') ->Tuple[np.ndarray, np.ndarray]:
    """Transforms polygonal faces to sender and receiver indices.

    It does so by transforming every face into N_i edges. Such if the triangular
    face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

    If all faces have consistent orientation, and the surface represented by the
    faces is closed, then every edge in a polygon with a certain orientation
    is also part of another polygon with the opposite orientation. In this
    situation, the edges returned by the method are always bidirectional.

    Args:
      faces: Integer array of shape [num_faces, 3]. Contains node indices
          adjacent to each face.
    Returns:
      Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

    """
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def get_face_centroids(vertices: 'List[Tuple[float, float, float]]', faces: 'List[List[int]]') ->List[Tuple[float, float, float]]:
    """
    Compute the centroids of triangular faces in a graph.

    Parameters:
    vertices (List[Tuple[float, float, float]]): A list of tuples representing the coordinates of the vertices.
    faces (List[List[int]]): A list of lists, where each inner list contains three indices representing a triangular face.

    Returns:
    List[Tuple[float, float, float]]: A list of tuples representing the centroids of the faces.
    """
    centroids = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        centroid = (v0[0] + v1[0] + v2[0]) / 3, (v0[1] + v1[1] + v2[1]) / 3, (v0[2] + v1[2] + v2[2]) / 3
        centroids.append(centroid)
    return centroids


class _ChildVerticesBuilder(object):
    """Bookkeeping of new child vertices added to an existing set of vertices."""

    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        """Creates a new vertex."""
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(0)
        child_vertex_position /= np.linalg.norm(child_vertex_position)
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        """Returns index for a child vertex, creating it if necessary."""
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        """Returns an array with old vertices."""
        return np.array(self._all_vertices_list)


def deg2rad(deg: 'Tensor') ->Tensor:
    """Converts degrees to radians

    Parameters
    ----------
    deg :
        Tensor of shape (N, ) containing the degrees

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the radians
    """
    return deg * np.pi / 180


def latlon2xyz(latlon: 'Tensor', radius: 'float'=1, unit: 'str'='deg') ->Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    latlon : Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    """
    if unit == 'deg':
        latlon = deg2rad(latlon)
    elif unit == 'rad':
        pass
    else:
        raise ValueError('Not a valid unit')
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def max_edge_length(vertices: 'List[List[float]]', source_nodes: 'List[int]', destination_nodes: 'List[int]') ->float:
    """
    Compute the maximum edge length in a graph.

    Parameters:
    vertices (List[List[float]]): A list of tuples representing the coordinates of the vertices.
    source_nodes (List[int]): A list of indices representing the source nodes of the edges.
    destination_nodes (List[int]): A list of indices representing the destination nodes of the edges.

    Returns:
    The maximum edge length in the graph (float).
    """
    vertices_np = np.array(vertices)
    source_coords = vertices_np[source_nodes]
    dest_coords = vertices_np[destination_nodes]
    squared_differences = np.sum((source_coords - dest_coords) ** 2, axis=1)
    max_length = np.sqrt(np.max(squared_differences))
    return max_length


class Graph:
    """Graph class for creating the graph2mesh, latent mesh, and mesh2graph graphs.

    Parameters
    ----------
    lat_lon_grid : Tensor
        Tensor with shape (lat, lon, 2) that includes the latitudes and longitudes
        meshgrid.
    mesh_level: int, optional
        Level of the latent mesh, by default 6
    multimesh: bool, optional
        If the latent mesh is a multimesh, by default True
        If True, the latent mesh includes the nodes corresponding
        to the specified `mesh_level`and incorporates the edges from
        all mesh levels ranging from level 0 up to and including `mesh_level`.
    khop_neighbors: int, optional
        This option is used to retrieve a list of indices for the k-hop neighbors
        of all mesh nodes. It is applicable when a graph transformer is used as the
        processor. If set to 0, this list is not computed. If a message passing
        processor is used, it is forced to 0. By default 0.
    dtype : torch.dtype, optional
        Data type of the graph, by default torch.float
    """

    def __init__(self, lat_lon_grid: 'Tensor', mesh_level: 'int'=6, multimesh: 'bool'=True, khop_neighbors: 'int'=0, dtype=torch.float) ->None:
        self.khop_neighbors = khop_neighbors
        self.dtype = dtype
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)
        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]
        self.finest_mesh_src, self.finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        self.finest_mesh_vertices = np.array(finest_mesh.vertices)
        if multimesh:
            mesh = merge_meshes(_meshes)
            self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
            self.mesh_vertices = np.array(mesh.vertices)
        else:
            mesh = finest_mesh
            self.mesh_src, self.mesh_dst = self.finest_mesh_src, self.finest_mesh_dst
            self.mesh_vertices = self.finest_mesh_vertices
        self.mesh_faces = mesh.faces

    @staticmethod
    def khop_adj_all_k(g, kmax):
        if not g.is_homogeneous:
            raise NotImplementedError('only homogeneous graph is supported')
        min_degree = g.in_degrees().min()
        with torch.no_grad():
            adj = g.adj_external(transpose=True, scipy_fmt=None)
            adj_k = adj
            adj_all = adj.clone()
            for _ in range(2, kmax + 1):
                adj_k = adj @ adj_k / min_degree
                adj_all += adj_k
        return adj_all.to_dense().bool()

    def create_mesh_graph(self, verbose: 'bool'=True) ->Tensor:
        """Create the multimesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Multimesh graph
        """
        mesh_graph = create_graph(self.mesh_src, self.mesh_dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32)
        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        mesh_graph = add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = add_node_features(mesh_graph, mesh_pos)
        mesh_graph.ndata['lat_lon'] = xyz2latlon(mesh_pos)
        mesh_graph.ndata['x'] = mesh_graph.ndata['x']
        mesh_graph.edata['x'] = mesh_graph.edata['x']
        if self.khop_neighbors > 0:
            khop_adj_bool = self.khop_adj_all_k(g=mesh_graph, kmax=self.khop_neighbors)
            mask = ~khop_adj_bool
        else:
            mask = None
        if verbose:
            None
        return mesh_graph, mask

    def create_g2m_graph(self, verbose: 'bool'=True) ->Tensor:
        """Create the graph2mesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Graph2mesh graph.
        """
        max_edge_len = max_edge_length(self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst)
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(self.mesh_vertices)
        distances, indices = neighbors.kneighbors(cartesian_grid)
        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * max_edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
        g2m_graph = create_heterograph(src, dst, ('grid', 'g2m', 'mesh'), dtype=torch.int32)
        g2m_graph.srcdata['pos'] = cartesian_grid
        g2m_graph.dstdata['pos'] = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        g2m_graph.srcdata['lat_lon'] = self.lat_lon_grid_flat
        g2m_graph.dstdata['lat_lon'] = xyz2latlon(g2m_graph.dstdata['pos'])
        g2m_graph = add_edge_features(g2m_graph, (g2m_graph.srcdata['pos'], g2m_graph.dstdata['pos']))
        g2m_graph.srcdata['pos'] = g2m_graph.srcdata['pos']
        g2m_graph.dstdata['pos'] = g2m_graph.dstdata['pos']
        g2m_graph.ndata['pos']['grid'] = g2m_graph.ndata['pos']['grid']
        g2m_graph.ndata['pos']['mesh'] = g2m_graph.ndata['pos']['mesh']
        g2m_graph.edata['x'] = g2m_graph.edata['x']
        if verbose:
            None
        return g2m_graph

    def create_m2g_graph(self, verbose: 'bool'=True) ->Tensor:
        """Create the mesh2grid graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Mesh2grid graph.
        """
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        face_centroids = get_face_centroids(self.mesh_vertices, self.mesh_faces)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(face_centroids)
        _, indices = neighbors.kneighbors(cartesian_grid)
        indices = indices.flatten()
        src = [p for i in indices for p in self.mesh_faces[i]]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]
        m2g_graph = create_heterograph(src, dst, ('mesh', 'm2g', 'grid'), dtype=torch.int32)
        m2g_graph.srcdata['pos'] = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        m2g_graph.dstdata['pos'] = cartesian_grid
        m2g_graph.srcdata['lat_lon'] = xyz2latlon(m2g_graph.srcdata['pos'])
        m2g_graph.dstdata['lat_lon'] = self.lat_lon_grid_flat
        m2g_graph = add_edge_features(m2g_graph, (m2g_graph.srcdata['pos'], m2g_graph.dstdata['pos']))
        m2g_graph.srcdata['pos'] = m2g_graph.srcdata['pos']
        m2g_graph.dstdata['pos'] = m2g_graph.dstdata['pos']
        m2g_graph.ndata['pos']['grid'] = m2g_graph.ndata['pos']['grid']
        m2g_graph.ndata['pos']['mesh'] = m2g_graph.ndata['pos']['mesh']
        m2g_graph.edata['x'] = m2g_graph.edata['x']
        if verbose:
            None
        return m2g_graph


def checkpoint_identity(layer: 'Callable', *args: Any, **kwargs: Any) ->Any:
    """Applies the identity function for checkpointing.

    This function serves as an identity function for use with model layers
    when checkpointing is not enabled. It simply forwards the input arguments
    to the specified layer and returns its output.

    Parameters
    ----------
    layer : Callable
        The model layer or function to apply to the input arguments.
    *args
        Positional arguments to be passed to the layer.
    **kwargs
        Keyword arguments to be passed to the layer.

    Returns
    -------
    Any
        The output of the specified layer after processing the input arguments.
    """
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: 'bool') ->Callable:
    """Sets checkpoint function.

    This function returns the appropriate checkpoint function based on the
    provided `do_checkpointing` flag. If `do_checkpointing` is True, the
    function returns the checkpoint function from PyTorch's
    `torch.utils.checkpoint`. Otherwise, it returns an identity function
    that simply passes the inputs through the given layer.

    Parameters
    ----------
    do_checkpointing : bool
        Whether to use checkpointing for gradient computation. Checkpointing
        can reduce memory usage during backpropagation at the cost of
        increased computation time.

    Returns
    -------
    Callable
        The selected checkpoint function to use for gradient computation.
    """
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


class GraphCastProcessor(nn.Module):
    """Processor block used in GraphCast operating on a latent space
    represented by hierarchy of icosahedral meshes.

    Parameters
    ----------
    aggregation : str, optional
        message passing aggregation method ("sum", "mean"), by default "sum"
    processor_layers : int, optional
        number of processor layers, by default 16
    input_dim_nodes : int, optional
        input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        input dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(self, aggregation: 'str'='sum', processor_layers: 'int'=16, input_dim_nodes: 'int'=512, input_dim_edges: 'int'=512, hidden_dim: 'int'=512, hidden_layers: 'int'=1, activation_fn: 'nn.Module'=nn.SiLU(), norm_type: 'str'='LayerNorm', do_concat_trick: 'bool'=False, recompute_activation: 'bool'=False):
        super().__init__()
        edge_block_invars = input_dim_nodes, input_dim_edges, input_dim_edges, hidden_dim, hidden_layers, activation_fn, norm_type, do_concat_trick, recompute_activation
        node_block_invars = aggregation, input_dim_nodes, input_dim_edges, input_dim_nodes, hidden_dim, hidden_layers, activation_fn, norm_type, recompute_activation
        layers = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(*edge_block_invars))
            layers.append(MeshNodeBlock(*node_block_invars))
        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: 'int'):
        """
        Set the number of checkpoint segments

        Parameters
        ----------
        checkpoint_segments : int
            number of checkpoint segments

        Raises
        ------
        ValueError
            if the number of processor layers is not a multiple of the number of
            checkpoint segments
        """
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError('Processor layers must be a multiple of checkpoint_segments')
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(self, segment_start: 'int', segment_end: 'int'):
        """Custom forward for gradient checkpointing

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment
        segment_end : int
            Layer index as end of the segment

        Returns
        -------
        function
            Custom forward function
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(efeat, nfeat, graph):
            """Custom forward function"""
            for module in segment:
                efeat, nfeat = module(efeat, nfeat, graph)
            return efeat, nfeat
        return custom_forward

    def forward(self, efeat: 'Tensor', nfeat: 'Tensor', graph: 'Union[DGLGraph, CuGraphCSC]') ->Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(self.run_function(segment_start, segment_end), efeat, nfeat, graph, use_reentrant=False, preserve_rng_state=False)
        return efeat, nfeat


class GraphCastProcessorGraphTransformer(nn.Module):
    """Processor block used in GenCast operating on a latent space
    represented by hierarchy of icosahedral meshes.

    Parameters
    ----------
    attn_mask : torch.Tensor
        Attention mask to be applied within the transformer layers.
    processor_layers : int, optional (default=16)
        Number of processing layers.
    input_dim_nodes : int, optional (default=512)
        Dimension of the input features for each node.
    hidden_dim : int, optional (default=512)
        Dimension of the hidden features within the transformer layers.
    """

    def __init__(self, attention_mask: 'torch.Tensor', num_attention_heads: 'int'=4, processor_layers: 'int'=16, input_dim_nodes: 'int'=512, hidden_dim: 'int'=512):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        self.register_buffer('mask', self.attention_mask, persistent=False)
        layers = [te.pytorch.TransformerLayer(hidden_size=input_dim_nodes, ffn_hidden_size=hidden_dim, num_attention_heads=num_attention_heads, layer_number=i + 1, fuse_qkv_params=False) for i in range(processor_layers)]
        self.processor_layers = nn.ModuleList(layers)

    def forward(self, nfeat: 'Tensor') ->Tensor:
        nfeat = nfeat.unsqueeze(1)
        for module in self.processor_layers:
            nfeat = module(nfeat, attention_mask=self.mask, self_attn_mask_type='arbitrary')
        return torch.squeeze(nfeat, 1)


def get_lat_lon_partition_separators(partition_size: 'int'):
    """Utility Function to get separation intervals for lat-lon
    grid for partition_sizes of interest.

    Parameters
    ----------
    partition_size : int
        size of graph partition
    """

    def _divide(num_lat_chunks: 'int', num_lon_chunks: 'int'):
        if num_lon_chunks * num_lat_chunks != partition_size:
            raise ValueError("Can't divide lat-lon grid into grid {num_lat_chunks} x {num_lon_chunks} chunks for partition_size={partition_size}.")
        lat_bin_width = 180.0 / num_lat_chunks
        lon_bin_width = 360.0 / num_lon_chunks
        lat_ranges = []
        lon_ranges = []
        for p_lat in range(num_lat_chunks):
            for p_lon in range(num_lon_chunks):
                lat_ranges += [(lat_bin_width * p_lat - 90.0, lat_bin_width * (p_lat + 1) - 90.0)]
                lon_ranges += [(lon_bin_width * p_lon - 180.0, lon_bin_width * (p_lon + 1) - 180.0)]
        lat_ranges[-1] = lat_ranges[-1][0], None
        lon_ranges[-1] = lon_ranges[-1][0], None
        return lat_ranges, lon_ranges
    lat_chunks, lon_chunks, i = 1, partition_size, 0
    while lat_chunks < lon_chunks:
        i += 1
        if partition_size % i == 0:
            lat_chunks = i
            lon_chunks = partition_size // lat_chunks
    lat_ranges, lon_ranges = _divide(lat_chunks, lon_chunks)
    if lat_ranges is None or lon_ranges is None:
        raise ValueError('unexpected error, abort')
    min_seps = []
    max_seps = []
    for i in range(partition_size):
        lat = lat_ranges[i]
        lon = lon_ranges[i]
        min_seps.append([lat[0], lon[0]])
        max_seps.append([lat[1], lon[1]])
    return min_seps, max_seps


class CubeEmbedding(nn.Module):
    """
    3D Image Cube Embedding
    Args:
        img_size (tuple[int]): Image size [T, Lat, Lon].
        patch_size (tuple[int]): Patch token size [T, Lat, Lon].
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: 'torch.Tensor'):
        B, C, T, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class DGMLayer(nn.Module):
    """
    Deep Galerkin Model layer.

    Parameters
    ----------
    in_features_1 : int
        Number of input features for first input.
    in_features_2 : int
        Number of input features for second input.
    out_features : int
        Number of output features.
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor]], optional
        Activation function, by default Activation.IDENTITY
    weight_norm : bool, optional
        Apply weight normalization, by default False
    activation_par : Optional[nn.Parameter], optional
        Activation parameter, by default None

    Notes
    -----
    Reference: DGM: A deep learning algorithm for solving partial differential
    equations, https://arxiv.org/pdf/1708.07469.pdf
    """

    def __init__(self, in_features_1: 'int', in_features_2: 'int', out_features: 'int', activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, weight_norm: 'bool'=False, activation_par: 'Optional[nn.Parameter]'=None) ->None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = Identity()
        else:
            self.activation_fn = activation_fn
        self.weight_norm = weight_norm
        self.activation_par = activation_par
        if weight_norm:
            self.linear_1 = WeightNormLinear(in_features_1, out_features, bias=False)
            self.linear_2 = WeightNormLinear(in_features_2, out_features, bias=False)
        else:
            self.linear_1 = nn.Linear(in_features_1, out_features, bias=False)
            self.linear_2 = nn.Linear(in_features_2, out_features, bias=False)
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.bias, 0)
        if self.weight_norm:
            nn.init.constant_(self.linear_1.weight_g, 1.0)
            nn.init.constant_(self.linear_2.weight_g, 1.0)

    def forward(self, input_1: 'Tensor', input_2: 'Tensor') ->Tensor:
        x = self.linear_1(input_1) + self.linear_2(input_2) + self.bias
        if self.activation_fn is not None:
            if self.activation_par is None:
                x = self.activation_fn(x)
            else:
                x = self.activation_fn(self.activation_par * x)
        return x


class FourierLayer(nn.Module):
    """Fourier layer used in the Fourier feature network"""

    def __init__(self, in_features: 'int', frequencies) ->None:
        super().__init__()
        if isinstance(frequencies[0], str):
            if 'gaussian' in frequencies[0]:
                nr_freq = frequencies[2]
                np_f = np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
            else:
                nr_freq = len(frequencies[1])
                np_f = []
                if 'full' in frequencies[0]:
                    np_f_i = np.meshgrid(*[np.array(frequencies[1]) for _ in range(in_features)], indexing='ij')
                    np_f.append(np.reshape(np.stack(np_f_i, axis=-1), (nr_freq ** in_features, in_features)))
                if 'axis' in frequencies[0]:
                    np_f_i = np.zeros((nr_freq, in_features, in_features))
                    for i in range(in_features):
                        np_f_i[:, i, i] = np.reshape(np.array(frequencies[1]), nr_freq)
                    np_f.append(np.reshape(np_f_i, (nr_freq * in_features, in_features)))
                if 'diagonal' in frequencies[0]:
                    np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
                    np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                    np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    np_f.append(np_f_i)
                np_f = np.concatenate(np_f, axis=-2)
        else:
            np_f = frequencies
        frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
        frequencies = frequencies.t().contiguous()
        self.register_buffer('frequencies', frequencies)

    def out_features(self) ->int:
        return int(self.frequencies.size(1) * 2)

    def forward(self, x: 'Tensor') ->Tensor:
        x_hat = torch.matmul(x, self.frequencies)
        x_sin = torch.sin(2.0 * math.pi * x_hat)
        x_cos = torch.cos(2.0 * math.pi * x_hat)
        x_i = torch.cat([x_sin, x_cos], dim=-1)
        return x_i


class FourierFilter(nn.Module):
    """Fourier filter used in the multiplicative filter network"""

    def __init__(self, in_features: 'int', layer_size: 'int', nr_layers: 'int', input_scale: 'float') ->None:
        super().__init__()
        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)
        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)

    def forward(self, x: 'Tensor') ->Tensor:
        frequency = self.weight_scale * self.frequency
        x_i = torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class GaborFilter(nn.Module):
    """Gabor filter used in the multiplicative filter network"""

    def __init__(self, in_features: 'int', layer_size: 'int', nr_layers: 'int', input_scale: 'float', alpha: 'float', beta: 'float') ->None:
        super().__init__()
        self.layer_size = layer_size
        self.alpha = alpha
        self.beta = beta
        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)
        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.mu = nn.Parameter(torch.empty(in_features, layer_size))
        self.gamma = nn.Parameter(torch.empty(layer_size))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.uniform_(self.mu, -1.0, 1.0)
        with torch.no_grad():
            self.gamma.copy_(torch.from_numpy(np.random.gamma(self.alpha, 1.0 / self.beta, self.layer_size)))

    def forward(self, x: 'Tensor') ->Tensor:
        frequency = self.weight_scale * (self.frequency * self.gamma.sqrt())
        x_c = x.unsqueeze(-1)
        x_c = x_c - self.mu
        x_c = torch.square(x_c.norm(p=2, dim=-2))
        x_c = torch.exp(-0.5 * x_c * self.gamma)
        x_i = x_c * torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class ConvFCLayer(nn.Module):
    """Base class for 1x1 Conv layer for image channels

    Parameters
    ----------
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, activation_par: 'Union[nn.Parameter, None]'=None) ->None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = Identity()
        else:
            self.activation_fn = activation_fn
        self.activation_par = activation_par

    def apply_activation(self, x: 'Tensor') ->Tensor:
        """Applied activation / learnable activations

        Parameters
        ----------
        x : Tensor
            Input tensor
        """
        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 1d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, in_features: 'int', out_features: 'int', activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, activation_par: 'Union[nn.Parameter, None]'=None, weight_norm: 'bool'=False) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.reset_parameters()
        if weight_norm:
            raise NotImplementedError('Weight norm not supported for Conv FC layers')

    def reset_parameters(self) ->None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 2d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, activation_par: 'Union[nn.Parameter, None]'=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 3d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation_fn: 'Union[nn.Module, Callable[[Tensor], Tensor], None]'=None, activation_par: 'Union[nn.Parameter, None]'=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class ConvNdKernel1Layer(nn.Module):
    """Channel-wise FC like layer for convolutions of arbitrary dimensions
    CAUTION: if n_dims <= 3, use specific version for that n_dims instead

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    """

    def __init__(self, in_channels: 'int', out_channels: 'int') ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: 'Tensor') ->Tensor:
        dims = list(x.size())
        dims[1] = self.out_channels
        x = self.conv(x.view(dims[0], self.in_channels, -1)).view(dims)
        return x


class ConvNdFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with convolutions of arbitrary dimensions
    CAUTION: if n_dims <= 3, use specific version for that n_dims instead

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation_fn: 'Union[nn.Module, None]'=None, activation_par: 'Union[nn.Parameter, None]'=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ConvNdKernel1Layer(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.apply(self.initialise_parameters)

    def initialise_parameters(self, model):
        """Reset layer weights"""
        if hasattr(model, 'bias'):
            nn.init.constant_(model.bias, 0)
        if hasattr(model, 'weight'):
            nn.init.xavier_uniform_(model.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class UpSample3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: 'torch.Tensor'):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = x[:, :out_pl, pad_top:2 * in_lat - pad_bottom, pad_left:2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        pad_front = pad_back = 0
        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)
        x = self.norm(x)
        x = self.linear(x)
        return x


class SirenLayerType(enum.Enum):
    """
    SiReN layer types.
    """
    FIRST = enum.auto()
    HIDDEN = enum.auto()
    LAST = enum.auto()


class SirenLayer(nn.Module):
    """
    SiReN layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    layer_type : SirenLayerType
        Layer type.
    omega_0 : float
        Omega_0 parameter in SiReN.
    """

    def __init__(self, in_features: 'int', out_features: 'int', layer_type: 'SirenLayerType'=SirenLayerType.HIDDEN, omega_0: 'float'=30.0) ->None:
        super().__init__()
        self.in_features = in_features
        self.layer_type = layer_type
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.apply_activation = layer_type in {SirenLayerType.FIRST, SirenLayerType.HIDDEN}
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset layer parameters."""
        weight_ranges = {SirenLayerType.FIRST: 1.0 / self.in_features, SirenLayerType.HIDDEN: math.sqrt(6.0 / self.in_features) / self.omega_0, SirenLayerType.LAST: math.sqrt(6.0 / self.in_features)}
        weight_range = weight_ranges[self.layer_type]
        nn.init.uniform_(self.linear.weight, -weight_range, weight_range)
        k_sqrt = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.linear.bias, -k_sqrt, k_sqrt)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.linear(x)
        if self.apply_activation:
            x = torch.sin(self.omega_0 * x)
        return x


class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, 2))
        self.reset_parameters()

    def compl_mul1d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights)
        return torch.einsum('bix,iox->box', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        bsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(bsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', modes2: 'int'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.reset_parameters()

    def compl_mul2d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights)
        return torch.einsum('bixy,ioxy->boxy', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)


class SpectralConv3d(nn.Module):
    """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', modes1: 'int', modes2: 'int', modes3: 'int'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.reset_parameters()

    def compl_mul3d(self, input: 'Tensor', weights: 'Tensor') ->Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        cweights = torch.view_as_complex(weights)
        return torch.einsum('bixyz,ioxyz->boxyz', input, cweights)

    def forward(self, x: 'Tensor') ->Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)


class SpectralConv4d(nn.Module):
    """Spectral 4D layer from https://github.com/gegewen/nested-fno/blob/main/FNO4D.py"""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()
        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    def compl_mul4d(self, input, weights):
        return torch.einsum('bixyzt,ioxyzt->boxyzt', input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights4)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights5)
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights6)
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights7)
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers

    Parameters
    ----------:
        decoder_layer: torch.nn.Module
            Layer used for the doceder
        num_layers: int
            Number of sub-decoder-layers in the decoder.
        norm: str
            Layer normalization component.
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f'torch.nn.modules.{self.__class__.__name__}')
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: 'Tensor', tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, tgt_is_causal: 'Optional[bool]'=None) ->Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn."""
        output = tgt
        tgt_is_causal = True
        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation: 'str') ->Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}')


class DecoderOnlyLayer(Module):
    """

    Parameters
    ----------
        d_model: int
            Number of expected features in the input.
        nhead: int
            Number of heads in the multiheadattention models.
        dim_feedforward: int
            Dimension of the feedforward network model, by default 2048.
        dropout: float
            The dropout value, by default 0.1.
        activation: str
            The activation function of the intermediate layer, by default 'relu'.
        layer_norm_eps: float
            The eps value in layer normalization components, by default 1e-5.
        batch_first: Bool
            If ``True``, then the input and output tensors are provided
            as (batch, seq, feature), by default ``False`` (seq, batch, feature).
        norm_first: Bool
            If ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after,
            by default ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, layer_norm_eps: 'float'=1e-05, batch_first: 'bool'=False, norm_first: 'bool'=False, bias: 'bool'=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt: 'Tensor', tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, tgt_is_causal: 'bool'=False) ->Tensor:
        """Pass the inputs (and mask) through the decoder layer."""
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]', is_causal: 'bool'=False) ->Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]', is_causal: 'bool'=False) ->Tensor:
        x = self.multihead_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SwinTransformer(nn.Module):
    """Swin Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size, depth):
        super().__init__()
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, to_2tuple(window_size))
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.layer = SwinTransformerStage(dim=embed_dim, out_dim=embed_dim, input_resolution=input_resolution, depth=depth, downsample=None, num_heads=num_heads, window_size=window_size)

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)
        x = x[:, :, padding_top:pad_lat - padding_bottom, padding_left:pad_lon - padding_right]
        return x


class MeshGraphNet(Module):
    """MeshGraphNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_dim : int
        Number of outputs
    processor_size : int, optional
        Number of message passing blocks, by default 15
    mlp_activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'relu'
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : Union[int, None], optional
        Number of MLP layers for the node feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no node encoder
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no edge encoder
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no decoder
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled)

    Example
    -------
    >>> model = modulus.models.meshgraphnet.MeshGraphNet(
    ...         input_dim_nodes=4,
    ...         input_dim_edges=3,
    ...         output_dim=2,
    ...     )
    >>> graph = dgl.rand_graph(10, 5)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> output = model(node_features, edge_features, graph)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks."
    arXiv preprint arXiv:2010.03409 (2020).
    """

    def __init__(self, input_dim_nodes: 'int', input_dim_edges: 'int', output_dim: 'int', processor_size: 'int'=15, mlp_activation_fn: 'Union[str, List[str]]'='relu', num_layers_node_processor: 'int'=2, num_layers_edge_processor: 'int'=2, hidden_dim_processor: 'int'=128, hidden_dim_node_encoder: 'int'=128, num_layers_node_encoder: 'Union[int, None]'=2, hidden_dim_edge_encoder: 'int'=128, num_layers_edge_encoder: 'Union[int, None]'=2, hidden_dim_node_decoder: 'int'=128, num_layers_node_decoder: 'Union[int, None]'=2, aggregation: 'str'='sum', do_concat_trick: 'bool'=False, num_processor_checkpoint_segments: 'int'=0, recompute_activation: 'bool'=False):
        super().__init__(meta=MetaData())
        activation_fn = get_activation(mlp_activation_fn)
        self.edge_encoder = MeshGraphMLP(input_dim_edges, output_dim=hidden_dim_processor, hidden_dim=hidden_dim_edge_encoder, hidden_layers=num_layers_edge_encoder, activation_fn=activation_fn, norm_type='LayerNorm', recompute_activation=recompute_activation)
        self.node_encoder = MeshGraphMLP(input_dim_nodes, output_dim=hidden_dim_processor, hidden_dim=hidden_dim_node_encoder, hidden_layers=num_layers_node_encoder, activation_fn=activation_fn, norm_type='LayerNorm', recompute_activation=recompute_activation)
        self.node_decoder = MeshGraphMLP(hidden_dim_processor, output_dim=output_dim, hidden_dim=hidden_dim_node_decoder, hidden_layers=num_layers_node_decoder, activation_fn=activation_fn, norm_type=None, recompute_activation=recompute_activation)
        self.processor = MeshGraphNetProcessor(processor_size=processor_size, input_dim_node=hidden_dim_processor, input_dim_edge=hidden_dim_processor, num_layers_node=num_layers_node_processor, num_layers_edge=num_layers_edge_processor, aggregation=aggregation, norm_type='LayerNorm', activation_fn=activation_fn, do_concat_trick=do_concat_trick, num_processor_checkpoint_segments=num_processor_checkpoint_segments)

    def forward(self, node_features: 'Tensor', edge_features: 'Tensor', graph: 'Union[DGLGraph, List[DGLGraph], CuGraphCSC]', **kwargs) ->Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


class Mesh_Reduced(torch.nn.Module):
    """PbGMR-GMUS architecture
    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_decode_dim: int
        Number of decoding outputs (per node)
    output_encode_dim: int, optional
        Number of encoding outputs (per pivotal postion),  by default 3
    processor_size : int, optional
        Number of message passing blocks, by default 15
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : int, optional
        Number of MLP layers for the node feature encoder, by default 2
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : int, optional
        Number of MLP layers for the edge feature encoder, by default 2
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : int, optional
        Number of MLP layers for the node feature decoder, by default 2
    k: int, optional
        Number of nodes considered for per pivotal postion, by default 3
    aggregation: str, optional
        Message aggregation type, by default "mean"
    Note
    ----
    Reference: Han, Xu, et al. "Predicting physics in mesh-reduced space with temporal attention."
    arXiv preprint arXiv:2201.09113 (2022).

    """

    def __init__(self, input_dim_nodes: 'int', input_dim_edges: 'int', output_decode_dim: 'int', output_encode_dim: 'int'=3, processor_size: 'int'=15, num_layers_node_processor: 'int'=2, num_layers_edge_processor: 'int'=2, hidden_dim_processor: 'int'=128, hidden_dim_node_encoder: 'int'=128, num_layers_node_encoder: 'int'=2, hidden_dim_edge_encoder: 'int'=128, num_layers_edge_encoder: 'int'=2, hidden_dim_node_decoder: 'int'=128, num_layers_node_decoder: 'int'=2, k: 'int'=3, aggregation: 'str'='mean'):
        super(Mesh_Reduced, self).__init__()
        self.knn_encoder_already = False
        self.knn_decoder_already = False
        self.encoder_processor = MeshGraphNet(input_dim_nodes, input_dim_edges, output_encode_dim, processor_size, 'relu', num_layers_node_processor, num_layers_edge_processor, hidden_dim_processor, hidden_dim_node_encoder, num_layers_node_encoder, hidden_dim_edge_encoder, num_layers_edge_encoder, hidden_dim_node_decoder, num_layers_node_decoder, aggregation)
        self.decoder_processor = MeshGraphNet(output_encode_dim, input_dim_edges, output_decode_dim, processor_size, 'relu', num_layers_node_processor, num_layers_edge_processor, hidden_dim_processor, hidden_dim_node_encoder, num_layers_node_encoder, hidden_dim_edge_encoder, num_layers_edge_encoder, hidden_dim_node_decoder, num_layers_node_decoder, aggregation)
        self.k = k
        self.PivotalNorm = torch.nn.LayerNorm(output_encode_dim)

    def knn_interpolate(self, x: 'torch.Tensor', pos_x: 'torch.Tensor', pos_y: 'torch.Tensor', batch_x: 'torch.Tensor'=None, batch_y: 'torch.Tensor'=None, k: 'int'=3, num_workers: 'int'=1):
        with torch.no_grad():
            assign_index = torch_cluster.knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        y = torch_scatter.scatter(x[x_idx] * weights, y_idx, 0, dim_size=pos_y.size(0), reduce='sum')
        y = y / torch_scatter.scatter(weights, y_idx, 0, dim_size=pos_y.size(0), reduce='sum')
        return y.float(), x_idx, y_idx, weights

    def encode(self, x, edge_features, graph, position_mesh, position_pivotal):
        x = self.encoder_processor(x, edge_features, graph)
        x = self.PivotalNorm(x)
        nodes_index = torch.arange(graph.batch_size)
        batch_mesh = nodes_index.repeat_interleave(graph.batch_num_nodes())
        position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(torch.tensor([len(position_pivotal)] * graph.batch_size))
        x, _, _, _ = self.knn_interpolate(x=x, pos_x=position_mesh_batch, pos_y=position_pivotal_batch, batch_x=batch_mesh, batch_y=batch_pivotal)
        return x

    def decode(self, x, edge_features, graph, position_mesh, position_pivotal):
        nodes_index = torch.arange(graph.batch_size)
        batch_mesh = nodes_index.repeat_interleave(graph.batch_num_nodes())
        position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(torch.tensor([len(position_pivotal)] * graph.batch_size))
        x, _, _, _ = self.knn_interpolate(x=x, pos_x=position_pivotal_batch, pos_y=position_mesh_batch, batch_x=batch_pivotal, batch_y=batch_mesh)
        x = self.decoder_processor(x, edge_features, graph)
        return x


class Sequence_Model(torch.nn.Module):
    """Decoder-only multi-head attention architecture
    Parameters
    ----------
    input_dim : int
        Number of latent features for the graph (#povital_position x output_decode_dim)
    input_context_dim: int
        Number of physical context features
    dropout_rate: float
        Dropout value for attention decoder, by default 2
    num_layers_decoder: int
        Number of sub-decoder-layers in the attention decoder by default 3
    num_heads: int
        Number of heads in the attention decoder, by default 8
    dim_feedforward_scale: int
        The ration between the dimension of the feedforward network model and input_dim
    num_layers_context_encoder: int
        Number of MLP layers for the physical context feature encoder, by default 2
    num_layers_input_encoder: int
        Number of MLP layers for the input feature encoder, by default 2
    num_layers_output_encoder: int
        Number of MLP layers for the output feature encoder, by default 2
    activation: str
        Activation function of the attention decoder, can be 'relu' or 'gelu', by default 'gelu'
    Note
    ----
    Reference: Han, Xu, et al. "Predicting physics in mesh-reduced space with temporal attention."
    arXiv preprint arXiv:2201.09113 (2022).
    """

    def __init__(self, input_dim: 'int', input_context_dim: 'int', dist, dropout_rate: 'float'=0.0, num_layers_decoder: 'int'=3, num_heads: 'int'=8, dim_feedforward_scale: 'int'=4, num_layers_context_encoder: 'int'=2, num_layers_input_encoder: 'int'=2, num_layers_output_encoder: 'int'=2, activation: 'str'='gelu'):
        super().__init__()
        self.dist = dist
        decoder_layer = DecoderOnlyLayer(input_dim, num_heads, dim_feedforward_scale * input_dim, dropout_rate, activation, layer_norm_eps=1e-05, batch_first=True, norm_first=False, bias=True)
        decoder_norm = LayerNorm(input_dim, eps=1e-05, bias=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers_decoder, decoder_norm)
        self.input_encoder = MeshGraphMLP(input_dim, output_dim=input_dim, hidden_dim=input_dim * 2, hidden_layers=num_layers_input_encoder, activation_fn=nn.ReLU(), norm_type='LayerNorm', recompute_activation=False)
        self.output_encoder = MeshGraphMLP(input_dim, output_dim=input_dim, hidden_dim=input_dim * 2, hidden_layers=num_layers_output_encoder, activation_fn=nn.ReLU(), norm_type=None, recompute_activation=False)
        self.context_encoder = MeshGraphMLP(input_context_dim, output_dim=input_dim, hidden_dim=input_dim * 2, hidden_layers=num_layers_context_encoder, activation_fn=nn.ReLU(), norm_type='LayerNorm', recompute_activation=False)

    def forward(self, x, context=None):
        if context is not None:
            context = self.context_encoder(context)
            x = torch.cat([context, x], dim=1)
        x = self.input_encoder(x)
        tgt_mask = self.generate_square_subsequent_mask(x.size()[1], device=self.dist.device)
        output = self.decoder(x, tgt_mask=tgt_mask)
        output = self.output_encoder(output)
        return output[:, 1:]

    @torch.no_grad()
    def sample(self, z0, step_size, context=None):
        """
        Samples a sequence starting from the initial input `z0` for a given number of steps using
        the model's `forward` method.
        """
        z = z0
        for i in range(step_size):
            prediction = self.forward(z, context)[:, -1].unsqueeze(1)
            z = torch.concat([z, prediction], dim=1)
        return z

    @staticmethod
    def generate_square_subsequent_mask(sz: 'int', device: 'torch.device'=torch.device(torch._C._get_default_device()), dtype: 'torch.dtype'=torch.get_default_dtype()) ->Tensor:
        """Generates a square mask for the sequence. The mask shows which entries should not be used."""
        return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device), diagonal=1)


class ModelRegistry:
    _shared_state = {'_model_registry': None}

    def __new__(cls, *args, **kwargs):
        obj = super(ModelRegistry, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if cls._shared_state['_model_registry'] is None:
            cls._shared_state['_model_registry'] = cls._construct_registry()
        return obj

    @staticmethod
    def _construct_registry() ->dict:
        registry = {}
        entrypoints = entry_points(group='modulus.models')
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    def register(self, model: "'modulus.Module'", name: 'Union[str, None]'=None) ->None:
        """
        Registers a modulus model in the model registry under the provided name. If no name
        is provided, the model's name (from its `__name__` attribute) is used. If the
        name is already in use, raises a ValueError.

        Parameters
        ----------
        model : modulus.Module
            The model to be registered. Can be an instance of any class.
        name : str, optional
            The name to register the model under. If None, the model's name is used.

        Raises
        ------
        ValueError
            If the provided name is already in use in the registry.
        """
        if not issubclass(model, modulus.Module):
            raise ValueError(f'Only subclasses of modulus.Module can be registered. Provided model is of type {type(model)}')
        if name is None:
            name = model.__name__
        if name in self._model_registry:
            raise ValueError(f'Name {name} already in use')
        self._model_registry[name] = model

    def factory(self, name: 'str') ->'modulus.Module':
        """
        Returns a registered model given its name.

        Parameters
        ----------
        name : str
            The name of the registered model.

        Returns
        -------
        model : modulus.Module
            The registered model.

        Raises
        ------
        KeyError
            If no model is registered under the provided name.
        """
        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, (EntryPoint, importlib_metadata.EntryPoint)):
                model = model.load()
            return model
        raise KeyError(f'No model is registered under the name {name}')

    def list_models(self) ->List[str]:
        """
        Returns a list of the names of all models currently registered in the registry.

        Returns
        -------
        List[str]
            A list of the names of all registered models. The order of the names is not
            guaranteed to be consistent.
        """
        return list(self._model_registry.keys())

    def __clear_registry__(self):
        self._model_registry = {}

    def __restore_registry__(self):
        self._model_registry = self._construct_registry()


def _download_ngc_model_file(path: 'str', out_path: 'str', timeout: 'int'=300) ->str:
    """Pulls files from model registry on NGC. Supports private registries when NGC
    API key is set the the `NGC_API_KEY` environment variable. If download file is a zip
    folder it will get unzipped.

    Args:
        path (str): NGC model file path of form:
            `ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>`
            or if no team
            `ngc://models/<org_id/model_id>@<version>/<path/in/repo>`
        out_path (str): Output path to save file / folder as
        timeout (int): Time out of requests, default 5 minutes

    Raises:
        ValueError: Invlaid url

    Returns:
        str: output file / folder path
    """
    suffix = 'ngc://models/'
    pattern = re.compile(f'{suffix}[\\w-]+(/[\\w-]+)?/[\\w-]+@[A-Za-z0-9.]+/[\\w/](.*)')
    if not pattern.match(path):
        raise ValueError('Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>/<path/in/repo>')
    path = path.replace(suffix, '')
    if len(path.split('@')[0].split('/')) == 3:
        org, team, model_version, filename = path.split('/', 3)
        model, version = model_version.split('@', 1)
    else:
        org, model_version, filename = path.split('/', 2)
        model, version = model_version.split('@', 1)
        team = None
    token = ''
    if 'NGC_API_KEY' in os.environ:
        try:
            if os.environ['NGC_API_KEY'].startswith('nvapi-'):
                raise NotImplementedError('New personal keys not supported yet')
            else:
                session = requests.Session()
                session.auth = '$oauthtoken', os.environ['NGC_API_KEY']
                headers = {'Accept': 'application/json'}
                authn_url = f'https://authn.nvidia.com/token?service=ngc&scope=group/ngc:{org}&group/ngc:{org}/{team}'
                r = session.get(authn_url, headers=headers, timeout=5)
                r.raise_for_status()
                token = json.loads(r.content)['token']
        except requests.exceptions.RequestException:
            logger.warning('Failed to get JWT using the API set in NGC_API_KEY environment variable')
            raise
    if len(token) > 0:
        if team:
            file_url = f'https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model}/versions/{version}/files/{filename}'
        else:
            file_url = f'https://api.ngc.nvidia.com/v2/org/{org}/models/{model}/versions/{version}/files/{filename}'
    elif team:
        file_url = f'https://api.ngc.nvidia.com/v2/models/{org}/{team}/{model}/versions/{version}/files/{filename}'
    else:
        file_url = f'https://api.ngc.nvidia.com/v2/models/{org}/{model}/versions/{version}/files/{filename}'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    with requests.get(file_url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        progress_bar.set_description(f'Fetching {filename}')
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    if zipfile.is_zipfile(out_path) and path.endswith('.zip'):
        temp_path = out_path + '.zip'
        os.rename(out_path, temp_path)
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(out_path)
        os.remove(temp_path)
    return out_path


def _get_fs(path):
    if path.startswith('s3://'):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url='https://pbss.s8k.io'))
    else:
        return fsspec.filesystem('file')


class Module(torch.nn.Module):
    """The base class for all network models in Modulus.

    This should be used as a direct replacement for torch.nn.module and provides
    additional functionality for saving and loading models, as well as
    handling file system abstractions.

    There is one important requirement for all models in Modulus. They must
    have json serializable arguments in their __init__ function. This is
    required for saving and loading models and allow models to be instantiated
    from a checkpoint.

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None
    """
    _file_extension = '.mdlus'
    __model_checkpoint_version__ = '0.1.0'

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)
        sig = inspect.signature(cls.__init__)
        bound_args = sig.bind_partial(*([None] + list(args)), **kwargs)
        bound_args.apply_defaults()
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            if k == 'self':
                continue
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v
        out._args = {'__name__': cls.__name__, '__module__': cls.__module__, '__args__': instantiate_args}
        return out

    def __init__(self, meta: 'Union[ModelMetaData, None]'=None):
        super().__init__()
        self.meta = meta
        self.register_buffer('device_buffer', torch.empty(0))
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger('core.module')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _safe_members(tar, local_path):
        for member in tar.getmembers():
            if '..' in member.name or os.path.isabs(member.name) or os.path.realpath(os.path.join(local_path, member.name)).startswith(os.path.realpath(local_path)):
                yield member
            else:
                None

    @classmethod
    def instantiate(cls, arg_dict: 'Dict[str, Any]') ->'Module':
        """Instantiate a model from a dictionary of arguments

        Parameters
        ----------
        arg_dict : Dict[str, Any]
            Dictionary of arguments to instantiate model with. This should be
            have three keys: '__name__', '__module__', and '__args__'. The first two
            are used to import the class and the last is used to instantiate
            the class. The '__args__' key should be a dictionary of arguments
            to pass to the class's __init__ function.

        Returns
        -------
        Module

        Examples
        --------
        >>> from modulus.models import Module
        >>> fcn = Module.instantiate({'__name__': 'FullyConnected', '__module__': 'modulus.models.mlp', '__args__': {'in_features': 10}})
        >>> fcn
        FullyConnected(
          (layers): ModuleList(
            (0): FCLayer(
              (activation_fn): SiLU()
              (linear): Linear(in_features=10, out_features=512, bias=True)
            )
            (1-5): 5 x FCLayer(
              (activation_fn): SiLU()
              (linear): Linear(in_features=512, out_features=512, bias=True)
            )
          )
          (final_layer): FCLayer(
            (activation_fn): Identity()
            (linear): Linear(in_features=512, out_features=512, bias=True)
          )
        )
        """
        _cls_name = arg_dict['__name__']
        registry = ModelRegistry()
        if cls.__name__ == arg_dict['__name__']:
            _cls = cls
        elif _cls_name in registry.list_models():
            _cls = registry.factory(_cls_name)
        else:
            try:
                _mod = importlib.import_module(arg_dict['__module__'])
                _cls = getattr(_mod, arg_dict['__name__'])
            except AttributeError:
                _cls = cls
        return _cls(**arg_dict['__args__'])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def save(self, file_name: 'Union[str, None]'=None, verbose: 'bool'=False) ->None:
        """Simple utility for saving just the model

        Parameters
        ----------
        file_name : Union[str,None], optional
            File name to save model weight to. When none is provide it will default to
            the model's name set in the meta data, by default None
        verbose : bool, optional
            Whether to save the model in verbose mode which will include git hash, etc, by default False

        Raises
        ------
        ValueError
            If file_name does not end with .mdlus extension
        """
        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(f'File name must end with {self._file_extension} extension')
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            torch.save(self.state_dict(), local_path / 'model.pt')
            with open(local_path / 'args.json', 'w') as f:
                json.dump(self._args, f)
            metadata_info = {'modulus_version': modulus.__version__, 'mdlus_file_version': self.__model_checkpoint_version__}
            if verbose:
                try:
                    repo = git.Repo(search_parent_directories=True)
                    metadata_info['git_hash'] = repo.head.object.hexsha
                except git.InvalidGitRepositoryError:
                    metadata_info['git_hash'] = None
            with open(local_path / 'metadata.json', 'w') as f:
                json.dump(metadata_info, f)
            with tarfile.open(local_path / 'model.tar', 'w') as tar:
                for file in local_path.iterdir():
                    tar.add(str(file), arcname=file.name)
            if file_name is None:
                file_name = self.meta.name + '.mdlus'
            fs = _get_fs(file_name)
            fs.put(str(local_path / 'model.tar'), file_name)

    @staticmethod
    def _check_checkpoint(local_path: 'str') ->bool:
        if not local_path.joinpath('args.json').exists():
            raise IOError("File 'args.json' not found in checkpoint")
        if not local_path.joinpath('metadata.json').exists():
            raise IOError("File 'metadata.json' not found in checkpoint")
        if not local_path.joinpath('model.pt').exists():
            raise IOError("Model weights 'model.pt' not found in checkpoint")
        with open(local_path.joinpath('metadata.json'), 'r') as f:
            metadata_info = json.load(f)
            if metadata_info['mdlus_file_version'] != Module.__model_checkpoint_version__:
                raise IOError(f"Model checkpoint version {metadata_info['mdlus_file_version']} is not compatible with current version {Module.__version__}")

    def load(self, file_name: 'str', map_location: 'Union[None, str, torch.device]'=None, strict: 'bool'=True) ->None:
        """Simple utility for loading the model weights from checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name
        map_location : Union[None, str, torch.device], optional
            Map location for loading the model weights, by default None will use model's device
        strict: bool, optional
            whether to strictly enforce that the keys in state_dict match, by default True

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """
        cached_file_name = _download_cached(file_name)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            with tarfile.open(cached_file_name, 'r') as tar:
                tar.extractall(path=local_path, members=list(Module._safe_members(tar, local_path)))
            Module._check_checkpoint(local_path)
            device = map_location if map_location is not None else self.device
            model_dict = torch.load(local_path.joinpath('model.pt'), map_location=device)
            self.load_state_dict(model_dict, strict=strict)

    @classmethod
    def from_checkpoint(cls, file_name: 'str') ->'Module':
        """Simple utility for constructing a model from a checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name

        Returns
        -------
        Module

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """
        cached_file_name = _download_cached(file_name)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            with tarfile.open(cached_file_name, 'r') as tar:
                tar.extractall(path=local_path, members=list(cls._safe_members(tar, local_path)))
            Module._check_checkpoint(local_path)
            with open(local_path.joinpath('args.json'), 'r') as f:
                args = json.load(f)
            model = cls.instantiate(args)
            model_dict = torch.load(local_path.joinpath('model.pt'), map_location=model.device)
            model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_torch(torch_model_class: 'torch.nn.Module', meta: 'ModelMetaData'=None) ->'Module':
        """Construct a Modulus module from a PyTorch module

        Parameters
        ----------
        torch_model_class : torch.nn.Module
            PyTorch module class
        meta : ModelMetaData, optional
            Meta data for the model, by default None

        Returns
        -------
        Module
        """


        class ModulusModel(Module):

            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)
        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(zip(model_argnames[-len(model_defaults):], model_defaults))
        params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [inspect.Parameter(argname, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=defaults_dict.get(argname, inspect.Parameter.empty)) for argname in model_argnames]
        init_signature = inspect.Signature(params)
        ModulusModel.__init__.__signature__ = init_signature
        new_class_name = f'{torch_model_class.__name__}ModulusModel'
        ModulusModel.__name__ = new_class_name
        registry = ModelRegistry()
        registry.register(ModulusModel, new_class_name)
        return ModulusModel

    @property
    def device(self) ->torch.device:
        """Get device model is on

        Returns
        -------
        torch.device
            PyTorch device
        """
        return self.device_buffer.device

    def num_parameters(self) ->int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count


class PatchEmbed3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
        l_remainder = level % l_patch_size
        h_remainder = height % l_patch_size
        w_remainder = width % w_patch_size
        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left
        self.pad = nn.ZeroPad3d((padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back))
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: 'torch.Tensor'):
        B, C, L, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


class PatchRecovery3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: 'torch.Tensor'):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape
        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left
        return output[:, :, padding_front:Pl - padding_back, padding_top:Lat - padding_bottom, padding_left:Lon - padding_right]


class Pangu(Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(self, img_size=(721, 1440), patch_size=(2, 4, 4), embed_dim=192, num_heads=(6, 12, 12, 6), window_size=(2, 6, 12)):
        super().__init__(meta=MetaData())
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.patchembed2d = PatchEmbed2D(img_size=img_size, patch_size=patch_size[1:], in_chans=4 + 3, embed_dim=embed_dim)
        self.patchembed3d = PatchEmbed3D(img_size=(13, img_size[0], img_size[1]), patch_size=patch_size, in_chans=5, embed_dim=embed_dim)
        patched_inp_shape = 8, math.ceil(img_size[0] / patch_size[1]), math.ceil(img_size[1] / patch_size[2])
        self.layer1 = FuserLayer(dim=embed_dim, input_resolution=patched_inp_shape, depth=2, num_heads=num_heads[0], window_size=window_size, drop_path=drop_path[:2])
        patched_inp_shape_downsample = 8, math.ceil(patched_inp_shape[1] / 2), math.ceil(patched_inp_shape[2] / 2)
        self.downsample = DownSample3D(in_dim=embed_dim, input_resolution=patched_inp_shape, output_resolution=patched_inp_shape_downsample)
        self.layer2 = FuserLayer(dim=embed_dim * 2, input_resolution=patched_inp_shape_downsample, depth=6, num_heads=num_heads[1], window_size=window_size, drop_path=drop_path[2:])
        self.layer3 = FuserLayer(dim=embed_dim * 2, input_resolution=patched_inp_shape_downsample, depth=6, num_heads=num_heads[2], window_size=window_size, drop_path=drop_path[2:])
        self.upsample = UpSample3D(embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape)
        self.layer4 = FuserLayer(dim=embed_dim, input_resolution=patched_inp_shape, depth=2, num_heads=num_heads[3], window_size=window_size, drop_path=drop_path[:2])
        self.patchrecovery2d = PatchRecovery2D(img_size, patch_size[1:], 2 * embed_dim, 4)
        self.patchrecovery3d = PatchRecovery3D((13, img_size[0], img_size[1]), patch_size, 2 * embed_dim, 5)

    def prepare_input(self, surface, surface_mask, upper_air):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        upper_air = upper_air.reshape(upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4])
        surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        return torch.concat([surface, surface_mask, upper_air], dim=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[:, :7, :, :]
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)
        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = self.layer1(x)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]
        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air


class ResnetBlock(nn.Module):
    """A simple ResNet block

    Parameters
    ----------
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    channels : int
        Number of feature channels
    padding_type : str, optional
        Padding type ('reflect', 'replicate' or 'zero'), by default "reflect"
    activation : nn.Module, optional
        Activation function, by default nn.ReLU()
    use_batch_norm : bool, optional
        Batch normalization, by default False
    """

    def __init__(self, dimension: 'int', channels: 'int', padding_type: 'str'='reflect', activation: 'nn.Module'=nn.ReLU(), use_batch_norm: 'bool'=False, use_dropout: 'bool'=False):
        super().__init__()
        if padding_type not in ['reflect', 'zero', 'replicate']:
            raise ValueError(f'Invalid padding type {padding_type}')
        if dimension == 1:
            conv = nn.Conv1d
            if padding_type == 'reflect':
                padding = nn.ReflectionPad1d(1)
            elif padding_type == 'replicate':
                padding = nn.ReplicationPad1d(1)
            else:
                padding = None
            norm = nn.BatchNorm1d
        elif dimension == 2:
            conv = nn.Conv2d
            if padding_type == 'reflect':
                padding = nn.ReflectionPad2d(1)
            elif padding_type == 'replicate':
                padding = nn.ReplicationPad2d(1)
            else:
                padding = None
            norm = nn.BatchNorm2d
        elif dimension == 3:
            conv = nn.Conv3d
            if padding_type == 'reflect':
                padding = nn.ReflectionPad3d(1)
            elif padding_type == 'replicate':
                padding = nn.ReplicationPad3d(1)
            else:
                padding = None
            norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(f'Pix2Pix ResnetBlock only supported dimensions 1, 2, 3. Got {dimension}')
        conv_block = []
        if padding_type != 'zero':
            conv_block += [padding]
            p = 0
        else:
            p = 1
        conv_block.append(conv(channels, channels, kernel_size=3, padding=p))
        if use_batch_norm:
            conv_block.append(norm(channels))
        conv_block.append(activation)
        if padding_type != 'zero':
            conv_block += [padding]
        conv_block += [conv(channels, channels, kernel_size=3, padding=p)]
        if use_batch_norm:
            conv_block.append(norm(channels))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: 'Tensor') ->Tensor:
        out = x + self.conv_block(x)
        return out


class Pix2Pix(Module):
    """Convolutional encoder-decoder based on pix2pix generator models.

    Note
    ----
    The pix2pix architecture supports options for 1D, 2D and 3D fields which can
    be constroled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels: Union[int, Any], optional
        Number of output channels
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    conv_layer_size : int, optional
        Latent channel size after first convolution, by default 64
    n_downsampling : int, optional
        Number of downsampling blocks, by default 3
    n_upsampling : int, optional
        Number of upsampling blocks, by default 3
    n_blocks : int, optional
        Number of residual blocks in middle of model, by default 3
    activation_fn : Any, optional
        Activation function, by default "relu"
    batch_norm : bool, optional
        Batch normalization, by default False
    padding_type : str, optional
        Padding type ('reflect', 'replicate' or 'zero'), by default "reflect"

    Example
    -------
    >>> #2D convolutional encoder decoder
    >>> model = modulus.models.pix2pix.Pix2Pix(
    ... in_channels=1,
    ... out_channels=2,
    ... dimension=2,
    ... conv_layer_size=4)
    >>> input = torch.randn(4, 1, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 2, 32, 32])

    Note
    ----
    Reference:  Isola, Phillip, et al. “Image-To-Image translation with conditional
    adversarial networks” Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. “High-Resolution image synthesis and semantic
    manipulation with conditional GANs” Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/NVIDIA/pix2pixHD
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', dimension: 'int', conv_layer_size: 'int'=64, n_downsampling: 'int'=3, n_upsampling: 'int'=3, n_blocks: 'int'=3, activation_fn: 'str'='relu', batch_norm: 'bool'=False, padding_type: 'str'='reflect'):
        if not (n_blocks >= 0 and n_downsampling >= 0 and n_upsampling >= 0):
            raise ValueError('Invalid arch params')
        if padding_type not in ['reflect', 'zero', 'replicate']:
            raise ValueError('Invalid padding type')
        super().__init__(meta=MetaData())
        if isinstance(activation_fn, str):
            activation = get_activation(activation_fn)
        else:
            activation = activation_fn
        if dimension == 1:
            padding = nn.ReflectionPad1d(3)
            conv = nn.Conv1d
            trans_conv = nn.ConvTranspose1d
            norm = nn.BatchNorm1d
        elif dimension == 2:
            padding = nn.ReflectionPad2d(3)
            conv = nn.Conv2d
            trans_conv = nn.ConvTranspose2d
            norm = nn.BatchNorm2d
        elif dimension == 3:
            padding = nn.ReflectionPad3d(3)
            conv = nn.Conv3d
            trans_conv = nn.ConvTranspose3d
            norm = nn.BatchNorm3d
        else:
            raise NotImplementedError(f'Pix2Pix only supported dimensions 1, 2, 3. Got {dimension}')
        model = [padding, conv(in_channels, conv_layer_size, kernel_size=7, padding=0)]
        if batch_norm:
            model.append(norm(conv_layer_size))
        model.append(activation)
        for i in range(n_downsampling):
            mult = 2 ** i
            model.append(conv(conv_layer_size * mult, conv_layer_size * mult * 2, kernel_size=3, stride=2, padding=1))
            if batch_norm:
                model.append(norm(conv_layer_size * mult * 2))
            model.append(activation)
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(dimension, conv_layer_size * mult, padding_type=padding_type, activation=activation, use_batch_norm=batch_norm)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.append(trans_conv(int(conv_layer_size * mult), int(conv_layer_size * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1))
            if batch_norm:
                model.append(norm(int(conv_layer_size * mult / 2)))
            model.append(activation)
        for i in range(max([0, n_upsampling - n_downsampling])):
            model.append(trans_conv(int(conv_layer_size), int(conv_layer_size), kernel_size=3, stride=2, padding=1, output_padding=1))
            if batch_norm:
                model.append(norm(conv_layer_size))
            model.append(activation)
        model += [padding, conv(conv_layer_size, out_channels, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input: 'Tensor') ->Tensor:
        y = self.model(input)
        return y


class _ConvLayer(nn.Module):
    """Generalized Convolution Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input, 1, 2, 3, or 4
    kernel_size : int
        Kernel size for the convolution
    stride : int
        Stride for the convolution, by default 1
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.Identity()
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', dimension: 'int', kernel_size: 'int', stride: 'int'=1, activation_fn: 'nn.Module'=nn.Identity()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn
        if self.dimension == 1:
            self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        elif self.dimension == 2:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        elif self.dimension == 3:
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        else:
            raise ValueError('Only 1D, 2D and 3D dimensions are supported')
        self.reset_parameters()

    def exec_activation_fn(self, x: 'Tensor') ->Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def reset_parameters(self) ->None:
        """Initialization for network parameters"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        input_length = len(x.size()) - 2
        if input_length != self.dimension:
            raise ValueError('Input dimension not compatible')
        if input_length == 1:
            iw = x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2], mode='constant', value=0.0)
        elif input_length == 2:
            ih, iw = x.size()[-2:]
            pad_h, pad_w = _get_same_padding(ih, self.kernel_size, self.stride), _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(x, [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2], mode='constant', value=0.0)
        else:
            _id, ih, iw = x.size()[-3:]
            pad_d, pad_h, pad_w = _get_same_padding(_id, self.kernel_size, self.stride), _get_same_padding(ih, self.kernel_size, self.stride), _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(x, [pad_d // 2, pad_d - pad_d // 2, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2], mode='constant', value=0.0)
        x = self.conv(x)
        if self.activation_fn is not nn.Identity():
            x = self.exec_activation_fn(x)
        return x


class _TransposeConvLayer(nn.Module):
    """Generalized Transposed Convolution Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input, 1, 2, 3, or 4
    kernel_size : int
        Kernel size for the convolution
    stride : int
        Stride for the convolution, by default 1
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.Identity()
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', dimension: 'int', kernel_size: 'int', stride: 'int'=1, activation_fn=nn.Identity()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn
        if dimension == 1:
            self.trans_conv = nn.ConvTranspose1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        elif dimension == 2:
            self.trans_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        elif dimension == 3:
            self.trans_conv = nn.ConvTranspose3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=True)
        else:
            raise ValueError('Only 1D, 2D and 3D dimensions are supported')
        self.reset_parameters()

    def exec_activation_fn(self, x: 'Tensor') ->Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def reset_parameters(self) ->None:
        """Initialization for network parameters"""
        nn.init.constant_(self.trans_conv.bias, 0)
        nn.init.xavier_uniform_(self.trans_conv.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        orig_x = x
        input_length = len(orig_x.size()) - 2
        if input_length != self.dimension:
            raise ValueError('Input dimension not compatible')
        x = self.trans_conv(x)
        if input_length == 1:
            iw = orig_x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[:, :, pad_w // 2:x.size(-1) - (pad_w - pad_w // 2)]
        elif input_length == 2:
            ih, iw = orig_x.size()[-2:]
            pad_h, pad_w = _get_same_padding(ih, self.kernel_size, self.stride), _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[:, :, pad_h // 2:x.size(-2) - (pad_h - pad_h // 2), pad_w // 2:x.size(-1) - (pad_w - pad_w // 2)]
        else:
            _id, ih, iw = orig_x.size()[-3:]
            pad_d, pad_h, pad_w = _get_same_padding(_id, self.kernel_size, self.stride), _get_same_padding(ih, self.kernel_size, self.stride), _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[:, :, pad_d // 2:x.size(-3) - (pad_d - pad_d // 2), pad_h // 2:x.size(-2) - (pad_h - pad_h // 2), pad_w // 2:x.size(-1) - (pad_w - pad_w // 2)]
        if self.activation_fn is not nn.Identity():
            x = self.exec_activation_fn(x)
        return x


class _ConvGRULayer(nn.Module):
    """Convolutional GRU layer

    Parameters
    ----------
    in_features : int
        Input features/channels
    hidden_size : int
        Hidden layer features/channels
    dimension : int
        Spatial dimension of the input
    activation_fn : nn.Module, optional
        Activation Function to use, by default nn.ReLU()
    """

    def __init__(self, in_features: 'int', hidden_size: 'int', dimension: 'int', activation_fn: 'nn.Module'=nn.ReLU()) ->None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.conv_1 = _ConvLayer(in_channels=in_features + hidden_size, out_channels=2 * hidden_size, kernel_size=3, stride=1, dimension=dimension)
        self.conv_2 = _ConvLayer(in_channels=in_features + hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, dimension=dimension)

    def exec_activation_fn(self, x: 'Tensor') ->Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def forward(self, x: 'Tensor', hidden: 'Tensor') ->Tensor:
        concat = torch.cat((x, hidden), dim=1)
        conv_concat = self.conv_1(concat)
        conv_r, conv_z = torch.split(conv_concat, self.hidden_size, 1)
        reset_gate = torch.special.expit(conv_r)
        update_gate = torch.special.expit(conv_z)
        concat = torch.cat((x, torch.mul(hidden, reset_gate)), dim=1)
        n = self.exec_activation_fn(self.conv_2(concat))
        h_next = torch.mul(1 - update_gate, n) + torch.mul(update_gate, hidden)
        return h_next


class _ConvResidualBlock(nn.Module):
    """Convolutional ResNet Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input
    stride : int
        Stride of the convolutions, by default 1
    gated : bool, optional
        Residual Gate, by default False
    layer_normalization : bool, optional
        Layer Normalization, by default False
    begin_activation_fn : bool, optional
        Whether to use activation function in the beginning, by default True
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.ReLU()

    Raises
    ------
    ValueError
        Stride not supported
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', dimension: 'int', stride: 'int'=1, gated: 'bool'=False, layer_normalization: 'bool'=False, begin_activation_fn: 'bool'=True, activation_fn: 'nn.Module'=nn.ReLU()) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dimension = dimension
        self.gated = gated
        self.layer_normalization = layer_normalization
        self.begin_activation_fn = begin_activation_fn
        self.activation_fn = activation_fn
        if self.stride == 1:
            self.conv_1 = _ConvLayer(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride, dimension=self.dimension)
        elif self.stride == 2:
            self.conv_1 = _ConvLayer(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=4, stride=self.stride, dimension=self.dimension)
        else:
            raise ValueError('stride > 2 is not supported')
        if not self.gated:
            self.conv_2 = _ConvLayer(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, dimension=self.dimension)
        else:
            self.conv_2 = _ConvLayer(in_channels=self.out_channels, out_channels=2 * self.out_channels, kernel_size=3, stride=1, dimension=self.dimension)

    def exec_activation_fn(self, x: 'Tensor') ->Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def forward(self, x: 'Tensor') ->Tensor:
        orig_x = x
        if self.begin_activation_fn:
            if self.layer_normalization:
                layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
                x = layer_norm(x)
            x = self.exec_activation_fn(x)
        x = self.conv_1(x)
        if self.layer_normalization:
            layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
            x = layer_norm(x)
        x = self.exec_activation_fn(x)
        x = self.conv_2(x)
        if self.gated:
            x_1, x_2 = torch.split(x, x.size(1) // 2, 1)
            x = x_1 * torch.special.expit(x_2)
        if orig_x.size(-1) > x.size(-1):
            if len(orig_x.size()) - 2 == 1:
                iw = orig_x.size()[-1:][0]
                pad_w = _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool1d(2, 2, padding=pad_w // 2, count_include_pad=False)
            elif len(orig_x.size()) - 2 == 2:
                ih, iw = orig_x.size()[-2:]
                pad_h, pad_w = _get_same_padding(ih, 2, 2), _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool2d(2, 2, padding=(pad_h // 2, pad_w // 2), count_include_pad=False)
            elif len(orig_x.size()) - 2 == 3:
                _id, ih, iw = orig_x.size()[-3:]
                pad_d, pad_h, pad_w = _get_same_padding(_id, 2, 2), _get_same_padding(ih, 2, 2), _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool3d(2, 2, padding=(pad_d // 2, pad_h // 2, pad_w // 2), count_include_pad=False)
            else:
                raise ValueError('Only 1D, 2D and 3D dimensions are supported')
            orig_x = pool(orig_x)
        in_channels = int(orig_x.size(1))
        if self.out_channels > in_channels:
            orig_x = F.pad(orig_x, (len(orig_x.size()) - 2) * (0, 0) + (self.out_channels - self.in_channels, 0))
        elif self.out_channels < in_channels:
            pass
        return orig_x + x


class One2ManyRNN(Module):
    """A RNN model with encoder/decoder for 2d/3d problems that provides predictions
    based on single initial condition.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input
    dimension : int, optional
        Spatial dimension of the input. Only 2d and 3d are supported, by default 2
    nr_latent_channels : int, optional
        Channels for encoding/decoding, by default 512
    nr_residual_blocks : int, optional
        Number of residual blocks, by default 2
    activation_fn : str, optional
        Activation function to use, by default "relu"
    nr_downsamples : int, optional
        Number of downsamples, by default 2
    nr_tsteps : int, optional
        Time steps to predict, by default 32

    Example
    -------
    >>> model = modulus.models.rnn.One2ManyRNN(
    ... input_channels=6,
    ... dimension=2,
    ... nr_latent_channels=32,
    ... activation_fn="relu",
    ... nr_downsamples=2,
    ... nr_tsteps=16,
    ... )
    >>> input = invar = torch.randn(4, 6, 1, 16, 16) # [N, C, T, H, W]
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 6, 16, 16, 16])
    """

    def __init__(self, input_channels: 'int', dimension: 'int'=2, nr_latent_channels: 'int'=512, nr_residual_blocks: 'int'=2, activation_fn: 'str'='relu', nr_downsamples: 'int'=2, nr_tsteps: 'int'=32) ->None:
        super().__init__(meta=MetaData())
        self.nr_tsteps = nr_tsteps
        self.nr_residual_blocks = nr_residual_blocks
        self.nr_downsamples = nr_downsamples
        self.encoder_layers = nn.ModuleList()
        channels_out = nr_latent_channels
        activation_fn = get_activation(activation_fn)
        if dimension not in [2, 3]:
            raise ValueError('Only 2D and 3D spatial dimensions are supported')
        for i in range(nr_downsamples):
            for j in range(nr_residual_blocks):
                stride = 1
                if i == 0 and j == 0:
                    channels_in = input_channels
                else:
                    channels_in = channels_out
                if j == nr_residual_blocks - 1 and i < nr_downsamples - 1:
                    channels_out = channels_out * 2
                    stride = 2
                self.encoder_layers.append(_ConvResidualBlock(in_channels=channels_in, out_channels=channels_out, stride=stride, dimension=dimension, gated=True, layer_normalization=False, begin_activation_fn=not (i == 0 and j == 0), activation_fn=activation_fn))
        self.rnn_layer = _ConvGRULayer(in_features=channels_out, hidden_size=channels_out, dimension=dimension)
        self.conv_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(nr_downsamples):
            self.upsampling_layers = nn.ModuleList()
            channels_in = channels_out
            channels_out = channels_out // 2
            self.upsampling_layers.append(_TransposeConvLayer(in_channels=channels_in, out_channels=channels_out, kernel_size=4, stride=2, dimension=dimension))
            for j in range(nr_residual_blocks):
                self.upsampling_layers.append(_ConvResidualBlock(in_channels=channels_out, out_channels=channels_out, stride=1, dimension=dimension, gated=True, layer_normalization=False, begin_activation_fn=not (i == 0 and j == 0), activation_fn=activation_fn))
            self.conv_layers.append(_ConvLayer(in_channels=channels_in, out_channels=nr_latent_channels, kernel_size=1, stride=1, dimension=dimension))
            self.decoder_layers.append(self.upsampling_layers)
        if dimension == 2:
            self.final_conv = nn.Conv2d(nr_latent_channels, input_channels, (1, 1), (1, 1), padding='valid')
        else:
            self.final_conv = nn.Conv3d(nr_latent_channels, input_channels, (1, 1, 1), (1, 1, 1), padding='valid')

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward pass

        Parameters
        ----------
        x : Tensor
            Expects a tensor of size [N, C, 1, H, W] for 2D or [N, C, 1, D, H, W] for 3D
            Where, N is the batch size, C is the number of channels, 1 is the number of
            input timesteps and D, H, W are spatial dimensions.
        Returns
        -------
        Tensor
            Size [N, C, T, H, W] for 2D or [N, C, T, D, H, W] for 3D.
            Where, T is the number of timesteps being predicted.
        """
        encoded_inputs = []
        for t in range(1):
            x_in = x[:, :, t, ...]
            for layer in self.encoder_layers:
                x_in = layer(x_in)
            encoded_inputs.append(x_in)
        rnn_output = []
        for t in range(self.nr_tsteps):
            if t == 0:
                h = torch.zeros(list(x_in.size()))
                x_in_rnn = encoded_inputs[0]
            h = self.rnn_layer(x_in_rnn, h)
            x_in_rnn = h
            rnn_output.append(h)
        decoded_output = []
        for t in range(self.nr_tsteps):
            x_out = rnn_output[t]
            latent_context_grid = []
            for conv_layer, decoder in zip(self.conv_layers, self.decoder_layers):
                latent_context_grid.append(conv_layer(x_out))
                upsampling_layers = decoder
                for upsampling_layer in upsampling_layers:
                    x_out = upsampling_layer(x_out)
            out = self.final_conv(latent_context_grid[-1])
            decoded_output.append(out)
        decoded_output = torch.stack(decoded_output, dim=2)
        return decoded_output


class Seq2SeqRNN(Module):
    """A RNN model with encoder/decoder for 2d/3d problems. Given input 0 to t-1,
    predicts signal t to t + nr_tsteps

    Parameters
    ----------
    input_channels : int
        Number of channels in the input
    dimension : int, optional
        Spatial dimension of the input. Only 2d and 3d are supported, by default 2
    nr_latent_channels : int, optional
        Channels for encoding/decoding, by default 512
    nr_residual_blocks : int, optional
        Number of residual blocks, by default 2
    activation_fn : str, optional
        Activation function to use, by default "relu"
    nr_downsamples : int, optional
        Number of downsamples, by default 2
    nr_tsteps : int, optional
        Time steps to predict, by default 32

    Example
    -------
    >>> model = modulus.models.rnn.Seq2SeqRNN(
    ... input_channels=6,
    ... dimension=2,
    ... nr_latent_channels=32,
    ... activation_fn="relu",
    ... nr_downsamples=2,
    ... nr_tsteps=16,
    ... )
    >>> input = invar = torch.randn(4, 6, 16, 16, 16) # [N, C, T, H, W]
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 6, 16, 16, 16])
    """

    def __init__(self, input_channels: 'int', dimension: 'int'=2, nr_latent_channels: 'int'=512, nr_residual_blocks: 'int'=2, activation_fn: 'str'='relu', nr_downsamples: 'int'=2, nr_tsteps: 'int'=32) ->None:
        super().__init__(meta=MetaData())
        self.nr_tsteps = nr_tsteps
        self.nr_residual_blocks = nr_residual_blocks
        self.nr_downsamples = nr_downsamples
        self.encoder_layers = nn.ModuleList()
        channels_out = nr_latent_channels
        activation_fn = get_activation(activation_fn)
        if dimension not in [2, 3]:
            raise ValueError('Only 2D and 3D spatial dimensions are supported')
        for i in range(nr_downsamples):
            for j in range(nr_residual_blocks):
                stride = 1
                if i == 0 and j == 0:
                    channels_in = input_channels
                else:
                    channels_in = channels_out
                if j == nr_residual_blocks - 1 and i < nr_downsamples - 1:
                    channels_out = channels_out * 2
                    stride = 2
                self.encoder_layers.append(_ConvResidualBlock(in_channels=channels_in, out_channels=channels_out, stride=stride, dimension=dimension, gated=True, layer_normalization=False, begin_activation_fn=not (i == 0 and j == 0), activation_fn=activation_fn))
        self.rnn_layer = _ConvGRULayer(in_features=channels_out, hidden_size=channels_out, dimension=dimension)
        self.conv_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(nr_downsamples):
            self.upsampling_layers = nn.ModuleList()
            channels_in = channels_out
            channels_out = channels_out // 2
            self.upsampling_layers.append(_TransposeConvLayer(in_channels=channels_in, out_channels=channels_out, kernel_size=4, stride=2, dimension=dimension))
            for j in range(nr_residual_blocks):
                self.upsampling_layers.append(_ConvResidualBlock(in_channels=channels_out, out_channels=channels_out, stride=1, dimension=dimension, gated=True, layer_normalization=False, begin_activation_fn=not (i == 0 and j == 0), activation_fn=activation_fn))
            self.conv_layers.append(_ConvLayer(in_channels=channels_in, out_channels=nr_latent_channels, kernel_size=1, stride=1, dimension=dimension))
            self.decoder_layers.append(self.upsampling_layers)
        if dimension == 2:
            self.final_conv = nn.Conv2d(nr_latent_channels, input_channels, (1, 1), (1, 1), padding='valid')
        else:
            self.final_conv = nn.Conv3d(nr_latent_channels, input_channels, (1, 1, 1), (1, 1, 1), padding='valid')

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward pass

        Parameters
        ----------
        x : Tensor
            Expects a tensor of size [N, C, T, H, W] for 2D or [N, C, T, D, H, W] for 3D
            Where, N is the batch size, C is the number of channels, T is the number of
            input timesteps and D, H, W are spatial dimensions. Currently, this
            requires input time steps to be same as predicted time steps.
        Returns
        -------
        Tensor
            Size [N, C, T, H, W] for 2D or [N, C, T, D, H, W] for 3D.
            Where, T is the number of timesteps being predicted.
        """
        encoded_inputs = []
        for t in range(self.nr_tsteps):
            x_in = x[:, :, t, ...]
            for layer in self.encoder_layers:
                x_in = layer(x_in)
            encoded_inputs.append(x_in)
        for t in range(x.size(2)):
            if t == 0:
                h = torch.zeros(list(x_in.size()))
            x_in_rnn = encoded_inputs[t]
            h = self.rnn_layer(x_in_rnn, h)
        rnn_output = []
        for t in range(self.nr_tsteps):
            if t == 0:
                x_in_rnn = encoded_inputs[-1]
            h = self.rnn_layer(x_in_rnn, h)
            x_in_rnn = h
            rnn_output.append(h)
        decoded_output = []
        for t in range(self.nr_tsteps):
            x_out = rnn_output[t]
            latent_context_grid = []
            for conv_layer, decoder in zip(self.conv_layers, self.decoder_layers):
                latent_context_grid.append(conv_layer(x_out))
                upsampling_layers = decoder
                for upsampling_layer in upsampling_layers:
                    x_out = upsampling_layer(x_out)
            out = self.final_conv(latent_context_grid[-1])
            decoded_output.append(out)
        decoded_output = torch.stack(decoded_output, dim=2)
        return decoded_output


class ConvolutionalBlock3d(nn.Module):
    """3D convolutional block

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    kernel_size : int
        Kernel size
    stride : int, optional
        Convolutional stride, by default 1
    batch_norm : bool, optional
        Use batchnorm, by default False
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, batch_norm: 'bool'=False, activation_fn: 'nn.Module'=nn.Identity()):
        super().__init__()
        layers = list()
        layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
        if batch_norm is True:
            layers.append(nn.BatchNorm3d(num_features=out_channels))
        self.activation_fn = activation_fn
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input: 'Tensor') ->Tensor:
        output = self.activation_fn(self.conv_block(input))
        return output


class ResidualConvBlock3d(nn.Module):
    """3D ResNet block

    Parameters
    ----------
    n_layers : int, optional
        Number of convolutional layers, by default 1
    kernel_size : int, optional
        Kernel size, by default 3
    conv_layer_size : int, optional
        Latent channel size, by default 64
    activation_fn : nn.Module, optional
        Activation function, by default nn.Identity()
    """

    def __init__(self, n_layers: 'int'=1, kernel_size: 'int'=3, conv_layer_size: 'int'=64, activation_fn: 'nn.Module'=nn.Identity()):
        super().__init__()
        layers = [ConvolutionalBlock3d(in_channels=conv_layer_size, out_channels=conv_layer_size, kernel_size=kernel_size, batch_norm=True, activation_fn=activation_fn) for _ in range(n_layers - 1)]
        layers.append(ConvolutionalBlock3d(in_channels=conv_layer_size, out_channels=conv_layer_size, kernel_size=kernel_size, batch_norm=True))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, input: 'Tensor') ->Tensor:
        residual = input
        output = self.conv_layers(input)
        output = output + residual
        return output


class PixelShuffle3d(nn.Module):
    """3D pixel-shuffle operation

    Parameters
    ----------
    scale : int
        Factor to downscale channel count by

    Note
    ----
    Reference: http://www.multisilicon.com/blog/a25332339.html
    """

    def __init__(self, scale: 'int'):
        super().__init__()
        self.scale = scale

    def forward(self, input: 'Tensor') ->Tensor:
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = int(channels // self.scale ** 3)
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class SubPixel_ConvolutionalBlock3d(nn.Module):
    """Convolutional block with Pixel Shuffle operation

    Parameters
    ----------
    kernel_size : int, optional
        Kernel size, by default 3
    conv_layer_size : int, optional
        Latent channel size, by default 64
    scaling_factor : int, optional
        Pixel shuffle scaling factor, by default 2
    """

    def __init__(self, kernel_size: 'int'=3, conv_layer_size: 'int'=64, scaling_factor: 'int'=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=conv_layer_size, out_channels=conv_layer_size * scaling_factor ** 3, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = PixelShuffle3d(scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input: 'Tensor') ->Tensor:
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)
        return output


class SRResNet(Module):
    """3D convolutional super-resolution network

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels: int
        Number of outout channels
    large_kernel_size : int, optional
        convolutional kernel size for first and last convolution, by default 7
    small_kernel_size : int, optional
        convolutional kernel size for internal convolutions, by default 3
    conv_layer_size : int, optional
        Latent channel size, by default 32
    n_resid_blocks : int, optional
        Number of residual blocks before , by default 8
    scaling_factor : int, optional
        Scaling factor to increase the output feature size
        compared to the input (2, 4, or 8), by default 8
    activation_fn : Any, optional
        Activation function, by default "prelu"

    Example
    -------
    >>> #3D convolutional encoder decoder
    >>> model = modulus.models.srrn.SRResNet(
    ... in_channels=1,
    ... out_channels=2,
    ... conv_layer_size=4,
    ... scaling_factor=2)
    >>> input = torch.randn(4, 1, 8, 8, 8) #(N, C, D, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 2, 16, 16, 16])

    Note
    ----
    Based on the implementation:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution


    """

    def __init__(self, in_channels: 'int', out_channels: 'int', large_kernel_size: 'int'=7, small_kernel_size: 'int'=3, conv_layer_size: 'int'=32, n_resid_blocks: 'int'=8, scaling_factor: 'int'=8, activation_fn: 'str'='prelu'):
        super().__init__(meta=MetaData())
        self.var_dim = 1
        if isinstance(activation_fn, str):
            activation_fn = get_activation(activation_fn)
        scaling_factor = int(scaling_factor)
        if scaling_factor not in {2, 4, 8}:
            raise ValueError('The scaling factor must be 2, 4, or 8!')
        self.conv_block1 = ConvolutionalBlock3d(in_channels=in_channels, out_channels=conv_layer_size, kernel_size=large_kernel_size, batch_norm=False, activation_fn=activation_fn)
        self.residual_blocks = nn.Sequential(*[ResidualConvBlock3d(n_layers=2, kernel_size=small_kernel_size, conv_layer_size=conv_layer_size, activation_fn=activation_fn) for i in range(n_resid_blocks)])
        self.conv_block2 = ConvolutionalBlock3d(in_channels=conv_layer_size, out_channels=conv_layer_size, kernel_size=small_kernel_size, batch_norm=True)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(*[SubPixel_ConvolutionalBlock3d(kernel_size=small_kernel_size, conv_layer_size=conv_layer_size, scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])
        self.conv_block3 = ConvolutionalBlock3d(in_channels=conv_layer_size, out_channels=out_channels, kernel_size=large_kernel_size, batch_norm=False)

    def forward(self, in_vars: 'Tensor') ->Tensor:
        output = self.conv_block1(in_vars)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        output = self.conv_block3(output)
        return output


class SwinRNN(Module):
    """
    Implementation of SwinRNN https://arxiv.org/abs/2205.13158
    Args:
        img_size (Sequence[int], optional): Image size [T, Lat, Lon].
        patch_size (Sequence[int], optional): Patch token size [T, Lat, Lon].
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        embed_dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """

    def __init__(self, img_size=(2, 721, 1440), patch_size=(2, 4, 4), in_chans=70, out_chans=70, embed_dim=1536, num_groups=32, num_heads=8, window_size=7):
        super().__init__(meta=MetaData())
        input_resolution = img_size[1:]
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.swin_block1 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=2)
        self.down1 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down1x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj1 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder1 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=12)
        input_resolution = input_resolution[0] // 2, input_resolution[1] // 2
        self.swin_block2 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=2)
        self.down2 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down2x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj2 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder2 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=12)
        input_resolution = input_resolution[0] // 2, input_resolution[1] // 2
        self.swin_block3 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=2)
        self.down3 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down3x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj3 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder3 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=12)
        input_resolution = input_resolution[0] // 2, input_resolution[1] // 2
        self.swin_block4 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=2)
        self.lin_proj4 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder4 = SwinTransformer(embed_dim, input_resolution, num_heads, window_size, depth=12)
        self.up3x = ConvBlock(embed_dim, embed_dim, num_groups, upsample=1)
        self.up2x = ConvBlock(embed_dim * 2, embed_dim, num_groups, upsample=1)
        self.up1x = ConvBlock(embed_dim * 2, embed_dim, num_groups, upsample=1)
        self.pred = ConvBlock(embed_dim * 2, out_chans, out_chans, upsample=0)
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x: 'torch.Tensor'):
        B, Cin, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        xT = x[:, :, -1, :, :]
        x = self.cube_embedding(x).squeeze(2)
        h1 = self.swin_block1(x)
        x = self.down1(h1)
        h2 = self.swin_block2(x)
        x = self.down2(h2)
        h3 = self.swin_block3(x)
        x = self.down3(h3)
        h4 = self.swin_block4(x)
        B, Cin, H, W = xT.shape
        h1_d = torch.cat([xT.reshape(B, Cin, -1), h1.reshape(B, self.embed_dim, -1)], dim=1).transpose(1, 2)
        h1_d = self.lin_proj1(h1_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h1_d = self.swin_decoder1(h1_d)
        h1 = h1 + h1_d
        x2T = self.down1x(xT)
        B, Cin, H, W = x2T.shape
        h2_d = torch.cat([x2T.reshape(B, Cin, -1), h2.reshape(B, self.embed_dim, -1)], dim=1).transpose(1, 2)
        h2_d = self.lin_proj2(h2_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h2_d = self.swin_decoder2(h2_d)
        h2 = h2 + h2_d
        x3T = self.down2x(x2T)
        B, Cin, H, W = x3T.shape
        h3_d = torch.cat([x3T.reshape(B, Cin, -1), h3.reshape(B, self.embed_dim, -1)], dim=1).transpose(1, 2)
        h3_d = self.lin_proj3(h3_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h3_d = self.swin_decoder3(h3_d)
        h3 = h3 + h3_d
        x4T = self.down3x(x3T)
        B, Cin, H, W = x4T.shape
        h4_d = torch.cat([x4T.reshape(B, Cin, -1), h4.reshape(B, self.embed_dim, -1)], dim=1).transpose(1, 2)
        h4_d = self.lin_proj4(h4_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h4_d = self.swin_decoder4(h4_d)
        h4 = h4 + h4_d
        h4_up = self.up3x(h4)
        h3_up = self.up2x(torch.cat([h3, h4_up], dim=1))
        h2_up = self.up1x(torch.cat([h2, h3_up], dim=1))
        h1_up = self.pred(torch.cat([h1, h2_up], dim=1))
        x_h1 = xT + h1_up
        return x_h1


class RotaryEmbedding(nn.Module):
    """ROPE: Rotary Position Embedding"""

    def __init__(self, dim, min_freq=1 / 2, scale=1.0):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        t = coordinates.type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=421 * 421):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Physics_Attention_Irregular_Mesh(nn.Module):
    """for irregular meshes in 1D, 2D or 3D space"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)
        out_x = torch.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    """for structured mesh in 2D space"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, H=101, W=31, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)
        out_x = torch.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    """for structured mesh in 3D space"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)
        out_x = torch.einsum('bhgc,bhng->bhnc', out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(self, num_heads: 'int', hidden_dim: 'int', dropout: 'float', act='gelu', mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32, H=85, W=85):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num, H=H, W=W)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


@dataclass
class Model(torch.nn.Module):
    """Minimal torch.nn.Module to test MSE_SSIM"""
    output_channels = 2
    output_time_dim = 1
    input_time_dim = 1
    output_variables = ['tcwv', 't2m']


class Transolver(Module):
    """Transformer-based solver for PDEs.

    Note
    ----
    Transolver is a model specifically designed for structured 2D mesh data.

    Parameters
    ----------
    space_dim : int
        The spatial dimension of the input data.
    n_layers : int
        The number of transformer layers.
    n_hidden : int
        The hidden dimension of the transformer.
    dropout : float
        The dropout rate.
    n_head : int
        The number of attention heads.
    Time_Input : bool
        Whether to include time embeddings.
    act : str
        The activation function.
    mlp_ratio : int
        The ratio of hidden dimension in the MLP.
    fun_dim : int
        The dimension of the function.
    out_dim : int
        The output dimension.
    slice_num : int
        The number of slices in the structured attention.
    ref : int
        The reference dimension.
    unified_pos : bool
        Whether to use unified positional embeddings.
    H : int
        The height of the mesh.
    W : int
        The width of the mesh.
    """

    def __init__(self, space_dim: 'int', n_layers: 'int', n_hidden: 'int', dropout: 'float', n_head: 'int', Time_Input: 'bool', act: 'str', mlp_ratio: 'int', fun_dim: 'int', out_dim: 'int', slice_num: 'int', ref: 'int', unified_pos: 'bool', H: 'int', W: 'int') ->None:
        super().__init__(meta=MetaData())
        self.H = H
        self.W = W
        self.model = Model(space_dim=space_dim, n_layers=n_layers, n_hidden=n_hidden, dropout=dropout, n_head=n_head, Time_Input=Time_Input, act=act, mlp_ratio=mlp_ratio, fun_dim=fun_dim, out_dim=out_dim, slice_num=slice_num, ref=ref, unified_pos=unified_pos, H=H, W=W)

    def forward(self, x: 'torch.Tensor', fx: 'torch.Tensor'=None, T: 'torch.Tensor'=None) ->torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        fx : torch.Tensor
            The function tensor.
        T : torch.Tensor
            The time tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        y = self.model(x, fx, T)
        y = y.reshape(x.shape[0], self.H, self.W, -1)
        return y


class MLPNet(Module):
    """
    A Multilayer Perceptron (MLP) network implemented in PyTorch, configurable with
    a variable number of hidden layers and layer normalization.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    output_size : int
        Number of output channels
    layer_norm : boolean
        If to apply layer normalization in the output layer, default True

    Example
    -------
    # # Use MLPNet to encode the features
    # >>> model = modulus.models.graph_network.MLPNet(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... output_size=128)
    # >>> input = torch.randn([5193, 128]) #(N, C)
    # >>> output = model(input)
    # >>> output.size()
    # torch.Size([5193, 128])
    ----
    """

    def __init__(self, mlp_hidden_size: 'int'=128, mlp_num_hidden_layers: 'int'=2, output_size: 'int'=128, layer_norm: 'bool'=True):
        if not (mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and output_size >= 0):
            raise ValueError('Invalid arch params')
        super().__init__(meta=MetaData(name='vfgn_mlpnet'))
        self.mlp_hidden_size = mlp_hidden_size
        self.lins = []
        if mlp_num_hidden_layers > 1:
            for i in range(mlp_num_hidden_layers - 1):
                self.lins.append(Linear(mlp_hidden_size, mlp_hidden_size))
        self.lins = torch.nn.ModuleList(self.lins)
        self.lin_e = Linear(mlp_hidden_size, output_size)
        self.layer_norm = layer_norm
        self.relu = ReLU()

    def dynamic(self, name: 'str', module_class, *args, **kwargs):
        """Use dynamic layer to create 1st layer according to the input node number"""
        if not hasattr(self, name):
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)

    def forward(self, x):
        origin_device = x.device
        lin_s = self.dynamic('lin_s', Linear, x.shape[-1], self.mlp_hidden_size)
        lin_s = lin_s
        x = lin_s(x)
        x = self.relu(x)
        for lin_i in self.lins:
            x = lin_i(x)
            x = self.relu(x)
        x = self.lin_e(x)
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[1:])
        return x


class EncoderNet(Module):
    """
    Construct EncoderNet based on the NLPNet architecture.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    """

    def __init__(self, mlp_hidden_size: 'int'=128, mlp_num_hidden_layers: 'int'=2, latent_size: 'int'=128):
        if not (mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and latent_size >= 0):
            raise ValueError('Invalid arch params - EncoderNet')
        super().__init__(meta=MetaData(name='vfgn_encoder'))
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self.edge_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
        self.node_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, node_attr, edge_attr):
        node_attr = self.node_mlp(node_attr)
        edge_attr = self.edge_mlp(edge_attr)
        return node_attr, edge_attr


class EdgeBlock(Module):
    """
    Update the edge attributes by collecting the sender and/or receiver-nodes'
    edge attributes, pass through the edge-MLP network.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    use_receiver_nodes : bool, optional, default = True
        whether to take the receiver-node's edges atrributes into compute
    use_sender_nodes : bool, optional, default = True
        whether to take the sender-node's edges atrributes into compute

    Example
    -------
    # >>> #2D convolutional encoder decoder
    # >>> model = modulus.models.graph_network.EdgeBlock(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... latent_size=128,
    # ... node_dim=0)
    # >>> input = (node_attr, edge_attr, receiver_list, sender_list)
    # >>> output = node_attr, updated_edge_attr, receiver_list, sender_list
    # >>> output.size()

    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, latent_size, node_dim=0, use_receiver_nodes=True, use_sender_nodes=True):
        super().__init__(meta=MetaData(name='vfgn_edgeblock'))
        self.node_dim = node_dim
        self._edge_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
        self.use_receiver_nodes = use_receiver_nodes
        self.use_sender_nodes = use_sender_nodes

    def forward(self, node_attr, edge_attr, receivers, senders):
        edges_to_collect = []
        edges_to_collect.append(edge_attr)
        if self.use_receiver_nodes:
            receivers_edge = node_attr[receivers, :]
            edges_to_collect.append(receivers_edge)
        if self.use_sender_nodes:
            senders_edge = node_attr[senders, :]
            edges_to_collect.append(senders_edge)
        collected_edges = torch.cat(edges_to_collect, axis=-1)
        updated_edges = self._edge_model(collected_edges)
        return node_attr, updated_edges, receivers, senders


class NodeBlock(Module):
    """
    Update the nodes attributes by collecting the sender and/or receiver-nodes'
    edge attributes, pass through the node-MLP network.

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    use_receiver_nodes : bool, optional, default = True
        whether to take the receiver-node's edges atrributes into compute
    use_sender_nodes : bool, optional, default = True
        whether to take the sender-node's edges atrributes into compute

    # Example
    # -------
    # >>> #2D convolutional encoder decoder
    # >>> model = modulus.models.graph_network.NodeBlock(
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... latent_size=128,
    # ... node_dim=0)
    # >>> input = (node_attr, edge_attr, receiver_list, sender_list)
    # >>> output = updated_node_attr, edge_attr, receiver_list, sender_list
    # >>> output.size()

    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr='add', node_dim=0, use_received_edges=True, use_sent_edges=False):
        super().__init__(meta=MetaData(name='vfgn_nodeblock'))
        self.aggr = aggr
        self.node_dim = node_dim
        self.use_received_edges = use_received_edges
        self.use_sent_edges = use_sent_edges
        self._node_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, x, edge_attr, receivers, senders):
        nodes_to_collect = []
        nodes_to_collect.append(x)
        dim_size = x.shape[self.node_dim]
        if self.use_received_edges:
            receivers_edge = scatter(dim=self.node_dim, dim_size=dim_size, index=receivers, src=edge_attr, reduce=self.aggr)
            nodes_to_collect.append(receivers_edge)
        if self.use_sent_edges:
            senders_edge = scatter(dim=self.node_dim, dim_size=dim_size, index=senders, src=edge_attr, reduce=self.aggr)
            nodes_to_collect.append(senders_edge)
        collected_nodes = torch.cat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        return updated_nodes, edge_attr, receivers, senders


class InteractionNet(torch.nn.Module):
    """
    Iterate to compute the edge attributes, then node attributes

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr='add', node_dim=0):
        super(InteractionNet, self).__init__()
        self._edge_block = EdgeBlock(mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim)
        self._node_block = NodeBlock(mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim)

    def forward(self, x, edge_attr, receivers, senders):
        if not x.shape[-1] == edge_attr.shape[-1]:
            raise ValueError('node feature size should equal to edge feature size')
        return self._node_block(*self._edge_block(x, edge_attr, receivers, senders))


class ResInteractionNet(torch.nn.Module):
    """
    Update the edge attributes and node attributes

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    latent_size : int
        Number of latent channels
    aggr : str, optional, default = "add"
        operation to collect the node attributes
    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr='add', node_dim=0):
        super(ResInteractionNet, self).__init__()
        self.itn = InteractionNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim)

    def forward(self, x, edge_attr, receivers, senders):
        x_res, edge_attr_res, receivers, senders = self.itn(x, edge_attr, receivers, senders)
        x_new = x + x_res
        edge_attr_new = edge_attr + edge_attr_res
        return x_new, edge_attr_new, receivers, senders


class DecoderNet(Module):
    """
    Construct DecoderNet based on the NLPNet architecture. Used for
    decoding the predicted features with multi-layer perceptron network module

    Parameters
    ----------
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    output_size : int
        Number of output channels
    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, output_size):
        if not (mlp_hidden_size >= 0 and mlp_num_hidden_layers >= 0 and output_size >= 0):
            raise ValueError('Invalid arch params - DecoderNet')
        super().__init__(meta=MetaData(name='vfgn_decoder'))
        self.mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, output_size, layer_norm=False)

    def forward(self, x):
        x = self.mlp(x)
        return x


class EncodeProcessDecode(Module):
    """
    Construct the network architecture that consists of Encoder - Processor - Decoder modules

    Parameters
    ----------
    latent_size : int
        Number of latent channels
    mlp_hidden_size : int
        Number of channels/ features in the hidden layers
    mlp_num_hidden_layers : int
        Number of hidden layers
    num_message_passing_steps : int, default = 10
        Number of message passing steps
    output_size : int
        Number of output channels
    device_list : list[str], optional
        device to execute the computation

    # Example
    # -------
    # >>> #Use EncodeProcessDecode to update the node, edge features
    # >>> model = modulus.models.graph_network.EncodeProcessDecode(
    # ... latent_size=128,
    # ... mlp_hidden_size=128,
    # ... mlp_num_hidden_layers=2,
    # ... num_message_passing_steps=10,
    # ... output_size=3)
    # >>> node_attr = torch.randn([1394, 61]) #(node_cnt, node_feat_sizes)
    # >>> edge_attr = torch.randn([5193, 4]) #(edge_cnt, edge_feat_sizes)
    # >>> invar_receivers = torch.Size([5193]) : int # node index list
    # >>> invar_senders = torch.Size([5193]) : int # node index list
    # >>> invar = (node_attr, edge_attr, invar_receivers, invar_senders)
    # >>> output = model(*invar, )
    # >>> output.size()
    # torch.Size([1394, 3])    #(node_cnt, output_size)
    """

    def __init__(self, latent_size, mlp_hidden_size, mlp_num_hidden_layers, num_message_passing_steps, output_size, device_list=None):
        if not (latent_size > 0 and mlp_hidden_size > 0 and mlp_num_hidden_layers > 0):
            raise ValueError('Invalid arch params - EncodeProcessDecode')
        if not num_message_passing_steps > 0:
            raise ValueError('Invalid arch params - EncodeProcessDecode')
        super().__init__(meta=MetaData(name='vfgn_encoderprocess_decode'))
        if device_list is None:
            self.device_list = ['cpu']
        else:
            self.device_list = device_list
        self._encoder_network = EncoderNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
        self._processor_networks = []
        for _ in range(num_message_passing_steps):
            self._processor_networks.append(InteractionNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size))
        self._processor_networks = torch.nn.ModuleList(self._processor_networks)
        self._decoder_network = DecoderNet(mlp_hidden_size, mlp_num_hidden_layers, output_size)

    def set_device(self, device_list):
        """device list"""
        self.device_list = device_list

    def forward(self, x, edge_attr, receivers, senders):
        """
        x:
            Torch tensor of node attributes, shape: (batch_size, node_number, feature_size)
        edge_attr:
            Torch tensor of edge_attr attributes, shape: (batch_size, edge_number, feature_size)
        receivers/ senders:
            Torch tensor, list of node indexes, shape: (batch_size,  edge_list_size:[list of node indexes])
        """
        x, edge_attr = self._encoder_network(x, edge_attr)
        pre_x = x
        pre_edge_attr = edge_attr
        n_steps = len(self._processor_networks)
        n_inter = int(n_steps / len(self.device_list))
        i = 0
        j = 0
        origin_device = x.device
        for processor_network_k in self._processor_networks:
            p_device = self.device_list[j]
            processor_network_k = processor_network_k
            pre_x = pre_x
            pre_edge_attr = pre_edge_attr
            receivers = receivers
            senders = senders
            diff_x, diff_edge_attr, receivers, senders = checkpoint(processor_network_k, pre_x, pre_edge_attr, receivers, senders)
            pre_x = x + diff_x
            pre_edge_attr = edge_attr + diff_edge_attr
            i += 1
            if i % n_inter == 0:
                j += 1
        x = self._decoder_network(pre_x)
        return x


STD_EPSILON = 1e-08


class LearnedSimulator(Module):
    """
    Construct the Simulator model architecture

    Parameters
    ----------
    num_dimensions : int
        Number of dimensions to make the prediction
    num_seq : int
        Number of sintering steps
    boundaries : list[list[float]]
        boundary value that the object is placed/ normalized in,
        i.e.[[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    num_particle_types : int
        Number of types to differentiate the different nodes, i.e. fixed/ moving nodes
    particle_type_embedding_size: int
        positional embedding dimension with different particle types,
        in torch.nn.Embedding()
    normalization_stats: dict{list[float]}
        Stored in metadata.json
        {'acceleration': acceleration_stats, 'velocity': velocity_stats, 'context': context_stats}
    graph_mode : str, optional
    connectivity_param: float
        Distance to normalize the displacement between nodes

    Example
    -------
    # >>> model = modulus.models.graph_network.LearnedSimulator(
    # ... num_dimensions=3*5, # metadata['dim'] * PREDICT_LENGTH
    # ... num_seq=2,
    # ... boundaries=128)

    # >>> input = torch.randn([5193, 128]) #(N, C)
    # >>> output = model(input)
    # >>> output.size()
    # torch.Size([5193, 128])
    ----
    """

    def __init__(self, num_dimensions: 'int'=3, num_seq: 'int'=5, boundaries: 'list[list[float]]'=None, num_particle_types: 'int'=3, particle_type_embedding_size: 'int'=16, normalization_stats: 'map'=None, graph_mode: 'str'='radius', connectivity_param: 'float'=0.015):
        if not (num_dimensions >= 0 and num_seq >= 3):
            raise ValueError('Invalid arch params - LearnedSimulator')
        super().__init__(meta=MetaData(name='vfgn_simulator'))
        self._latent_size = 128
        self._mlp_hidden_size = 128
        self._mlp_num_hidden_layers = 2
        self._num_message_passing_steps = 10
        self._num_dimensions = num_dimensions
        self._num_seq = num_seq
        self._connectivity_param = connectivity_param
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats
        self.graph_mode = graph_mode
        self._graph_network = EncodeProcessDecode(self._latent_size, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._num_message_passing_steps, self._num_dimensions)
        self._num_particle_types = num_particle_types
        self.embedding = Embedding(self._num_particle_types + 1, particle_type_embedding_size)
        self.message_passing_devices = []

    def setMessagePassingDevices(self, devices):
        """
        setts the devices to be used for message passing in the neural network model.
        """
        self.message_passing_devices = devices

    def to(self, device):
        """Device transfer"""
        new_self = super(LearnedSimulator, self)
        new_self._boundaries = self._boundaries
        for key in self._normalization_stats:
            new_self._normalization_stats[key]
        if device != 'cpu':
            self._graph_network.set_device(self.message_passing_devices)
        return new_self

    def time_diff(self, input_seq):
        """
        Calculates the difference between consecutive elements in a sequence, effectively computing the discrete time derivative.
        """
        return input_seq[:, 1:] - input_seq[:, :-1]

    def _compute_connectivity_for_batch(self, senders_list, receivers_list, n_node, n_edge):
        """
        Dynamically update the edge features with random dropout
        For each graph, randomly select whether apply edge drop-out to this node
        If applying random drop-out, a default drop_out_rate = 0.6 is applied to the edges
        """
        senders_per_graph_list = np.split(senders_list, np.cumsum(n_edge[:-1]), axis=0)
        receivers_per_graph_list = np.split(receivers_list, np.cumsum(n_edge[:-1]), axis=0)
        receivers_list = []
        senders_list = []
        n_edge_list = []
        num_nodes_in_previous_graphs = 0
        n = n_node.shape[0]
        drop_out_rate = 0.6
        for i in range(n):
            total_num_edges_graph_i = len(senders_per_graph_list[i])
            random_num = False
            if random_num:
                choiced_indices = random.choices([j for j in range(total_num_edges_graph_i)], k=int(total_num_edges_graph_i * drop_out_rate))
                choiced_indices = sorted(choiced_indices)
                senders_graph_i = senders_per_graph_list[i][choiced_indices]
                receivers_graph_i = receivers_per_graph_list[i][choiced_indices]
            else:
                senders_graph_i = senders_per_graph_list[i]
                receivers_graph_i = receivers_per_graph_list[i]
            num_edges_graph_i = len(senders_graph_i)
            n_edge_list.append(num_edges_graph_i)
            receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
            senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)
            num_nodes_graph_i = n_node[i]
            num_nodes_in_previous_graphs += num_nodes_graph_i
        senders = np.concatenate(senders_list, axis=0).astype(np.int32)
        receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
        return senders, receivers

    def get_random_walk_noise_for_position_sequence(self, position_sequence, noise_std_last_step):
        """Returns random-walk noise in the velocity applied to the position."""
        velocity_sequence = self.time_diff(position_sequence)
        num_velocities = velocity_sequence.shape[1]
        velocity_sequence_noise = torch.empty(velocity_sequence.shape, dtype=velocity_sequence.dtype).normal_(mean=0, std=noise_std_last_step / num_velocities ** 0.5)
        velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
        position_sequence_noise = torch.cat([torch.zeros(velocity_sequence_noise[:, 0:1].shape, dtype=velocity_sequence.dtype), torch.cumsum(velocity_sequence_noise, axis=1)], axis=1)
        return position_sequence_noise

    def EncodingFeature(self, position_sequence, n_node, n_edge, senders_list, receivers_list, global_context, particle_types):
        """
        Feature encoder contains 3 parts:
            - Adding the node features that includes: position, velocity, sequence of accelerations
            - Adding the edge features with random dropout applied
            - Adding the global features to the node features, in this case, sintering temperature is includes
        """
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = self.time_diff(position_sequence)
        acceleration_sequence = self.time_diff(velocity_sequence)
        senders, receivers = self._compute_connectivity_for_batch(senders_list.cpu().detach().numpy(), receivers_list.cpu().detach().numpy(), n_node.cpu().detach().numpy(), n_edge.cpu().detach().numpy())
        senders = torch.LongTensor(senders)
        receivers = torch.LongTensor(receivers)
        node_features = []
        velocity_stats = self._normalization_stats['velocity']
        normalized_velocity_sequence = (velocity_sequence - velocity_stats.mean) / velocity_stats.std
        normalized_velocity_sequence = normalized_velocity_sequence[:, -1]
        flat_velocity_sequence = normalized_velocity_sequence.reshape([normalized_velocity_sequence.shape[0], -1])
        node_features.append(flat_velocity_sequence)
        acceleration_stats = self._normalization_stats['acceleration']
        normalized_acceleration_sequence = (acceleration_sequence - acceleration_stats.mean) / acceleration_stats.std
        flat_acceleration_sequence = normalized_acceleration_sequence.reshape([normalized_acceleration_sequence.shape[0], -1])
        node_features.append(flat_acceleration_sequence)
        if self._num_particle_types > 1:
            particle_type_embedding = self.embedding(particle_types)
            node_features.append(particle_type_embedding)
        edge_features = []
        normalized_relative_displacements = (most_recent_position.index_select(0, senders.squeeze()) - most_recent_position.index_select(0, receivers.squeeze())) / self._connectivity_param
        edge_features.append(normalized_relative_displacements)
        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)
        if global_context is not None:
            context_stats = self._normalization_stats['context']
            global_context = (global_context - context_stats.mean) / torch.maximum(context_stats.std, torch.FloatTensor([STD_EPSILON]))
            global_features = []
            for i in range(global_context.shape[0]):
                global_context_ = torch.unsqueeze(global_context[i], 0)
                context_i = torch.repeat_interleave(global_context_, n_node[i], dim=0)
                global_features.append(context_i)
            global_features = torch.cat(global_features, 0)
            global_features = global_features.reshape(global_features.shape[0], -1)
            node_features.append(global_features)
        x = torch.cat(node_features, -1)
        edge_attr = torch.cat(edge_features, -1)
        x = x.float()
        edge_attr = edge_attr.float()
        return x, edge_attr, senders, receivers

    def DecodingFeature(self, normalized_accelerations, position_sequence, predict_length):
        """Feature decoder"""
        normalized_accelerations = normalized_accelerations.double()
        acceleration_stats = self._normalization_stats['acceleration']
        normalized_accelerations = normalized_accelerations.reshape([-1, predict_length, 3])
        accelerations = normalized_accelerations * acceleration_stats.std + acceleration_stats.mean
        velocity_changes = torch.cumsum(accelerations, axis=1, dtype=accelerations.dtype)
        most_recent_velocity = position_sequence[:, -1] - position_sequence[:, -2]
        most_recent_velocity = torch.unsqueeze(most_recent_velocity, axis=1)
        most_recent_velocities = torch.tile(most_recent_velocity, [1, predict_length, 1])
        velocities = most_recent_velocities + velocity_changes
        position_changes = torch.cumsum(velocities, axis=1, dtype=velocities.dtype)
        most_recent_position = position_sequence[:, -1]
        most_recent_position = torch.unsqueeze(most_recent_position, axis=1)
        most_recent_positions = torch.tile(most_recent_position, [1, predict_length, 1])
        new_positions = most_recent_positions + position_changes
        return new_positions

    def _inverse_decoder_postprocessor(self, next_positions, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        most_recent_positions = position_sequence[:, -2:]
        previous_positions = torch.cat([most_recent_positions, next_positions[:, :-1]], axis=1)
        positions = torch.cat([torch.unsqueeze(position_sequence[:, -1], axis=1), next_positions], axis=1)
        velocities = positions - previous_positions
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        acceleration_stats = self._normalization_stats['acceleration']
        normalized_accelerations = (accelerations - acceleration_stats.mean) / acceleration_stats.std
        normalized_accelerations = normalized_accelerations.reshape([-1, self._num_dimensions])
        normalized_accelerations = normalized_accelerations.float()
        return normalized_accelerations

    def inference(self, position_sequence: 'Tensor', n_particles_per_example, n_edges_per_example, senders, receivers, predict_length, global_context=None, particle_types=None) ->Tensor:
        """
        Inference with the LearnedSimulator network

        Args:
        position_sequence: Model inference input tensor
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
            i.e. torch.Size([1394, 5, 3])

        n_particles_per_example: torch.Size([1]), [tf.shape(pos)[0]]
            torch.Tensor([node_cnt], dtype=torch.int32)
            i.e. tensor([1394])
        n_edges_per_example: torch.Size([1]), [tf.shape(context['senders'])[0]]
            torch.Tensor([edge_cnt], dtype=torch.int32)
            i.e. tensor([8656])

        senders: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        receivers: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        predict_length: prediction steps, int
            i.e. 1
        particle_types: torch.Tensor([node_cnt], dtype=torch.int32)
            torch.Size([1394])
        global_context: torch.Tensor([sim_step, feat_dim], dtype=torch.float)
            i.e. torch.Size([34, 1])
        """
        input_graph = self.EncodingFeature(position_sequence, n_particles_per_example, n_edges_per_example, senders, receivers, global_context, particle_types)
        predicted_normalized_accelerations = self._graph_network(*input_graph)
        next_position = self.DecodingFeature(predicted_normalized_accelerations, position_sequence, predict_length)
        return next_position

    def forward(self, next_positions: 'Tensor', position_sequence_noise: 'Tensor', position_sequence: 'Tensor', n_particles_per_example, n_edges_per_example, senders: 'Tensor', receivers: 'Tensor', predict_length, global_context=None, particle_types=None) ->Tensor:
        """
        Training step with the LearnedSimulator network,
        Produces normalized and predicted nodal acceleration.

        Args:
        next_position: Model prediction target tensor
            torch.Tensor([node_cnt, pred_dim] ,)
            i.e. torch.Size([1394, 3])
        position_sequence_noise: Tensor of the same shape as `position_sequence`
            with the noise to apply to each particle.
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
        position_sequence: Model inference input tensor
            torch.Tensor([node_cnt, input_step, pred_dim] ,)
            i.e. torch.Size([1394, 5, 3])

        n_particles_per_example: torch.Size([1]), [tf.shape(pos)[0]]
            i.e. tensor([1394])
        n_edges_per_example: torch.Size([1]), [tf.shape(context['senders'])[0]]
            i.e. tensor([8656])
        senders: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        receivers: torch.Size([edge_cnt], dtype=torch.int32)
            contains node index
        predict_length: prediction steps, int
            i.e. 1
        particle_types: torch.Tensor([node_cnt], dtype=torch.int32)
            torch.Size([1394])
        global_context: torch.Tensor([sim_step, feat_dim], dtype=torch.float)
            i.e. torch.Size([34, 1])

        Returns:
            Tensors of shape [num_particles_in_batch, num_dimensions] with the
            predicted and target normalized accelerations.
        """
        noisy_position_sequence = position_sequence + position_sequence_noise
        input_graph = self.EncodingFeature(noisy_position_sequence, n_particles_per_example, n_edges_per_example, senders, receivers, global_context, particle_types)
        predicted_normalized_accelerations = self._graph_network(*input_graph)
        most_recent_noise = position_sequence_noise[:, -1]
        most_recent_noise = torch.unsqueeze(most_recent_noise, axis=1)
        most_recent_noises = torch.tile(most_recent_noise, [1, predict_length, 1])
        next_position_adjusted = next_positions + most_recent_noises
        target_normalized_acceleration = self._inverse_decoder_postprocessor(next_position_adjusted, noisy_position_sequence)
        return predicted_normalized_accelerations, target_normalized_acceleration

    def get_normalized_acceleration(self, acceleration, predict_length):
        """
        Normalizes the acceleration data using predefined statistics and
        replicates it across a specified prediction length.
        """
        acceleration_stats = self._normalization_stats['acceleration']
        normalized_acceleration = (acceleration - acceleration_stats.mean) / acceleration_stats.std
        normalized_acceleration = torch.tile(normalized_acceleration, [predict_length])
        return normalized_acceleration


class CellAreaWeightedLossFunction(nn.Module):
    """Loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """
        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss


class CustomCellAreaWeightedLossAutogradFunction(torch.autograd.Function):
    """Autograd fuunction for custom loss with cell area weighting."""

    @staticmethod
    def forward(ctx, invar: 'torch.Tensor', outvar: 'torch.Tensor', area: 'torch.Tensor'):
        """Forward of custom loss function with cell area weighting."""
        diff = invar - outvar
        loss = diff ** 2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, area)
        loss = loss.mean()
        loss_grad = diff * (2.0 / math.prod(invar.shape))
        loss_grad *= area.unsqueeze(0).unsqueeze(0)
        ctx.save_for_backward(loss_grad)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: 'torch.Tensor'):
        """Backward method of custom loss function with cell area weighting."""
        grad_invar, = ctx.saved_tensors
        return grad_invar * grad_loss, None, None


class CustomCellAreaWeightedLossFunction(CellAreaWeightedLossFunction):
    """Custom loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area: 'torch.Tensor'):
        super().__init__(area)

    def forward(self, invar: 'torch.Tensor', outvar: 'torch.Tensor') ->torch.Tensor:
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """
        return CustomCellAreaWeightedLossAutogradFunction.apply(invar, outvar, self.area)


class GraphCastLossFunction(nn.Module):
    """Loss function as specified in GraphCast.
    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area, channels_list, dataset_metadata_path, time_diff_std_path):
        super().__init__()
        self.area = area
        self.channel_dict = self.get_channel_dict(dataset_metadata_path, channels_list)
        self.variable_weights = self.assign_variable_weights()
        self.time_diff_std = self.get_time_diff_std(time_diff_std_path, channels_list)

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.
        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """
        loss = (invar - outvar) ** 2
        loss = loss * 1.0 / torch.square(self.time_diff_std.view(1, -1, 1, 1))
        variable_weights = self.variable_weights.view(1, -1, 1, 1)
        loss = loss * variable_weights
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss

    def get_time_diff_std(self, time_diff_std_path, channels_list):
        """Gets the time difference standard deviation"""
        if time_diff_std_path is not None:
            time_diff_np = np.load(time_diff_std_path)
            time_diff_np = time_diff_np[:, channels_list, ...]
            return torch.FloatTensor(time_diff_np)
        else:
            return torch.tensor([1.0], dtype=torch.float)

    def get_channel_dict(self, dataset_metadata_path, channels_list):
        """Gets lists of surface and atmospheric channels"""
        with open(dataset_metadata_path, 'r') as f:
            data_json = json.load(f)
            channel_list = [data_json['coords']['channel'][c] for c in channels_list]
            channel_dict = {'surface': [], 'atmosphere': []}
            for each_channel in channel_list:
                if each_channel[-1].isdigit():
                    channel_dict['atmosphere'].append(each_channel)
                else:
                    channel_dict['surface'].append(each_channel)
            return channel_dict

    def parse_variable(self, variable_list):
        """Parse variable into its letter and numeric parts."""
        for i, char in enumerate(variable_list):
            if char.isdigit():
                return variable_list[:i], int(variable_list[i:])

    def calculate_linear_weights(self, variables):
        """Calculate weights for each variable group."""
        groups = defaultdict(list)
        for variable in variables:
            letter, number = self.parse_variable(variable)
            groups[letter].append((variable, number))
        weights = {}
        for values in groups.values():
            total = sum(number for _, number in values)
            for variable, number in values:
                weights[variable] = number / total
        return weights

    def assign_surface_weights(self):
        """Assigns weights to surface variables"""
        surface_weights = {i: (0.1) for i in self.channel_dict['surface']}
        if 't2m' in surface_weights:
            surface_weights['t2m'] = 1
        return surface_weights

    def assign_atmosphere_weights(self):
        """Assigns weights to atmospheric variables"""
        return self.calculate_linear_weights(self.channel_dict['atmosphere'])

    def assign_variable_weights(self):
        """assigns per-variable per-pressure level weights"""
        surface_weights_dict = self.assign_surface_weights()
        atmosphere_weights_dict = self.assign_atmosphere_weights()
        surface_weights = list(surface_weights_dict.values())
        atmosphere_weights = list(atmosphere_weights_dict.values())
        variable_weights = torch.cat((torch.FloatTensor(surface_weights), torch.FloatTensor(atmosphere_weights)))
        return variable_weights


class MiniNetwork(torch.nn.Module):
    """Mini network with one parameter for testing cuda graph support of data pipes"""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1))

    def forward(self, inputs: 'Tuple[Tensor, ...]') ->Tuple[Tensor, ...]:
        output = tuple(self.param * invar for invar in inputs)
        return output


class ProcessGroupNode:
    """
    Class to store the attributes of a distributed process group

    Attributes
    ----------
    name : str
        Name of the process group
    size : Optional[int]
        Optional, number of processes in the process group
    """

    def __init__(self, name: 'str', size: 'Optional[int]'=None):
        """
        Constructor for the ProcessGroupNode class

        Parameters
        ----------
        name : str
            Name of the process group
        size : Optional[int]
            Optional, size of the process group
        """
        self.name = name
        self.size = size

    def __str__(self):
        """
        String representation of the process group node

        Returns
        -------
        str
            String representation of the process group node
        """
        return f'ProcessGroupNode(name={self.name}, size={self.size}, '

    def __repr__(self):
        """
        String representation of the process group node

        Returns
        -------
        str
            String representation of the process group node
        """
        return self.__str__()


def _tree_product_reduction(tree, node_id, verbose=False):
    """
    Function to traverse a tree and compute the product reduction of
    the sub-tree for each node starting from `node_id`
    """
    children = tree.children(node_id)
    node = tree.get_node(node_id)
    if not children:
        if node.data.size is None:
            raise AssertionError('Leaf nodes should have a valid size set')
        return node.data.size
    product = 1
    for child in children:
        product *= _tree_product_reduction(tree, child.identifier)
    if node.data.size != product:
        if verbose:
            None
        node.data.size = product
    return product


class ProcessGroupConfig:
    """
    Class to define the configuration of a model's parallel process group structure as a
    tree. Each node of the tree is of type `ProcessGroupNode`.

    Once the process group config structure (i.e, the tree structure) is set, it is
    sufficient to set only the sizes for each leaf process group. Then, the size of
    every parent group can be automatically computed as the product reduction of the
    sub-tree of that parent group node.

    Examples
    --------
    >>> from modulus.distributed import ProcessGroupNode, ProcessGroupConfig
    >>>
    >>> # Create world group that contains all processes that are part of this job
    >>> world = ProcessGroupNode("world")
    >>>
    >>> # Create the process group config with the highest level process group
    >>> config = ProcessGroupConfig(world)
    >>>
    >>> # Create model and data parallel sub-groups
    >>> # Sub-groups of a single node are guaranteed to be orthogonal by construction
    >>> # Nodes can be added with either the name of the node or the node itself
    >>> config.add_node(ProcessGroupNode("model_parallel"), parent=world)
    >>> config.add_node(ProcessGroupNode("data_parallel"), parent="world")
    >>>
    >>> # Create spatial and channel parallel sub-groups
    >>> config.add_node(ProcessGroupNode("spatial_parallel"), parent="model_parallel")
    >>> config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")
    >>>
    >>> config.leaf_groups()
    ['data_parallel', 'spatial_parallel', 'channel_parallel']
    >>>
    >>> # Set leaf group sizes
    >>> # Note: product of all leaf-node sizes should be the world size
    >>> group_sizes = {"channel_parallel": 3, "spatial_parallel": 2, "data_parallel": 4}
    >>> config.set_leaf_group_sizes(group_sizes)  # Update all parent group sizes too
    >>> config.get_node("model_parallel").size
    6
    """

    def __init__(self, node: 'ProcessGroupNode'):
        """
        Constructor to the ProcessGroupConfig class

        Parameters
        ----------
        node : ProcessGroupNode
            Root node of the tree, typically would be 'world'
            Note, it is generally recommended to set the child groups for 'world'
            to 'model_parallel' and 'data_parallel' to aid with distributed
            data parallel training unless there is a specific reason to choose a
            different structure
        """
        self.root = node
        self.root_id = node.name
        self.tree = Tree()
        self.tree.create_node(node.name, node.name, data=node)

    def add_node(self, node: 'ProcessGroupNode', parent=Union[str, ProcessGroupNode]):
        """
        Add a node to the process group config

        Parameters
        ----------
        node : ProcessGroupNode
            The new node to be added to the config
        parent : Union[str, ProcessGroupNode]
            Parent node of the node to be added. Should already be in the config.
            If str, it is the name of the parent node. Otherwise, the parent
            ProcessGroupNode itself.
        """
        if isinstance(parent, ProcessGroupNode):
            parent = parent.name
        self.tree.create_node(node.name, node.name, data=node, parent=parent)

    def get_node(self, name: 'str') ->ProcessGroupNode:
        """
        Method to get the node given the name of the node

        Parameters
        ----------
        name : str
            Name of the node to retrieve

        Returns
        -------
        ProcessGroupNode
            Node with the given name from the config
        """
        return self.tree.get_node(name).data

    def update_parent_sizes(self, verbose: 'bool'=False) ->int:
        """
        Method to update parent node sizes after setting the sizes for each leaf node

        Parameters
        ----------
        verbose : bool
            If True, print a message each time a parent node size was updated

        Returns
        -------
        int
            Size of the root node
        """
        return _tree_product_reduction(self.tree, self.root_id, verbose=verbose)

    def leaf_groups(self) ->List[str]:
        """
        Get a list of all leaf group names

        Returns
        -------
        List[str]
            List of all leaf node names
        """
        return [n.identifier for n in self.tree.leaves()]

    def set_leaf_group_sizes(self, group_sizes: 'Dict[str, int]', update_parent_sizes: 'bool'=True):
        """
        Set process group sizes for all leaf groups

        Parameters
        ----------
        group_sizes : Dict[str, int]
            Dictionary with a mapping of each leaf group name to its size
        update_parent_sizes : bool
            Update all parent group sizes based on the leaf group if True
            If False, only set the leaf group sizes.
        """
        for id, size in group_sizes.items():
            if not self.tree.contains(id):
                raise AssertionError(f'Process group {id} is not in this process group config')
            node = self.tree.get_node(id)
            if not node.is_leaf():
                raise AssertionError(f'Process group {id} is not a leaf group')
            node.data.size = size
        if update_parent_sizes:
            self.update_parent_sizes()


class MockDistributedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.manager = DistributedManager()
        self.alpha = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.group = 'model_parallel'

    def forward(self, x):
        return reduce_from_parallel_region(self.alpha * x, self.group)

    @staticmethod
    def get_process_group_config() ->ProcessGroupConfig:
        world = ProcessGroupNode('world')
        config = ProcessGroupConfig(world)
        config.add_node(ProcessGroupNode('model_parallel'), parent='world')
        config.add_node(ProcessGroupNode('data_parallel'), parent='world')
        return config


class MulX(torch.nn.Module):
    """Helper class that just multiplies the values of an input tensor"""

    def __init__(self, multiplier: 'int'=1):
        super(MulX, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * self.multiplier


class MockModel(Module):

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)

    def forward(self, x):
        return self.layer(x)


class CustomModel(torch.nn.Module):
    """Custom User Model"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AFNOMlp,
     lambda: ([], {'in_features': 4, 'latent_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CappedGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CappedLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CellAreaWeightedLossFunction,
     lambda: ([], {'area': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Conv1dFCLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dFCLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3dFCLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvGRUBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (ConvNdFCLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNdKernel1Layer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvolutionalBlock3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CustomModel,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DGMLayer,
     lambda: ([], {'in_features_1': 4, 'in_features_2': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FCLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierFilter,
     lambda: ([], {'in_features': 4, 'layer_size': 1, 'nr_layers': 1, 'input_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaborFilter,
     lambda: ([], {'in_features': 4, 'layer_size': 1, 'nr_layers': 1, 'input_scale': 1.0, 'alpha': 4, 'beta': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeometricL2Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 720, 4]), torch.rand([4, 4, 720, 4])], {})),
    (GraphCastDecoderEmbedder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HEALPixFoldFaces,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Interpolate,
     lambda: ([], {'scale_factor': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'n_input': 4, 'n_hidden': 4, 'n_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MeshGraphMLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MiniNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModAFNOMlp,
     lambda: ([], {'in_features': 4, 'latent_features': 4, 'out_features': 4, 'mod_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MulX,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OneHotEmbedding,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed2D,
     lambda: ([], {'img_size': [4, 4], 'patch_size': [4, 4], 'in_chans': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed3D,
     lambda: ([], {'img_size': [4, 4, 4], 'patch_size': [4, 4, 4], 'in_chans': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Pool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (R2Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RRMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ResidualLinearBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ScaleShiftMlp,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinusoidalEncoding,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SirenLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SquarePlus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransposedConvUpsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (TruncatedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (WeightFactLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WeightNormLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_GraphCastWrapper,
     lambda: ([], {'model': torch.nn.ReLU(), 'dtype': torch.float32}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

