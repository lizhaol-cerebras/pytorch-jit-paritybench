
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


from collections import OrderedDict


from functools import partial


from inspect import signature


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import copy


import random


import warnings


import torch.nn as nn


import functools


import torch.nn.functional as F


from torch import nn


from time import sleep


from typing import Union


from typing import List


from typing import Tuple


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torch import distributed as dist


from torch.nn import functional as F


from torch.distributed import barrier


import math


import torch.distributed as dist


from torchvision.models import inception_v3


from scipy import linalg


from collections import namedtuple


import torchvision.models as tv


from scipy.sparse import lil_matrix


from scipy.sparse import diags


from scipy.sparse import eye


from torchvision.models import inception


from torchvision.models import vgg16


import torch.hub


from types import SimpleNamespace


import re


from torch.nn import Upsample as NearestUpsample


import types


from torchvision import transforms


from torch.utils.checkpoint import checkpoint


import collections


from torch.nn.utils import spectral_norm


from torch.nn.utils import weight_norm


from torch.nn.utils.spectral_norm import SpectralNorm


from torch.nn.utils.spectral_norm import SpectralNormStateDictHook


from torch.nn.utils.spectral_norm import SpectralNormLoadStateDictPreHook


import torchvision


import matplotlib.pyplot as plt


import time


from scipy import ndimage


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from sklearn.cluster import KMeans


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.autograd import Variable


from torch.nn.modules.module import Module


from torch.nn import init


from inspect import isclass


import inspect


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import torch.backends.cudnn as cudnn


import torchvision.utils


from torch.utils.tensorboard import SummaryWriter


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim import lr_scheduler


from scipy.optimize import curve_fit


from scipy.signal import medfilt


import torch.autograd.profiler as profiler


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    """Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * gain ** (f.ndim / 2)
    f = f
    return f


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    assert fw >= 1 and fh >= 1
    return fw, fh


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


_upfirdn2d_cuda_cache = dict()


def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    key = upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]


    class Upfirdn2dCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, f):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
                y = upfirdn2d_cuda.upfirdn2d_cuda(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, np.sqrt(gain))
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [fw - padx0 - 1, iw * upx - ow * downx + padx0 - upx + 1, fh - pady0 - 1, ih * upy - oh * downy + pady0 - upy + 1]
            dx = None
            df = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=not flip_filter, gain=gain).apply(dy, f)
            assert not ctx.needs_input_grad[1]
            return dx, df
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda


def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0):x.shape[2] - max(-pady1, 0), max(-padx0, 0):x.shape[3] - max(-padx1, 0)]
    f = f * gain ** (f.ndim / 2)
    f = f
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = F.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = F.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = F.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    """Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda':
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


class BlurDownsample(nn.Module):

    def __init__(self, kernel=(1, 3, 3, 1), factor=2, padding_mode='zeros'):
        super().__init__()
        p = len(kernel)
        px0 = (p - factor + 1) // 2
        px1 = (p - factor) // 2
        py0 = (p - factor + 1) // 2
        py1 = (p - factor) // 2
        self.pad = [px0, px1, py0, py1]
        self.factor = factor
        self.register_buffer('kernel', setup_filter(kernel))
        self.kernel_1d = kernel
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, list(self.pad) * 2, mode=self.padding_mode)
            out = upfirdn2d(x, self.kernel, down=self.factor)
        else:
            out = upfirdn2d(x, self.kernel, down=self.factor, padding=self.pad)
        return out

    def extra_repr(self):
        s = 'kernel={kernel_1d}, padding_mode={padding_mode}, pad={pad}'
        return s.format(**self.__dict__)


class ApplyNoise(nn.Module):
    """Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.conditional = True

    def forward(self, x, *_args, noise=None, **_kwargs):
        """

        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()
        return x + self.scale * noise


class Blur(nn.Module):

    def __init__(self, kernel=(1, 3, 3, 1), pad=0, padding_mode='zeros'):
        super().__init__()
        self.register_buffer('kernel', setup_filter(kernel))
        self.kernel_1d = kernel
        self.padding_mode = padding_mode
        self.pad = pad

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, list(self.pad) * 2, mode=self.padding_mode)
            out = upfirdn2d(x, self.kernel)
        else:
            out = upfirdn2d(x, self.kernel, padding=self.pad)
        return out

    def extra_repr(self):
        s = 'kernel={kernel_1d}, padding_mode={padding_mode}, pad={pad}'
        return s.format(**self.__dict__)

