
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


import torch.hub


from copy import deepcopy


import math


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


import torch.distributed as dist


from typing import Union


import logging


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from torch.utils.data import default_collate


from torchvision.transforms.functional import to_tensor


from torchvision.io.image import ImageReadMode


from torchvision.io.image import decode_image


from typing import Callable


from typing import Tuple


from typing import cast


from torch import Tensor


from torchvision.io import read_image


from torchvision.datasets import VisionDataset


from torchvision.datasets.folder import IMG_EXTENSIONS


from torchvision.datasets.folder import default_loader


import warnings


from collections import OrderedDict


from torch import nn


import torch.nn.functional as F


from torchvision import transforms as T


from torchvision.io.image import read_image


from torchvision.io.image import write_png


from torchvision.transforms.functional import convert_image_dtype


import torch.nn as nn


from torchvision import models


from torchvision.models import VGG16_Weights


from collections import namedtuple


from typing import Sequence


from torch import device


from torch.nn import Module


from torch.nn.parallel import DistributedDataParallel


import numpy as np


from functools import partial


from itertools import repeat


import collections.abc


import random


from torch import nn as nn


from torch.nn import functional as F


from torch.nn.functional import scaled_dot_product_attention as slow_attn


from torch import distributed as tdist


import functools


import torch.distributed as tdist


import torch.multiprocessing as mp


from math import sqrt


from torch.nn import InstanceNorm2d


from torchvision.transforms.functional import pil_to_tensor


from torchvision.io.image import encode_png


from time import sleep


from typing import Type


from torchvision.transforms.functional import to_pil_image


from torch import distributed as dist


import abc


from enum import Enum


from functools import wraps


from collections import Counter


from torch.optim.optimizer import Optimizer


from torch.cuda.amp import GradScaler


import torch.nn.functional as tf


from torchvision.transforms import functional as F


from torch.distributions import Categorical


from torch.distributions import kl_divergence


from torchvision.transforms.functional import resize


from torchvision.transforms.functional import center_crop


from torchvision.models import inception_v3


from torch.utils.data import Dataset


from torch.utils.data.dataloader import DataLoader


class AlignedCrop(nn.Module):

    def __init__(self, base: 'int'=128):
        super().__init__()
        self._base = base

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        wCrop = w - w // self._base * self._base
        hCrop = h - h // self._base * self._base
        cropLeft = wCrop // 2
        cropRight = wCrop - cropLeft
        cropTop = hCrop // 2
        cropBottom = hCrop - cropTop
        if cropBottom == 0:
            cropBottom = -h
        if cropRight == 0:
            cropRight = -w
        x = x[..., cropTop:-cropBottom, cropLeft:-cropRight]
        return x


class AlignedPadding(nn.Module):

    def __init__(self, base: 'int'=128):
        super().__init__()
        self._base = base

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        wPadding = (w // self._base + 1) * self._base - w
        hPadding = (h // self._base + 1) * self._base - h
        wPadding = wPadding % self._base
        hPadding = hPadding % self._base
        padLeft = wPadding // 2
        padRight = wPadding - padLeft
        padTop = hPadding // 2
        padBottom = hPadding - padTop
        return F.pad(x, (padLeft, padRight, padTop, padBottom), 'reflect')


class Distortion(nn.Module):

    def __init__(self, formatter: 'nn.Module'):
        super().__init__()
        self._formatter = formatter

    def formatDistortion(self, loss):
        return self._formatter(loss)


class Rate(nn.Module):

    def __init__(self, formatter: 'nn.Module'):
        super().__init__()
        self._formatter = formatter

    def formatRate(self, loss):
        return self._formatter(loss)


class BasicRate(Rate):

    def __init__(self, gamma: 'float'=0.0):
        super().__init__(nn.Identity())
        self._gamma = gamma

    def _cosineLoss(self, codebook):
        losses = list()
        for c in codebook:
            pairwise = c @ c.T
            norm = (c ** 2).sum(-1)
            cos = pairwise / (norm[:, None] * norm).sqrt()
            losses.append(cos.triu(1).clamp(0.0, 2).sum())
        return sum(losses)

    def forward(self, logits, codebooks, *_):
        return self._gamma * sum(self._cosineLoss(codebook) for codebook in codebooks)


_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


def _fspecial_gauss_1d(size, sigma, device=None):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, device=device).float()
    coords -= size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(x, win):
    """ Blur x with 1-D kernel
    Args:
        x (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all(ws == 1 for ws in win.shape[1:-1]), win.shape
    if len(x.shape) == 4:
        conv = F.conv2d
    elif len(x.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(x.shape)
    C = x.shape[1]
    out = x
    for i, s in enumerate(x.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(f'Skipping Gaussian Smoothing at dimension 2+{i} for input: {x.shape} and win size: {win.shape[-1]}')
    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = _gaussian_filter(X, win)
    mu2 = _gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (_gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ms_ssim(X, Y, win, weights, poolMethod, data_range=255, sizeAverage=True, K=(0.01, 0.03)):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (Tensor, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    assert X.shape == Y.shape, 'Input images should have the same dimensions.'
    win_size = win.shape[-1]
    assert win_size % 2 == 1, 'Window size should be odd.'
    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * 2 ** 4, 'Image size should be larger than %d due to the 4 downsamplings in ms-ssim' % ((win_size - 1) * 2 ** 4)
    levels = weights.shape[0]
    mcs = []
    ssim_per_channel = None
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [(s % 2) for s in X.shape[2:]]
            X = poolMethod(X, kernel_size=2, padding=padding)
            Y = poolMethod(Y, kernel_size=2, padding=padding)
    if ssim_per_channel is None:
        raise ValueError()
    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=1)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(1, -1, 1), dim=1)
    if sizeAverage:
        return ms_ssim_val.mean()
    return ms_ssim_val.mean(1)


class MsSSIM(nn.Module):

    def __init__(self, shape=4, data_range=255, sizeAverage=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        """ class for ms-ssim
        Args:
            shape (int): 4 for NCHW, 5 for NCTHW
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super().__init__()
        if shape == 4:
            self._avg_pool = F.avg_pool2d
        elif shape == 5:
            self._avg_pool = F.avg_pool3d
        else:
            raise ValueError(f'Input shape should be 4-d or 5-d tensors, but got {shape}')
        self.register_buffer('win', _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims), persistent=False)
        self.sizeAverage = sizeAverage
        self.data_range = data_range
        if weights is None:
            weights = torch.tensor(_WEIGHTS)
        else:
            weights = torch.tensor(weights)
        self.register_buffer('weights', weights, persistent=False)
        self.K = K

    def forward(self, X, Y):
        return 1.0 - ms_ssim(X, Y, self.win, self.weights, self._avg_pool, data_range=self.data_range, sizeAverage=self.sizeAverage, K=self.K)


class PSNR(nn.Module):

    def __init__(self, sizeAverage: 'bool'=False, upperBound: 'float'=255.0):
        super().__init__()
        self.register_buffer('_upperBound', torch.tensor(upperBound ** 2))
        self._average = sizeAverage

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor'):
        mse = ((x.double() - y.double()) ** 2).mean(dim=(1, 2, 3))
        res = 10.0 * (self._upperBound / (mse + 0.0001)).log10()
        return res.mean() if self._average else res


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        ckpt_path = Path(__file__).parent / 'lpips_vgg.pth'
        self.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=False)
        logging.info('loaded pretrained LPIPS loss from {}'.format(ckpt_path))

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = F.mse_loss(feats0[kk], feats1[kk], reduction='none')
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class Compound(Module):

    def __init__(self, compressor: 'BaseCompressor', distortion: 'Distortion', lpips):
        super().__init__()
        self._compressor = compressor
        self._distortion = distortion
        self._lpips = lpips
        self._lpips.eval()
        for param in self._lpips.parameters():
            param.requires_grad = False

    def train(self, mode: 'bool'=True):
        retValue = super().train(mode)
        self._lpips.eval()
        self._distortion.eval()
        return retValue

    def forward(self, x: 'Tensor'):
        xHat, yHat, codes, logits = self._compressor(x)
        distortion = self._distortion(xHat, x, codes, logits)
        xHatSmall = F.interpolate(xHat, (224, 224), mode='bilinear')
        xSmall = F.interpolate(x, (224, 224), mode='bilinear')
        lpips = self._lpips(xHatSmall, xSmall)
        mse = F.mse_loss(xHat, x)
        return xHat, (distortion, mse, lpips.mean()), codes, logits

    @property
    def Freq(self):
        return self._compressor._quantizer._entropyCoder.NormalizedFreq

    @property
    def Compressor(self):
        return self._compressor

    def refresh(self, rank: 'int') ->torch.Tensor:
        if rank == 0:
            proportion = self.Compressor.reAssignCodebook()
        else:
            proportion = torch.zeros(())
        self.Compressor.syncCodebook()
        return proportion

    def formatDistortion(self, loss: 'torch.Tensor'):
        return self._distortion.formatDistortion(loss)


class ConstsMetaClass(type):

    @property
    def TempDir(cls):
        if getattr(cls, '_tempDir', None) is None:
            tempDir = os.path.dirname(tempfile.mktemp())
            tempDir = os.path.join(tempDir, 'mcquic')
            cls._tempDir = tempDir
            os.makedirs(cls._tempDir, exist_ok=True)

            def removeTmp():
                shutil.rmtree(tempDir, ignore_errors=True)
            atexit.register(removeTmp)
        return cls._tempDir


class Consts(metaclass=ConstsMetaClass):
    Name = 'mcquic'
    Eps = 1e-06
    CDot = '·'
    TimeOut = 15


def versionCheck(versionStr: 'str'):
    version = StrictVersion(versionStr)
    builtInVersion = StrictVersion(mcquic.__version__)
    if builtInVersion < version:
        raise ValueError(f"Version too new. Given {version}, but I'm {builtInVersion} now.")
    major, minor, revision = version.version
    bMajor, bMinor, bRev = builtInVersion.version
    if major != bMajor:
        raise ValueError(f"Major version mismatch. Given {version}, but I'm {builtInVersion} now.")
    if minor != bMinor:
        warnings.warn(f"Minor version mismatch. Given {version}, but I'm {builtInVersion} now.")
    return True


class _residulBlock(nn.Module):

    def __init__(self, act1: 'nn.Module', conv1: 'nn.Conv2d', act2: 'nn.Module', conv2: 'nn.Conv2d', skip: 'Union[nn.Module, None]'):
        super().__init__()
        self._branch = nn.Sequential(act1, conv1, act2, conv2)
        self._skip = skip

    def forward(self, x):
        identity = x
        out = self._branch(x)
        if self._skip is not None:
            identity = self._skip(x)
        out += identity
        return out


def conv1x1(inChannels: 'int', outChannels: 'int', stride: 'int'=1, bias: 'bool'=True, groups: 'int'=1) ->nn.Conv2d:
    """A wrapper of 1x1 convolution.

    Usage:
    ```python
        # A 1x1 conv with "same" feature map:
        conv = conv1x1(128, 128)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=1, stride=stride)


def conv3x3(inChannels: 'int', outChannels: 'int', stride: 'int'=1, bias: 'bool'=True, groups: 'int'=1) ->nn.Conv2d:
    """A wrapper of 3x3 convolution with pre-calculated padding.

    Usage:
    ```python
        # A 3x3 conv with "same" feature map:
        conv = conv3x3(128, 128)
        # A 3x3 conv with halved feature map:
        conv = conv3x3(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=3, stride=stride, padding=1)


class ResidualBlock(_residulBlock):
    """Basic residual block.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | Conv3s1   |  |
        | SiLU      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """

    def __init__(self, inChannels: 'int', outChannels: 'int', groups: 'int'=1, denseNorm: 'bool'=False):
        """Usage:
        ```python
            # A block with "same" feature map
            block = ResidualBlock(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            groups (int): Group convolution (default: 1).
        """
        if inChannels != outChannels:
            skip = conv1x1(inChannels, outChannels)
        else:
            skip = None
        super().__init__(nn.SiLU(), conv3x3(inChannels, outChannels), nn.GroupNorm(groups, outChannels) if denseNorm else nn.SiLU(), conv3x3(outChannels, outChannels), skip)


class AttentionBlock(nn.Module):
    """Self attention block.
    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Default structure:
    ```plain
        +----------------------+
        | Input ----┬--------╮ |
        | ResBlock  ResBlock | |
        | |         Sigmoid  | |
        | * <------ Mask     | |
        | Masked             | |
        | + <----------------╯ |
        | Output               |
        +----------------------+
    ```
    """

    def __init__(self, channel, groups=1, denseNorm: 'bool'=False):
        super().__init__()
        self._mainBranch = nn.Sequential(ResidualBlock(channel, channel, groups, denseNorm), ResidualBlock(channel, channel, groups, denseNorm), ResidualBlock(channel, channel, groups, denseNorm))
        self._sideBranch = nn.Sequential(ResidualBlock(channel, channel, groups, denseNorm), ResidualBlock(channel, channel, groups, denseNorm), ResidualBlock(channel, channel, groups, denseNorm), conv1x1(channel, channel))

    def forward(self, x):
        identity = x
        a = self._mainBranch(x)
        b = self._sideBranch(x)
        mask = torch.sigmoid(b)
        out = a * mask
        out += identity
        return out


class _lowerBound(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: 'float'):
        """Lower bound operator.

        Args:
            bound (float): The lower bound.
        """
        super().__init__()
        self.register_buffer('bound', torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return _lowerBound.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.
    Used for stability during training.
    """

    def __init__(self, minimum: 'float'=0.0, eps: 'float'=Consts.Eps):
        """Non negative reparametrization.

        Args:
            minimum (float, optional): The lower bound. Defaults to 0.
            reparam_offset (float, optional): Eps for stable training. Defaults to 2**-18.
        """
        super().__init__()
        minimum = float(minimum)
        eps = float(eps)
        self.register_buffer('eps', torch.Tensor([eps ** 2]))
        bound = (minimum + eps ** 2) ** 0.5
        self.lowerBound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.eps, self.eps))

    def forward(self, x):
        out = self.lowerBound(x)
        out = out ** 2 - self.eps
        return out


class GenDivNorm(nn.Module):
    """Generalized Divisive Normalization layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \\frac{x[i]}{\\sqrt{\\beta[i] + \\sum_j(\\gamma[j, i] * x[j]^2)}}
    """

    def __init__(self, inChannels: 'int', groups: 'int'=1, biasBound: 'float'=0.0001, weightInit: 'float'=0.1, fuseNorm: 'bool'=False):
        """Generalized Divisive Normalization layer.

        Args:
            inChannels (int): Channels of input tensor.
            inverse (bool, optional): GDN or I-GDN. Defaults to False.
            beta_min (float, optional): Lower bound of beta. Defaults to 1e-4.
            gamma_init (float, optional): Initial value of gamma. Defaults to 0.1.
        """
        super().__init__()
        self._groups = groups
        biasBound = float(biasBound)
        weightInit = float(weightInit)
        self.beta_reparam = NonNegativeParametrizer(minimum=biasBound)
        beta = torch.ones(inChannels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = [(weightInit * torch.eye(inChannels // self._groups)) for _ in range(self._groups)]
        gamma = torch.cat(gamma, 0)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma[..., None, None]
        std = F.conv2d(x ** 2, gamma, beta, groups=self._groups)
        return self._normalize(x, std)

    def _normalize(self, x: 'torch.Tensor', std: 'torch.Tensor') ->torch.Tensor:
        return x * torch.rsqrt(std)


class InvGenDivNorm(GenDivNorm):
    """I-GDN layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \\frac{x[i]}{\\sqrt{\\beta[i] + \\sum_j(\\gamma[j, i] * x[j]^2)}}
    """

    def _normalize(self, x: 'torch.Tensor', std: 'torch.Tensor') ->torch.Tensor:
        return x * torch.sqrt(std)


def pixelShuffle3x3(inChannels: 'int', outChannels: 'int', r: 'float'=1, groups: 'int'=1) ->nn.Conv2d:
    """A wrapper of 3x3 convolution and a 2x down-sampling by `PixelShuffle`.

    Usage:
    ```python
        # A 2x down-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 0.5)
        # A 2x up-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    if r < 1:
        r = int(1 / r)
        return nn.Sequential(nn.Conv2d(inChannels, outChannels // r ** 2, kernel_size=3, padding=1, groups=groups), nn.PixelUnshuffle(r))
    else:
        r = int(r)
        return nn.Sequential(nn.Conv2d(inChannels, outChannels * r ** 2, kernel_size=3, padding=1, groups=groups), nn.PixelShuffle(r))


class ResidualBlockShuffle(_residulBlock):
    """Residual block with PixelShuffle for up-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | PixShuf3  |  |
        | IGDN      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """

    def __init__(self, inChannels: 'int', outChannels: 'int', upsample: 'int'=2, groups: 'int'=1, denseNorm: 'bool'=False):
        """Usage:
        ```python
            # A block performs 2x up-sampling
            block = ResidualBlockShuffle(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            upsample (int): Up-sampling rate (default: 2).
            groups (int): Group convolution (default: 1).
        """
        super().__init__(nn.SiLU(), pixelShuffle3x3(inChannels, outChannels, upsample), InvGenDivNorm(outChannels), conv3x3(outChannels, outChannels), pixelShuffle3x3(inChannels, outChannels, upsample))


class ResidualBlockWithStride(_residulBlock):
    """Residual block with stride for down-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | Conv3s2   |  |
        | GDN       |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """

    def __init__(self, inChannels: 'int', outChannels: 'int', stride: 'int'=2, groups: 'int'=1, denseNorm: 'bool'=False):
        """Usage:
        ```python
            # A block performs 2x down-sampling
            block = ResidualBlockWithStride(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            stride (int): stride value (default: 2).
            groups (int): Group convolution (default: 1).
        """
        if stride != 1:
            skip = conv3x3(inChannels, outChannels, stride=stride)
        elif inChannels != outChannels:
            skip = conv1x1(inChannels, outChannels, stride=stride)
        else:
            skip = None
        super().__init__(nn.SiLU(), conv3x3(inChannels, outChannels, stride=stride), GenDivNorm(outChannels), conv3x3(outChannels, outChannels), skip)


class _multiCodebookDeQuantization(nn.Module):

    def __init__(self, codebook: 'nn.Parameter'):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self.register_buffer('_ix', torch.arange(self._m), persistent=False)

    def decode(self, code: 'torch.Tensor'):
        n, _, h, w = code.shape
        code = code.permute(0, 2, 3, 1).contiguous()
        ix = self._ix.expand_as(code)
        indexed = self._codebook[ix, code]
        return indexed.reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, sample: 'torch.Tensor'):
        n, _, h, w, _ = sample.shape
        left = sample.reshape(n * self._m, h * w, self._k).contiguous()
        right = self._codebook.expand(n, self._m, self._k, self._d).reshape(n * self._m, self._k, self._d).contiguous()
        result = torch.bmm(left, right)
        return result.reshape(n, self._m, h, w, self._d).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()


def gumbelSoftmax(logits: 'torch.Tensor', temperature: 'float'=1.0, hard: 'bool'=True, dim: 'int'=-1):
    eps = torch.finfo(logits.dtype).eps
    uniforms = torch.rand_like(logits).clamp_(eps, 1 - eps)
    gumbels = -(-uniforms.log()).log()
    y_soft = ((logits + gumbels) / temperature).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0).contiguous()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class _multiCodebookQuantization(nn.Module):

    def __init__(self, codebook: 'nn.Parameter', freqEMA: 'nn.Parameter'):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self._bits = math.log2(self._k)
        self._scale = math.sqrt(self._k)
        self._temperature = nn.Parameter(torch.ones((self._m, 1, 1, 1)))
        self._bound = LowerBound(Consts.Eps)
        self._freqEMA = freqEMA

    def reAssignCodebook(self, freq: 'torch.Tensor') ->torch.Tensor:
        codebook = self._codebook.detach().clone()
        freq = freq.detach().clone()
        for m, (codebookGroup, freqGroup) in enumerate(zip(self._codebook, freq)):
            neverAssignedLoc = freqGroup < Consts.Eps
            totalNeverAssigned = int(neverAssignedLoc.sum())
            if totalNeverAssigned > self._k // 2:
                mask = torch.zeros((totalNeverAssigned,), device=self._codebook.device)
                maskIdx = torch.randperm(len(mask))[self._k // 2:]
                mask[maskIdx] = -1.0
                freqGroup[neverAssignedLoc] = mask
                neverAssignedLoc = (freqGroup < Consts.Eps) * (freqGroup > -Consts.Eps)
                totalNeverAssigned = int(neverAssignedLoc.sum())
            argIdx = torch.argsort(freqGroup, descending=True)
            mostAssigned = codebookGroup[argIdx]
            codebook.data[m, neverAssignedLoc] = mostAssigned[:totalNeverAssigned]
        diff = ((codebook - self._codebook) ** 2).sum(-1) > 0.0001
        proportion = diff.flatten()
        self._codebook.data.copy_(codebook)
        return proportion

    def syncCodebook(self):
        codebook = self._codebook.detach().clone()
        dist.broadcast(codebook, 0)
        self._codebook.data.copy_(codebook)

    def encode(self, x: 'torch.Tensor'):
        distance = self._distance(x)
        code = distance.argmin(-1)
        return code

    def _distance(self, x: 'torch.Tensor') ->torch.Tensor:
        n, _, h, w = x.shape
        x = x.reshape(n, self._m, self._d, h, w).contiguous()
        x2 = (x ** 2).sum(2, keepdim=True)
        c2 = (self._codebook ** 2).sum(-1, keepdim=True)[..., None].contiguous()
        left = x.reshape(n * self._m, self._d, h * w).permute(0, 2, 1).contiguous()
        right = self._codebook.expand(n, self._m, self._k, self._d).reshape(n * self._m, self._k, self._d).permute(0, 2, 1).contiguous()
        inter = torch.bmm(left, right)
        inter = inter.reshape(n, self._m, h, w, self._k).permute(0, 1, 4, 2, 3).contiguous()
        distance = x2 + c2 - 2 * inter
        return distance.permute(0, 1, 3, 4, 2).contiguous()

    def _logit(self, x: 'torch.Tensor') ->torch.Tensor:
        logit = -1 * self._distance(x)
        return logit / self._scale

    def _randomDrop(self, logit):
        codeUsage = (self._freqEMA > Consts.Eps).float().mean().clamp(0.0, 1.0)
        randomMask = torch.rand_like(logit) ** (-(self._bits - 1) * codeUsage ** 2 + self._bits) < self._freqEMA[:, None, None, ...]
        logit[randomMask] += -1000000000.0
        return logit

    def _sample(self, x: 'torch.Tensor', temperature: 'float'):
        logit = self._logit(x) * self._bound(self._temperature)
        logit = self._randomDrop(logit)
        sampled = gumbelSoftmax(logit, temperature, True)
        return sampled, logit

    def forward(self, x: 'torch.Tensor'):
        sample, logit = self._sample(x, 1.0)
        code = logit.argmax(-1, keepdim=True)
        oneHot = torch.zeros_like(logit).scatter_(-1, code, 1).contiguous()
        return sample, code[..., 0].contiguous(), oneHot, logit


class _quantizerDecoder(nn.Module):
    """
    Default structure:
    ```plain
        q [H/2, W/2]            formerLevelRestored [H/2, W/2]
        | `dequantizaitonHead`  | `sideHead`
        ├-`add` ----------------╯
        xHat [H/2, W/2]
        | `restoreHead`
        nextLevelRestored [H, W]
    ```
    """

    def __init__(self, dequantizer: '_multiCodebookDeQuantization', dequantizationHead: 'nn.Module', sideHead: 'Union[None, nn.Module]', restoreHead: 'nn.Module'):
        super().__init__()
        self._dequantizer = dequantizer
        self._dequantizationHead = dequantizationHead
        self._sideHead = sideHead
        self._restoreHead = restoreHead

    def decode(self, code: 'torch.Tensor', formerLevel: 'Union[None, torch.Tensor]'):
        q = self._dequantizationHead(self._dequantizer.decode(code))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)

    def forward(self, q: 'torch.Tensor', formerLevel: 'Union[None, torch.Tensor]'):
        q = self._dequantizationHead(self._dequantizer(q))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)


class _quantizerEncoder(nn.Module):
    """
    Default structure:
    ```plain
        x [H, W]
        | `latentStageEncoder`
        z [H/2 , W/2] -------╮
        | `quantizationHead` | `latentHead`
        q [H/2, W/2]         z [H/2, w/2]
        |                    |
        ├-`subtract` --------╯
        residual for next level
    ```
    """

    def __init__(self, quantizer: '_multiCodebookQuantization', dequantizer: '_multiCodebookDeQuantization', latentStageEncoder: 'nn.Module', quantizationHead: 'nn.Module', latentHead: 'Union[None, nn.Module]'):
        super().__init__()
        self._quantizer = quantizer
        self._dequantizer = dequantizer
        self._latentStageEncoder = latentStageEncoder
        self._quantizationHead = quantizationHead
        self._latentHead = latentHead

    @property
    def Codebook(self):
        return self._quantizer._codebook

    def syncCodebook(self):
        self._quantizer.syncCodebook()

    def reAssignCodebook(self, freq: 'torch.Tensor') ->torch.Tensor:
        return self._quantizer.reAssignCodebook(freq)

    def encode(self, x: 'torch.Tensor'):
        z = self._latentStageEncoder(x)
        code = self._quantizer.encode(self._quantizationHead(z))
        if self._latentHead is None:
            return None, code
        z = self._latentHead(z)
        return z - self._dequantizer.decode(code), code

    def forward(self, x: 'torch.Tensor'):
        z = self._latentStageEncoder(x)
        q, code, oneHot, logit = self._quantizer(self._quantizationHead(z))
        if self._latentHead is None:
            return q, None, code, oneHot, logit
        z = self._latentHead(z)
        return q, z - self._dequantizer(q), code, oneHot, logit


def pixelShuffle5x5(inChannels: 'int', outChannels: 'int', r: 'float'=1) ->nn.Conv2d:
    """A wrapper of 5x5 convolution and a 2x up-sampling by `PixelShuffle`.

    Usage:
    ```python
        # A 2x up-sampling with a 5x5 conv:
        conv = pixelShuffleConv5x5(128, 128, 2)
        # A 2x down-sampling with a 5x5 conv:
        conv = pixelShuffleConv5x5(128, 128, 0.5)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    if r < 1:
        r = int(1 / r)
        return nn.Sequential(nn.Conv2d(inChannels, outChannels // r ** 2, kernel_size=5, stride=1, padding=5 // 2), nn.PixelUnshuffle(r))
    else:
        r = int(r ** 2)
        return nn.Sequential(nn.Conv2d(inChannels, outChannels * r, kernel_size=5, stride=1, padding=5 // 2), nn.PixelShuffle(r))


class BaseDecoder5x5(nn.Module):

    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(pixelShuffle5x5(channel, channel, 2), InvGenDivNorm(channel), pixelShuffle5x5(channel, channel, 2), InvGenDivNorm(channel), pixelShuffle5x5(channel, 3, 2))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class ResidualBaseDecoder(nn.Module):

    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(ResidualBlock(channel, channel, groups=groups), ResidualBlockShuffle(channel, channel, 2, groups=groups), AttentionBlock(channel, groups=groups), ResidualBlock(channel, channel, groups=groups), ResidualBlockShuffle(channel, channel, 2, groups=groups), ResidualBlock(channel, channel, groups=groups), pixelShuffle3x3(channel, 3, 2))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class UpSampler(nn.Module):

    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(AttentionBlock(channel, groups=groups), ResidualBlock(channel, channel, groups=groups), ResidualBlockShuffle(channel, channel, 2, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class UpSampler5x5(nn.Module):

    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(pixelShuffle5x5(channel, channel, 2), GenDivNorm(channel))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


def conv5x5(inChannels: 'int', outChannels: 'int', stride: 'int'=1, bias: 'bool'=True, groups: 'int'=1) ->nn.Conv2d:
    """A wrapper of 5x5 convolution with pre-calculated padding.

    Usage:
    ```python
        # A 5x5 conv with "same" feature map:
        conv = conv5x5(128, 128)
        # A 5x5 conv with halved feature map:
        conv = conv5x5(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=5, stride=stride, padding=5 // 2)


class BaseEncoder5x5(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(conv5x5(3, channel, groups=groups), GenDivNorm(channel), conv5x5(channel, channel, groups=groups), GenDivNorm(channel), conv5x5(channel, channel, groups=groups), GenDivNorm(channel))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class DownSampler5x5(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(conv5x5(channel, channel, groups=groups), GenDivNorm(channel))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class EncoderHead5x5(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(conv5x5(channel, channel, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class Director5x5(nn.Module):

    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(conv5x5(channel, channel, 1, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class ResidualBaseEncoder(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(ResidualBlockUnShuffle(3, channel), ResidualBlock(channel, channel, groups=groups), ResidualBlockUnShuffle(channel, channel, groups=groups), AttentionBlock(channel, groups=groups), ResidualBlock(channel, channel, groups=groups), ResidualBlockUnShuffle(channel, channel, groups=groups))
        else:
            self._net = nn.Sequential(ResidualBlockWithStride(3, channel, stride=2), ResidualBlock(channel, channel, groups=groups), ResidualBlockWithStride(channel, channel, stride=2, groups=groups), AttentionBlock(channel, groups=groups), ResidualBlock(channel, channel, groups=groups), ResidualBlockWithStride(channel, channel, stride=2, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class DownSampler(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(ResidualBlock(channel, channel, groups=groups), ResidualBlockUnShuffle(channel, channel, groups=groups))
        else:
            self._net = nn.Sequential(ResidualBlock(channel, channel, groups=groups), ResidualBlockWithStride(channel, channel, stride=2, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class EncoderHead(nn.Module):

    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(ResidualBlock(channel, channel, groups=groups), pixelShuffle3x3(channel, channel, 2), AttentionBlock(channel, groups=groups))
        else:
            self._net = nn.Sequential(ResidualBlock(channel, channel, groups=groups), conv3x3(channel, channel, stride=2, groups=groups), AttentionBlock(channel, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class Director(nn.Module):

    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(AttentionBlock(channel, groups=groups), ResidualBlock(channel, channel, groups=groups), conv3x3(channel, channel, stride=1, groups=groups))

    def forward(self, x: 'torch.Tensor'):
        return self._net(x)


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `maskType='A'` for the
    first layer (which also masks the "current pixel"), `maskType='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, maskType: str='A', **kwargs: Any):
        """Masked Conv 2D.

        Args:
            args: Positional arguments for `nn.Conv2d`.
            maskType (str, optional): Mask type, if "A", current pixel will be masked, otherwise "B". Use "A" for first layer and "B" for successive layers. Defaults to "A".
            kwargs: Keyword arguments for `nn.Conv2d`.

        Usage:
        ```python
            # First layer
            conv = MaskedConv2d(3, 6, 3, maskType='A')
            # Subsequent layers
            conv = MaskedConv2d(6, 6, 3, maskType='B')
        ```

        Raises:
            ValueError: Mask type not in ["A", "B"].
        """
        super().__init__(*args, **kwargs)
        if maskType not in ('A', 'B'):
            raise ValueError(f'Invalid `maskType` value "{maskType}"')
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.shape
        self.mask[:, :, h // 2, w // 2 + (maskType == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: 'Tensor') ->Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlockMasked(_residulBlock):
    """A residual block with MaskedConv for causal inference.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | MConv5s1A |  |
        | LReLU     |  |
        | MConv5s1B |  |
        | LReLU     |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """

    def __init__(self, inChannels, outChannels, maskType: 'str'='A'):
        """Usage:
        ```python
            # First block
            block = ResidualBlockMasked(128, 128, "A")
            # Subsequent blocks
            block = ResidualBlockMasked(128, 128, "B")
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            maskType (str): Mask type of MaskedConv2D (default: "A").
        """
        if inChannels != outChannels:
            skip = MaskedConv2d(inChannels, outChannels, maskType=maskType, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode='zeros')
        else:
            skip = None
        super().__init__(nn.ReLU(), MaskedConv2d(inChannels, outChannels, maskType=maskType, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode='zeros'), nn.ReLU(), MaskedConv2d(outChannels, outChannels, maskType='B', bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode='zeros'), skip)


class PixelCNN(nn.Module):

    def __init__(self, m: 'int', k: 'int', channel: 'int'):
        super().__init__()
        self._net = nn.Sequential(ResidualBlockMasked(channel, channel), ResidualBlockMasked(channel, channel), MaskedConv2d(channel, m * k, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode='zeros'))
        self._m = m
        self._k = k

    def forward(self, x: 'torch.Tensor'):
        n, c, h, w = x.shape
        logits = self._net(x).reshape(n, self._k, self._m, h, w)
        return logits


class WholePQBig(nn.Module):

    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super().__init__()
        self._k = k
        self._compressor = PQCompressorBig(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = CompressionLossBig(target)
        self._spreadLoss = CodebookSpreading()

    def forward(self, image, temp, **_):
        restored, allHards, latent, allCodes, allTrues, allLogits, (allFeatures, allQuantizeds), allCodebooks = self._compressor(image, temp, True)
        dLoss = self._cLoss(image, restored)
        weakCodebookLoss = list()
        for raws, codes, codebooks, k, logits in zip(allFeatures, allCodes, allCodebooks, self._k, allLogits):
            for raw, code, codebook, logit in zip(raws, codes, codebooks, logits):
                weakCodebookLoss.append(self._spreadLoss(codebook, temp))
        return dLoss, (sum(weakCodebookLoss), 0.0, 0.0), (restored, allTrues, allLogits)


class WholePQPixelCNN(nn.Module):

    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super().__init__()
        self._levels = len(k)
        self._compressor = PQCompressorBig(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = nn.CrossEntropyLoss()
        self._pixelCNN = nn.ModuleList(PixelCNN(m, ki, channel) for ki in k)

    def test(self, image):
        restored, allCodes, allHards = self._compressor.test(image)
        for i, (pixelCNN, hard, code) in enumerate(zip(self._pixelCNN, allHards, allCodes)):
            logits = pixelCNN(hard)
            correct = logits.argmax(1) == code
            code[correct] = -1
            code += 1
        return restored, allCodes

    def forward(self, image, temp, **_):
        with torch.no_grad():
            allZs, allHards, allCodes, allResiduals = self._compressor.getLatents(image)
        predictLoss = list()
        ratios = list()
        for i, (pixelCNN, hard, code) in enumerate(zip(self._pixelCNN, allHards, allCodes)):
            n, m, c, h, w = hard.shape
            logits = pixelCNN(hard.reshape(n, m * c, h, w))
            dLoss = self._cLoss(logits, code)
            predictLoss.append(dLoss)
            ratios.append((logits.argmax(1) == code).float().mean())
        return sum(predictLoss), sum(ratios) / len(ratios)


class WholePQ5x5(WholePQBig):

    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super(WholePQBig, self).__init__()
        self._compressor = PQCompressor5x5(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = CompressionLossBig(target)
        self._auxLoss = L1L2Loss()


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, prediction_num, prototypes=None):
        super().__init__()
        if prototypes is not None:
            self.register_buffer('prototypes', prototypes)
            self.prototype_proj = nn.Linear(prototypes.shape[-1], hidden_size)
            self.norm_final = nn.InstanceNorm2d(hidden_size, affine=True, eps=1e-06)
            self.linear = nn.Conv2d(hidden_size, hidden_size, 1)
            self.skip_connection = nn.Conv2d(hidden_size, hidden_size, 1)
        else:
            self.norm_final = nn.InstanceNorm2d(hidden_size, affine=True, eps=1e-06)
            self.linear = nn.Conv2d(hidden_size, prediction_num, 1)
            self.skip_connection = nn.Conv2d(hidden_size, prediction_num, 1)

    def forward(self, x):
        if hasattr(self, 'prototypes'):
            out = self.norm_final(x)
            x = self.linear(out) + self.skip_connection(x)
            n, c, h, w = x.shape
            codes = self.prototype_proj(self.prototypes)
            similarity = torch.bmm(x.permute(0, 2, 3, 1).reshape(n, h * w, c).contiguous(), codes.expand(n, *codes.shape).permute(0, 2, 1).contiguous())
            similarity = similarity.reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous()
            return similarity
        else:
            x = self.linear(self.norm_final(x)) + self.skip_connection(x)
            return x


class ProjLayer(nn.Module):
    """
    The prjection layer of Var
    Upsample hidden_size -> 4 * hidden_size
    """

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-06)
        self.proj = nn.Linear(hidden_size, out_dim, bias=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.0, use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-06)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=proj_drop)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class AnyResolutionBlock(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """

    def __init__(self, codebook: 'torch.Tensor', canvas_size, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        input_dim = codebook.shape[-1]
        self.in_channels = input_dim
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = nn.Conv2d(input_dim, hidden_size, 1, bias=False)
        self.pre_layer = checkpoint_wrapper(FinalLayer(hidden_size, hidden_size))
        self.final_layer = checkpoint_wrapper(FinalLayer(hidden_size, len(codebook), None))
        self.num_patches = canvas_size * canvas_size * 64
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([checkpoint_wrapper(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)])
        self.proj_layer = checkpoint_wrapper(ProjLayer(hidden_size, hidden_size))
        self._initialize_weights()

    def _initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[random_up:random_up + h, random_left:random_left + w].reshape(h * w, -1)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start + h, left_start:left_start + w].reshape(h * w, -1)

    def unpatchify(self, x, h, w):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, hw, dim = x.shape
        return x.permute(0, 2, 1).reshape(bs, dim, h, w).contiguous()
        return self.pixel_shuffle(x)

    def forward(self, input_embedding):
        bs, c, h, w = input_embedding.shape
        x = self.input_transform(input_embedding)
        x = self.pre_layer(x).permute(0, 2, 3, 1).reshape(bs, h * w, -1).contiguous()
        if self.training:
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)
        x = x + selected_pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.proj_layer(x)
        x = self.unpatchify(x, h, w)
        prediction = self.final_layer(x)
        return prediction


class AnyResolutionTransformer(nn.Module):

    def __init__(self, text_dimension, canvas_size: 'List[int]', codebooks, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.text_dimension = text_dimension
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList([AnyResolutionBlock(codebook, can, hidden_size, depth, num_heads, mlp_ratio) for can, codebook in zip(canvas_size, codebooks)])

    def forward(self, all_forwards_for_residual, pooled_condition, sequence_condition, attention_mask):
        if self.training:
            if not isinstance(all_forwards_for_residual, list):
                raise RuntimeError('The given training input is not a list.')
            results = list()
            for current, block in zip(all_forwards_for_residual, self.blocks):
                results.append(block(current))
            return results
        else:
            current, level = all_forwards_for_residual
            results = list()
            block = self.blocks[level]
            predict = block(current)
            return predict.argmax(1)


class CrossTransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=proj_drop)

    def forward(self, x, condition, attention_mask):
        x = x + self.attn(x, condition, condition, key_padding_mask=attention_mask != 1, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TextConditionedGenerator(nn.Module):

    def __init__(self, text_dimension: 'int', codebook, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.text_dimension = text_dimension
        self.canvas_size = 16
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_lift = nn.Linear(text_dimension, hidden_size, bias=False)
        self.pre_layer = checkpoint_wrapper(FinalLayer(hidden_size, hidden_size))
        self.post_layer = checkpoint_wrapper(ProjLayer(hidden_size, hidden_size))
        self.final_layer = checkpoint_wrapper(FinalLayer(hidden_size, len(codebook), None))
        self.num_patches = self.canvas_size * self.canvas_size
        self.pos_embed = nn.Parameter(nn.init.uniform_(torch.empty(1, self.num_patches, hidden_size), a=-math.sqrt(2 / (5 * hidden_size)), b=math.sqrt(2 / (5 * hidden_size))))
        self.blocks = nn.ModuleList([checkpoint_wrapper(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)])
        self.condition_blocks = nn.ModuleList([checkpoint_wrapper(CrossTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)])
        self._initialize_weights()

    def _initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[random_up:random_up + h, random_left:random_left + w].reshape(h * w, -1)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start + h, left_start:left_start + w].reshape(h * w, -1)

    def forward(self, target_shape, pooled_condition, sequence_condition, attention_mask):
        bs, h, w = target_shape
        pooled_condition = self.text_lift(pooled_condition)
        sequence_condition = self.text_lift(sequence_condition)
        if self.training:
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)
        x = selected_pos_embed.expand(bs, h * w, self.hidden_size)
        x = self.pre_layer(x.permute(0, 2, 1).reshape(bs, self.hidden_size, h, w).contiguous()).permute(0, 2, 3, 1).reshape(bs, h * w, self.hidden_size).contiguous()
        for block, cross in zip(self.blocks, self.condition_blocks):
            x = block(x) + cross(x, sequence_condition, attention_mask)
        x = self.post_layer(x)
        x = x.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()
        prediction = self.final_layer(x)
        return prediction if self.training else prediction.argmax(1)


def AnyRes_S(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, codebooks[0], hidden_size=288, depth=12, num_heads=6, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=12, hidden_size=288, num_heads=6, **kwargs)


class ForwardGenerator(nn.Module):

    def __init__(self, channel: 'int', k: 'List[int]', denseNorm: 'bool', loadFrom: 'str', *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, denseNorm)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items()})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        logging.debug('Start loading clip...')
        self.text_encoder = transformers.CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        self.text_tokenizer = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        for params in self.text_encoder.parameters():
            params.requires_grad_(False)
        clip_text_channels = self.text_encoder.text_model.config.hidden_size
        self.text_to_first_level, self.next_residual_predictor = AnyRes_S(clip_text_channels, [2, 4, 8, 16], [codebook.squeeze(0) for codebook in self.compressor.Codebooks])
        logging.debug('Created any-res transformer.')
        self.compressor.eval()
        self.text_encoder.eval()

    def train(self, mode: 'bool'=True):
        self.compressor.eval()
        self.text_encoder.eval()
        return super().train(mode)

    def forward(self, image, condition: 'List[str]'):
        if not isinstance(condition, list):
            raise NotImplementedError
        if self.training:
            with torch.no_grad():
                splitted = torch.split(image, 16)
                allCodes = list()
                all_forwards_for_residual = list()
                for sp in splitted:
                    formerLevel = None
                    codes = self.compressor.encode(sp)
                    allCodes.append(codes)
                    this_split_forward_residual = list()
                    formerLevel = None
                    for level, code in enumerate(codes[:-1]):
                        this_split_forward_residual.append(self.compressor.residual_forward(code, formerLevel, level))
                        formerLevel = this_split_forward_residual[-1]
                    all_forwards_for_residual.append(this_split_forward_residual)
                codes = [torch.cat(x) for x in zip(*allCodes)]
                all_forwards_for_residual = [torch.cat(x) for x in zip(*all_forwards_for_residual)]
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                input_ids = batch_encoding.input_ids
                attention_mask = batch_encoding.attention_mask
                text_embedding: 'transformers.modeling_outputs.BaseModelOutputWithPooling' = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            codes = [c.squeeze(1) for c in codes]
            first_level = self.text_to_first_level(codes[0].shape, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
            predictions = self.next_residual_predictor(all_forwards_for_residual, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
            loss = [F.cross_entropy(pre, gt, reduction='none') for pre, gt in zip([first_level, *predictions], codes)]
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            restoredCodes.insert(0, first_level.detach().clone().argmax(1, keepdim=True))
            with torch.no_grad():
                splitted = list(zip(*list(torch.split(x, 16) for x in restoredCodes)))
                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)
            return [first_level, *predictions], sum([(l.sum() / len(image)) for l in loss]), codes, restored, [l.mean() for l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                input_ids = batch_encoding.input_ids
                attention_mask = batch_encoding.attention_mask
                text_embedding: 'transformers.modeling_outputs.BaseModelOutputWithPooling' = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
                first_level = self.text_to_first_level([len(text_embedding), 2, 2], text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
                formerLevel = self.compressor.residual_forward(first_level.unsqueeze(1), None, 0)
                predictions = list()
                for i in range(0, len(self.compressor.Codebooks) - 1):
                    predictions.append(self.next_residual_predictor((formerLevel, i), text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask))
                    formerLevel = self.compressor.residual_forward(predictions[-1].unsqueeze(1), formerLevel, i)
                predictions.insert(0, first_level)
                predictions = [p.unsqueeze(1) for p in predictions]
                splitted = list(zip(*list(torch.split(x, 16) for x in predictions)))
                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)
                return predictions, restored


class Generator(nn.Module):

    def __init__(self, channel: 'int', k: 'List[int]', denseNorm: 'bool', loadFrom: 'str', *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, denseNorm)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items()})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        logging.debug('Start loading clip...')
        self.text_encoder = transformers.CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', local_files_only=False)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        self.text_tokenizer = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=False)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        for params in self.text_encoder.parameters():
            params.requires_grad_(False)
        clip_text_channels = self.text_encoder.text_model.config.hidden_size
        self.text_to_first_level, self.next_residual_predictor = AnyRes_S(clip_text_channels, [2, 4, 8, 16], [codebook.squeeze(0) for codebook in self.compressor.Codebooks])
        logging.debug('Created any-res transformer.')
        self.compressor.eval()
        self.text_encoder.eval()

    def train(self, mode: 'bool'=True):
        self.compressor.eval()
        self.text_encoder.eval()
        return super().train(mode)

    def forward(self, image, condition: 'List[str]'):
        if not isinstance(condition, list):
            raise NotImplementedError
        if self.training:
            with torch.no_grad():
                splitted = torch.split(image, 16)
                allCodes = list()
                all_forwards_for_residual = list()
                formerLevel = None
                for sp in splitted:
                    codes = self.compressor.encode(sp)
                    allCodes.append(codes)
                    this_split_forward_residual = list()
                    for level, code in enumerate(codes[:-1]):
                        this_split_forward_residual.append(self.compressor.residual_forward(code, formerLevel, level))
                        formerLevel = this_split_forward_residual[-1]
                    all_forwards_for_residual.append(this_split_forward_residual)
                codes = [torch.cat(x) for x in zip(*allCodes)]
                all_forwards_for_residual = [torch.cat(x) for x in zip(*all_forwards_for_residual)]
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                input_ids = batch_encoding.input_ids
                attention_mask = batch_encoding.attention_mask
                text_embedding: 'transformers.modeling_outputs.BaseModelOutputWithPooling' = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            codes = [c.squeeze(1) for c in codes]
            first_level = self.text_to_first_level(codes[0].shape, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
            predictions = self.next_residual_predictor(all_forwards_for_residual, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
            loss = [F.cross_entropy(pre, gt, reduction='none') for pre, gt in zip([first_level, *predictions], codes)]
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            restoredCodes.insert(0, first_level.detach().clone().argmax(1, keepdim=True))
            with torch.no_grad():
                splitted = list(zip(*list(torch.split(x, 16) for x in restoredCodes)))
                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)
            return [first_level, *predictions], sum([(l.sum() / len(image)) for l in loss]), codes, restored, [l.mean() for l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                input_ids = batch_encoding.input_ids
                attention_mask = batch_encoding.attention_mask
                text_embedding: 'transformers.modeling_outputs.BaseModelOutputWithPooling' = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
                first_level = self.text_to_first_level([len(text_embedding), 2, 2], text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)
                formerLevel = self.compressor.residual_forward(first_level.unsqueeze(1), None, 0)
                predictions = list()
                for i in range(0, len(self.compressor.Codebooks) - 1):
                    predictions.append(self.next_residual_predictor((formerLevel, i), text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask))
                    formerLevel = self.compressor.residual_forward(predictions[-1].unsqueeze(1), formerLevel, i)
                predictions.insert(0, first_level)
                predictions = [p.unsqueeze(1) for p in predictions]
                splitted = list(zip(*list(torch.split(x, 16) for x in predictions)))
                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)
                return predictions, restored


def AnyRes_L(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, codebooks[0], hidden_size=768, depth=24, num_heads=16, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=24, hidden_size=768, num_heads=16, **kwargs)


IMAGENET2012_CLASSES = OrderedDict({'n01440764': 'tench, Tinca tinca', 'n01443537': 'goldfish, Carassius auratus', 'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'n01491361': 'tiger shark, Galeocerdo cuvieri', 'n01494475': 'hammerhead, hammerhead shark', 'n01496331': 'electric ray, crampfish, numbfish, torpedo', 'n01498041': 'stingray', 'n01514668': 'cock', 'n01514859': 'hen', 'n01518878': 'ostrich, Struthio camelus', 'n01530575': 'brambling, Fringilla montifringilla', 'n01531178': 'goldfinch, Carduelis carduelis', 'n01532829': 'house finch, linnet, Carpodacus mexicanus', 'n01534433': 'junco, snowbird', 'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'n01558993': 'robin, American robin, Turdus migratorius', 'n01560419': 'bulbul', 'n01580077': 'jay', 'n01582220': 'magpie', 'n01592084': 'chickadee', 'n01601694': 'water ouzel, dipper', 'n01608432': 'kite', 'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus', 'n01616318': 'vulture', 'n01622779': 'great grey owl, great gray owl, Strix nebulosa', 'n01629819': 'European fire salamander, Salamandra salamandra', 'n01630670': 'common newt, Triturus vulgaris', 'n01631663': 'eft', 'n01632458': 'spotted salamander, Ambystoma maculatum', 'n01632777': 'axolotl, mud puppy, Ambystoma mexicanum', 'n01641577': 'bullfrog, Rana catesbeiana', 'n01644373': 'tree frog, tree-frog', 'n01644900': 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'n01664065': 'loggerhead, loggerhead turtle, Caretta caretta', 'n01665541': 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'n01667114': 'mud turtle', 'n01667778': 'terrapin', 'n01669191': 'box turtle, box tortoise', 'n01675722': 'banded gecko', 'n01677366': 'common iguana, iguana, Iguana iguana', 'n01682714': 'American chameleon, anole, Anolis carolinensis', 'n01685808': 'whiptail, whiptail lizard', 'n01687978': 'agama', 'n01688243': 'frilled lizard, Chlamydosaurus kingi', 'n01689811': 'alligator lizard', 'n01692333': 'Gila monster, Heloderma suspectum', 'n01693334': 'green lizard, Lacerta viridis', 'n01694178': 'African chameleon, Chamaeleo chamaeleon', 'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'n01697457': 'African crocodile, Nile crocodile, Crocodylus niloticus', 'n01698640': 'American alligator, Alligator mississipiensis', 'n01704323': 'triceratops', 'n01728572': 'thunder snake, worm snake, Carphophis amoenus', 'n01728920': 'ringneck snake, ring-necked snake, ring snake', 'n01729322': 'hognose snake, puff adder, sand viper', 'n01729977': 'green snake, grass snake', 'n01734418': 'king snake, kingsnake', 'n01735189': 'garter snake, grass snake', 'n01737021': 'water snake', 'n01739381': 'vine snake', 'n01740131': 'night snake, Hypsiglena torquata', 'n01742172': 'boa constrictor, Constrictor constrictor', 'n01744401': 'rock python, rock snake, Python sebae', 'n01748264': 'Indian cobra, Naja naja', 'n01749939': 'green mamba', 'n01751748': 'sea snake', 'n01753488': 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'n01755581': 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'n01756291': 'sidewinder, horned rattlesnake, Crotalus cerastes', 'n01768244': 'trilobite', 'n01770081': 'harvestman, daddy longlegs, Phalangium opilio', 'n01770393': 'scorpion', 'n01773157': 'black and gold garden spider, Argiope aurantia', 'n01773549': 'barn spider, Araneus cavaticus', 'n01773797': 'garden spider, Aranea diademata', 'n01774384': 'black widow, Latrodectus mactans', 'n01774750': 'tarantula', 'n01775062': 'wolf spider, hunting spider', 'n01776313': 'tick', 'n01784675': 'centipede', 'n01795545': 'black grouse', 'n01796340': 'ptarmigan', 'n01797886': 'ruffed grouse, partridge, Bonasa umbellus', 'n01798484': 'prairie chicken, prairie grouse, prairie fowl', 'n01806143': 'peacock', 'n01806567': 'quail', 'n01807496': 'partridge', 'n01817953': 'African grey, African gray, Psittacus erithacus', 'n01818515': 'macaw', 'n01819313': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'n01820546': 'lorikeet', 'n01824575': 'coucal', 'n01828970': 'bee eater', 'n01829413': 'hornbill', 'n01833805': 'hummingbird', 'n01843065': 'jacamar', 'n01843383': 'toucan', 'n01847000': 'drake', 'n01855032': 'red-breasted merganser, Mergus serrator', 'n01855672': 'goose', 'n01860187': 'black swan, Cygnus atratus', 'n01871265': 'tusker', 'n01872401': 'echidna, spiny anteater, anteater', 'n01873310': 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'n01877812': 'wallaby, brush kangaroo', 'n01882714': 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'n01883070': 'wombat', 'n01910747': 'jellyfish', 'n01914609': 'sea anemone, anemone', 'n01917289': 'brain coral', 'n01924916': 'flatworm, platyhelminth', 'n01930112': 'nematode, nematode worm, roundworm', 'n01943899': 'conch', 'n01944390': 'snail', 'n01945685': 'slug', 'n01950731': 'sea slug, nudibranch', 'n01955084': 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'n01968897': 'chambered nautilus, pearly nautilus, nautilus', 'n01978287': 'Dungeness crab, Cancer magister', 'n01978455': 'rock crab, Cancer irroratus', 'n01980166': 'fiddler crab', 'n01981276': 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'n01983481': 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'n01984695': 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'n01985128': 'crayfish, crawfish, crawdad, crawdaddy', 'n01986214': 'hermit crab', 'n01990800': 'isopod', 'n02002556': 'white stork, Ciconia ciconia', 'n02002724': 'black stork, Ciconia nigra', 'n02006656': 'spoonbill', 'n02007558': 'flamingo', 'n02009229': 'little blue heron, Egretta caerulea', 'n02009912': 'American egret, great white heron, Egretta albus', 'n02011460': 'bittern', 'n02012849': 'crane', 'n02013706': 'limpkin, Aramus pictus', 'n02017213': 'European gallinule, Porphyrio porphyrio', 'n02018207': 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'n02018795': 'bustard', 'n02025239': 'ruddy turnstone, Arenaria interpres', 'n02027492': 'red-backed sandpiper, dunlin, Erolia alpina', 'n02028035': 'redshank, Tringa totanus', 'n02033041': 'dowitcher', 'n02037110': 'oystercatcher, oyster catcher', 'n02051845': 'pelican', 'n02056570': 'king penguin, Aptenodytes patagonica', 'n02058221': 'albatross, mollymawk', 'n02066245': 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'n02071294': 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'n02074367': 'dugong, Dugong dugon', 'n02077923': 'sea lion', 'n02085620': 'Chihuahua', 'n02085782': 'Japanese spaniel', 'n02085936': 'Maltese dog, Maltese terrier, Maltese', 'n02086079': 'Pekinese, Pekingese, Peke', 'n02086240': 'Shih-Tzu', 'n02086646': 'Blenheim spaniel', 'n02086910': 'papillon', 'n02087046': 'toy terrier', 'n02087394': 'Rhodesian ridgeback', 'n02088094': 'Afghan hound, Afghan', 'n02088238': 'basset, basset hound', 'n02088364': 'beagle', 'n02088466': 'bloodhound, sleuthhound', 'n02088632': 'bluetick', 'n02089078': 'black-and-tan coonhound', 'n02089867': 'Walker hound, Walker foxhound', 'n02089973': 'English foxhound', 'n02090379': 'redbone', 'n02090622': 'borzoi, Russian wolfhound', 'n02090721': 'Irish wolfhound', 'n02091032': 'Italian greyhound', 'n02091134': 'whippet', 'n02091244': 'Ibizan hound, Ibizan Podenco', 'n02091467': 'Norwegian elkhound, elkhound', 'n02091635': 'otterhound, otter hound', 'n02091831': 'Saluki, gazelle hound', 'n02092002': 'Scottish deerhound, deerhound', 'n02092339': 'Weimaraner', 'n02093256': 'Staffordshire bullterrier, Staffordshire bull terrier', 'n02093428': 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'n02093647': 'Bedlington terrier', 'n02093754': 'Border terrier', 'n02093859': 'Kerry blue terrier', 'n02093991': 'Irish terrier', 'n02094114': 'Norfolk terrier', 'n02094258': 'Norwich terrier', 'n02094433': 'Yorkshire terrier', 'n02095314': 'wire-haired fox terrier', 'n02095570': 'Lakeland terrier', 'n02095889': 'Sealyham terrier, Sealyham', 'n02096051': 'Airedale, Airedale terrier', 'n02096177': 'cairn, cairn terrier', 'n02096294': 'Australian terrier', 'n02096437': 'Dandie Dinmont, Dandie Dinmont terrier', 'n02096585': 'Boston bull, Boston terrier', 'n02097047': 'miniature schnauzer', 'n02097130': 'giant schnauzer', 'n02097209': 'standard schnauzer', 'n02097298': 'Scotch terrier, Scottish terrier, Scottie', 'n02097474': 'Tibetan terrier, chrysanthemum dog', 'n02097658': 'silky terrier, Sydney silky', 'n02098105': 'soft-coated wheaten terrier', 'n02098286': 'West Highland white terrier', 'n02098413': 'Lhasa, Lhasa apso', 'n02099267': 'flat-coated retriever', 'n02099429': 'curly-coated retriever', 'n02099601': 'golden retriever', 'n02099712': 'Labrador retriever', 'n02099849': 'Chesapeake Bay retriever', 'n02100236': 'German short-haired pointer', 'n02100583': 'vizsla, Hungarian pointer', 'n02100735': 'English setter', 'n02100877': 'Irish setter, red setter', 'n02101006': 'Gordon setter', 'n02101388': 'Brittany spaniel', 'n02101556': 'clumber, clumber spaniel', 'n02102040': 'English springer, English springer spaniel', 'n02102177': 'Welsh springer spaniel', 'n02102318': 'cocker spaniel, English cocker spaniel, cocker', 'n02102480': 'Sussex spaniel', 'n02102973': 'Irish water spaniel', 'n02104029': 'kuvasz', 'n02104365': 'schipperke', 'n02105056': 'groenendael', 'n02105162': 'malinois', 'n02105251': 'briard', 'n02105412': 'kelpie', 'n02105505': 'komondor', 'n02105641': 'Old English sheepdog, bobtail', 'n02105855': 'Shetland sheepdog, Shetland sheep dog, Shetland', 'n02106030': 'collie', 'n02106166': 'Border collie', 'n02106382': 'Bouvier des Flandres, Bouviers des Flandres', 'n02106550': 'Rottweiler', 'n02106662': 'German shepherd, German shepherd dog, German police dog, alsatian', 'n02107142': 'Doberman, Doberman pinscher', 'n02107312': 'miniature pinscher', 'n02107574': 'Greater Swiss Mountain dog', 'n02107683': 'Bernese mountain dog', 'n02107908': 'Appenzeller', 'n02108000': 'EntleBucher', 'n02108089': 'boxer', 'n02108422': 'bull mastiff', 'n02108551': 'Tibetan mastiff', 'n02108915': 'French bulldog', 'n02109047': 'Great Dane', 'n02109525': 'Saint Bernard, St Bernard', 'n02109961': 'Eskimo dog, husky', 'n02110063': 'malamute, malemute, Alaskan malamute', 'n02110185': 'Siberian husky', 'n02110341': 'dalmatian, coach dog, carriage dog', 'n02110627': 'affenpinscher, monkey pinscher, monkey dog', 'n02110806': 'basenji', 'n02110958': 'pug, pug-dog', 'n02111129': 'Leonberg', 'n02111277': 'Newfoundland, Newfoundland dog', 'n02111500': 'Great Pyrenees', 'n02111889': 'Samoyed, Samoyede', 'n02112018': 'Pomeranian', 'n02112137': 'chow, chow chow', 'n02112350': 'keeshond', 'n02112706': 'Brabancon griffon', 'n02113023': 'Pembroke, Pembroke Welsh corgi', 'n02113186': 'Cardigan, Cardigan Welsh corgi', 'n02113624': 'toy poodle', 'n02113712': 'miniature poodle', 'n02113799': 'standard poodle', 'n02113978': 'Mexican hairless', 'n02114367': 'timber wolf, grey wolf, gray wolf, Canis lupus', 'n02114548': 'white wolf, Arctic wolf, Canis lupus tundrarum', 'n02114712': 'red wolf, maned wolf, Canis rufus, Canis niger', 'n02114855': 'coyote, prairie wolf, brush wolf, Canis latrans', 'n02115641': 'dingo, warrigal, warragal, Canis dingo', 'n02115913': 'dhole, Cuon alpinus', 'n02116738': 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'n02117135': 'hyena, hyaena', 'n02119022': 'red fox, Vulpes vulpes', 'n02119789': 'kit fox, Vulpes macrotis', 'n02120079': 'Arctic fox, white fox, Alopex lagopus', 'n02120505': 'grey fox, gray fox, Urocyon cinereoargenteus', 'n02123045': 'tabby, tabby cat', 'n02123159': 'tiger cat', 'n02123394': 'Persian cat', 'n02123597': 'Siamese cat, Siamese', 'n02124075': 'Egyptian cat', 'n02125311': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'n02127052': 'lynx, catamount', 'n02128385': 'leopard, Panthera pardus', 'n02128757': 'snow leopard, ounce, Panthera uncia', 'n02128925': 'jaguar, panther, Panthera onca, Felis onca', 'n02129165': 'lion, king of beasts, Panthera leo', 'n02129604': 'tiger, Panthera tigris', 'n02130308': 'cheetah, chetah, Acinonyx jubatus', 'n02132136': 'brown bear, bruin, Ursus arctos', 'n02133161': 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'n02134084': 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'n02134418': 'sloth bear, Melursus ursinus, Ursus ursinus', 'n02137549': 'mongoose', 'n02138441': 'meerkat, mierkat', 'n02165105': 'tiger beetle', 'n02165456': 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'n02167151': 'ground beetle, carabid beetle', 'n02168699': 'long-horned beetle, longicorn, longicorn beetle', 'n02169497': 'leaf beetle, chrysomelid', 'n02172182': 'dung beetle', 'n02174001': 'rhinoceros beetle', 'n02177972': 'weevil', 'n02190166': 'fly', 'n02206856': 'bee', 'n02219486': 'ant, emmet, pismire', 'n02226429': 'grasshopper, hopper', 'n02229544': 'cricket', 'n02231487': 'walking stick, walkingstick, stick insect', 'n02233338': 'cockroach, roach', 'n02236044': 'mantis, mantid', 'n02256656': 'cicada, cicala', 'n02259212': 'leafhopper', 'n02264363': 'lacewing, lacewing fly', 'n02268443': "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'n02268853': 'damselfly', 'n02276258': 'admiral', 'n02277742': 'ringlet, ringlet butterfly', 'n02279972': 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'n02280649': 'cabbage butterfly', 'n02281406': 'sulphur butterfly, sulfur butterfly', 'n02281787': 'lycaenid, lycaenid butterfly', 'n02317335': 'starfish, sea star', 'n02319095': 'sea urchin', 'n02321529': 'sea cucumber, holothurian', 'n02325366': 'wood rabbit, cottontail, cottontail rabbit', 'n02326432': 'hare', 'n02328150': 'Angora, Angora rabbit', 'n02342885': 'hamster', 'n02346627': 'porcupine, hedgehog', 'n02356798': 'fox squirrel, eastern fox squirrel, Sciurus niger', 'n02361337': 'marmot', 'n02363005': 'beaver', 'n02364673': 'guinea pig, Cavia cobaya', 'n02389026': 'sorrel', 'n02391049': 'zebra', 'n02395406': 'hog, pig, grunter, squealer, Sus scrofa', 'n02396427': 'wild boar, boar, Sus scrofa', 'n02397096': 'warthog', 'n02398521': 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'n02403003': 'ox', 'n02408429': 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'n02410509': 'bison', 'n02412080': 'ram, tup', 'n02415577': 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'n02417914': 'ibex, Capra ibex', 'n02422106': 'hartebeest', 'n02422699': 'impala, Aepyceros melampus', 'n02423022': 'gazelle', 'n02437312': 'Arabian camel, dromedary, Camelus dromedarius', 'n02437616': 'llama', 'n02441942': 'weasel', 'n02442845': 'mink', 'n02443114': 'polecat, fitch, foulmart, foumart, Mustela putorius', 'n02443484': 'black-footed ferret, ferret, Mustela nigripes', 'n02444819': 'otter', 'n02445715': 'skunk, polecat, wood pussy', 'n02447366': 'badger', 'n02454379': 'armadillo', 'n02457408': 'three-toed sloth, ai, Bradypus tridactylus', 'n02480495': 'orangutan, orang, orangutang, Pongo pygmaeus', 'n02480855': 'gorilla, Gorilla gorilla', 'n02481823': 'chimpanzee, chimp, Pan troglodytes', 'n02483362': 'gibbon, Hylobates lar', 'n02483708': 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'n02484975': 'guenon, guenon monkey', 'n02486261': 'patas, hussar monkey, Erythrocebus patas', 'n02486410': 'baboon', 'n02487347': 'macaque', 'n02488291': 'langur', 'n02488702': 'colobus, colobus monkey', 'n02489166': 'proboscis monkey, Nasalis larvatus', 'n02490219': 'marmoset', 'n02492035': 'capuchin, ringtail, Cebus capucinus', 'n02492660': 'howler monkey, howler', 'n02493509': 'titi, titi monkey', 'n02493793': 'spider monkey, Ateles geoffroyi', 'n02494079': 'squirrel monkey, Saimiri sciureus', 'n02497673': 'Madagascar cat, ring-tailed lemur, Lemur catta', 'n02500267': 'indri, indris, Indri indri, Indri brevicaudatus', 'n02504013': 'Indian elephant, Elephas maximus', 'n02504458': 'African elephant, Loxodonta africana', 'n02509815': 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'n02510455': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'n02514041': 'barracouta, snoek', 'n02526121': 'eel', 'n02536864': 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'n02606052': 'rock beauty, Holocanthus tricolor', 'n02607072': 'anemone fish', 'n02640242': 'sturgeon', 'n02641379': 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'n02643566': 'lionfish', 'n02655020': 'puffer, pufferfish, blowfish, globefish', 'n02666196': 'abacus', 'n02667093': 'abaya', 'n02669723': "academic gown, academic robe, judge's robe", 'n02672831': 'accordion, piano accordion, squeeze box', 'n02676566': 'acoustic guitar', 'n02687172': 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'n02690373': 'airliner', 'n02692877': 'airship, dirigible', 'n02699494': 'altar', 'n02701002': 'ambulance', 'n02704792': 'amphibian, amphibious vehicle', 'n02708093': 'analog clock', 'n02727426': 'apiary, bee house', 'n02730930': 'apron', 'n02747177': 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'n02749479': 'assault rifle, assault gun', 'n02769748': 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'n02776631': 'bakery, bakeshop, bakehouse', 'n02777292': 'balance beam, beam', 'n02782093': 'balloon', 'n02783161': 'ballpoint, ballpoint pen, ballpen, Biro', 'n02786058': 'Band Aid', 'n02787622': 'banjo', 'n02788148': 'bannister, banister, balustrade, balusters, handrail', 'n02790996': 'barbell', 'n02791124': 'barber chair', 'n02791270': 'barbershop', 'n02793495': 'barn', 'n02794156': 'barometer', 'n02795169': 'barrel, cask', 'n02797295': 'barrow, garden cart, lawn cart, wheelbarrow', 'n02799071': 'baseball', 'n02802426': 'basketball', 'n02804414': 'bassinet', 'n02804610': 'bassoon', 'n02807133': 'bathing cap, swimming cap', 'n02808304': 'bath towel', 'n02808440': 'bathtub, bathing tub, bath, tub', 'n02814533': 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'n02814860': 'beacon, lighthouse, beacon light, pharos', 'n02815834': 'beaker', 'n02817516': 'bearskin, busby, shako', 'n02823428': 'beer bottle', 'n02823750': 'beer glass', 'n02825657': 'bell cote, bell cot', 'n02834397': 'bib', 'n02835271': 'bicycle-built-for-two, tandem bicycle, tandem', 'n02837789': 'bikini, two-piece', 'n02840245': 'binder, ring-binder', 'n02841315': 'binoculars, field glasses, opera glasses', 'n02843684': 'birdhouse', 'n02859443': 'boathouse', 'n02860847': 'bobsled, bobsleigh, bob', 'n02865351': 'bolo tie, bolo, bola tie, bola', 'n02869837': 'bonnet, poke bonnet', 'n02870880': 'bookcase', 'n02871525': 'bookshop, bookstore, bookstall', 'n02877765': 'bottlecap', 'n02879718': 'bow', 'n02883205': 'bow tie, bow-tie, bowtie', 'n02892201': 'brass, memorial tablet, plaque', 'n02892767': 'brassiere, bra, bandeau', 'n02894605': 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'n02895154': 'breastplate, aegis, egis', 'n02906734': 'broom', 'n02909870': 'bucket, pail', 'n02910353': 'buckle', 'n02916936': 'bulletproof vest', 'n02917067': 'bullet train, bullet', 'n02927161': 'butcher shop, meat market', 'n02930766': 'cab, hack, taxi, taxicab', 'n02939185': 'caldron, cauldron', 'n02948072': 'candle, taper, wax light', 'n02950826': 'cannon', 'n02951358': 'canoe', 'n02951585': 'can opener, tin opener', 'n02963159': 'cardigan', 'n02965783': 'car mirror', 'n02966193': 'carousel, carrousel, merry-go-round, roundabout, whirligig', 'n02966687': "carpenter's kit, tool kit", 'n02971356': 'carton', 'n02974003': 'car wheel', 'n02977058': 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'n02978881': 'cassette', 'n02979186': 'cassette player', 'n02980441': 'castle', 'n02981792': 'catamaran', 'n02988304': 'CD player', 'n02992211': 'cello, violoncello', 'n02992529': 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'n02999410': 'chain', 'n03000134': 'chainlink fence', 'n03000247': 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'n03000684': 'chain saw, chainsaw', 'n03014705': 'chest', 'n03016953': 'chiffonier, commode', 'n03017168': 'chime, bell, gong', 'n03018349': 'china cabinet, china closet', 'n03026506': 'Christmas stocking', 'n03028079': 'church, church building', 'n03032252': 'cinema, movie theater, movie theatre, movie house, picture palace', 'n03041632': 'cleaver, meat cleaver, chopper', 'n03042490': 'cliff dwelling', 'n03045698': 'cloak', 'n03047690': 'clog, geta, patten, sabot', 'n03062245': 'cocktail shaker', 'n03063599': 'coffee mug', 'n03063689': 'coffeepot', 'n03065424': 'coil, spiral, volute, whorl, helix', 'n03075370': 'combination lock', 'n03085013': 'computer keyboard, keypad', 'n03089624': 'confectionery, confectionary, candy store', 'n03095699': 'container ship, containership, container vessel', 'n03100240': 'convertible', 'n03109150': 'corkscrew, bottle screw', 'n03110669': 'cornet, horn, trumpet, trump', 'n03124043': 'cowboy boot', 'n03124170': 'cowboy hat, ten-gallon hat', 'n03125729': 'cradle', 'n03126707': 'crane2', 'n03127747': 'crash helmet', 'n03127925': 'crate', 'n03131574': 'crib, cot', 'n03133878': 'Crock Pot', 'n03134739': 'croquet ball', 'n03141823': 'crutch', 'n03146219': 'cuirass', 'n03160309': 'dam, dike, dyke', 'n03179701': 'desk', 'n03180011': 'desktop computer', 'n03187595': 'dial telephone, dial phone', 'n03188531': 'diaper, nappy, napkin', 'n03196217': 'digital clock', 'n03197337': 'digital watch', 'n03201208': 'dining table, board', 'n03207743': 'dishrag, dishcloth', 'n03207941': 'dishwasher, dish washer, dishwashing machine', 'n03208938': 'disk brake, disc brake', 'n03216828': 'dock, dockage, docking facility', 'n03218198': 'dogsled, dog sled, dog sleigh', 'n03220513': 'dome', 'n03223299': 'doormat, welcome mat', 'n03240683': 'drilling platform, offshore rig', 'n03249569': 'drum, membranophone, tympan', 'n03250847': 'drumstick', 'n03255030': 'dumbbell', 'n03259280': 'Dutch oven', 'n03271574': 'electric fan, blower', 'n03272010': 'electric guitar', 'n03272562': 'electric locomotive', 'n03290653': 'entertainment center', 'n03291819': 'envelope', 'n03297495': 'espresso maker', 'n03314780': 'face powder', 'n03325584': 'feather boa, boa', 'n03337140': 'file, file cabinet, filing cabinet', 'n03344393': 'fireboat', 'n03345487': 'fire engine, fire truck', 'n03347037': 'fire screen, fireguard', 'n03355925': 'flagpole, flagstaff', 'n03372029': 'flute, transverse flute', 'n03376595': 'folding chair', 'n03379051': 'football helmet', 'n03384352': 'forklift', 'n03388043': 'fountain', 'n03388183': 'fountain pen', 'n03388549': 'four-poster', 'n03393912': 'freight car', 'n03394916': 'French horn, horn', 'n03400231': 'frying pan, frypan, skillet', 'n03404251': 'fur coat', 'n03417042': 'garbage truck, dustcart', 'n03424325': 'gasmask, respirator, gas helmet', 'n03425413': 'gas pump, gasoline pump, petrol pump, island dispenser', 'n03443371': 'goblet', 'n03444034': 'go-kart', 'n03445777': 'golf ball', 'n03445924': 'golfcart, golf cart', 'n03447447': 'gondola', 'n03447721': 'gong, tam-tam', 'n03450230': 'gown', 'n03452741': 'grand piano, grand', 'n03457902': 'greenhouse, nursery, glasshouse', 'n03459775': 'grille, radiator grille', 'n03461385': 'grocery store, grocery, food market, market', 'n03467068': 'guillotine', 'n03476684': 'hair slide', 'n03476991': 'hair spray', 'n03478589': 'half track', 'n03481172': 'hammer', 'n03482405': 'hamper', 'n03483316': 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'n03485407': 'hand-held computer, hand-held microcomputer', 'n03485794': 'handkerchief, hankie, hanky, hankey', 'n03492542': 'hard disc, hard disk, fixed disk', 'n03494278': 'harmonica, mouth organ, harp, mouth harp', 'n03495258': 'harp', 'n03496892': 'harvester, reaper', 'n03498962': 'hatchet', 'n03527444': 'holster', 'n03529860': 'home theater, home theatre', 'n03530642': 'honeycomb', 'n03532672': 'hook, claw', 'n03534580': 'hoopskirt, crinoline', 'n03535780': 'horizontal bar, high bar', 'n03538406': 'horse cart, horse-cart', 'n03544143': 'hourglass', 'n03584254': 'iPod', 'n03584829': 'iron, smoothing iron', 'n03590841': "jack-o'-lantern", 'n03594734': 'jean, blue jean, denim', 'n03594945': 'jeep, landrover', 'n03595614': 'jersey, T-shirt, tee shirt', 'n03598930': 'jigsaw puzzle', 'n03599486': 'jinrikisha, ricksha, rickshaw', 'n03602883': 'joystick', 'n03617480': 'kimono', 'n03623198': 'knee pad', 'n03627232': 'knot', 'n03630383': 'lab coat, laboratory coat', 'n03633091': 'ladle', 'n03637318': 'lampshade, lamp shade', 'n03642806': 'laptop, laptop computer', 'n03649909': 'lawn mower, mower', 'n03657121': 'lens cap, lens cover', 'n03658185': 'letter opener, paper knife, paperknife', 'n03661043': 'library', 'n03662601': 'lifeboat', 'n03666591': 'lighter, light, igniter, ignitor', 'n03670208': 'limousine, limo', 'n03673027': 'liner, ocean liner', 'n03676483': 'lipstick, lip rouge', 'n03680355': 'Loafer', 'n03690938': 'lotion', 'n03691459': 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', 'n03692522': "loupe, jeweler's loupe", 'n03697007': 'lumbermill, sawmill', 'n03706229': 'magnetic compass', 'n03709823': 'mailbag, postbag', 'n03710193': 'mailbox, letter box', 'n03710637': 'maillot', 'n03710721': 'maillot, tank suit', 'n03717622': 'manhole cover', 'n03720891': 'maraca', 'n03721384': 'marimba, xylophone', 'n03724870': 'mask', 'n03729826': 'matchstick', 'n03733131': 'maypole', 'n03733281': 'maze, labyrinth', 'n03733805': 'measuring cup', 'n03742115': 'medicine chest, medicine cabinet', 'n03743016': 'megalith, megalithic structure', 'n03759954': 'microphone, mike', 'n03761084': 'microwave, microwave oven', 'n03763968': 'military uniform', 'n03764736': 'milk can', 'n03769881': 'minibus', 'n03770439': 'miniskirt, mini', 'n03770679': 'minivan', 'n03773504': 'missile', 'n03775071': 'mitten', 'n03775546': 'mixing bowl', 'n03776460': 'mobile home, manufactured home', 'n03777568': 'Model T', 'n03777754': 'modem', 'n03781244': 'monastery', 'n03782006': 'monitor', 'n03785016': 'moped', 'n03786901': 'mortar', 'n03787032': 'mortarboard', 'n03788195': 'mosque', 'n03788365': 'mosquito net', 'n03791053': 'motor scooter, scooter', 'n03792782': 'mountain bike, all-terrain bike, off-roader', 'n03792972': 'mountain tent', 'n03793489': 'mouse, computer mouse', 'n03794056': 'mousetrap', 'n03796401': 'moving van', 'n03803284': 'muzzle', 'n03804744': 'nail', 'n03814639': 'neck brace', 'n03814906': 'necklace', 'n03825788': 'nipple', 'n03832673': 'notebook, notebook computer', 'n03837869': 'obelisk', 'n03838899': 'oboe, hautboy, hautbois', 'n03840681': 'ocarina, sweet potato', 'n03841143': 'odometer, hodometer, mileometer, milometer', 'n03843555': 'oil filter', 'n03854065': 'organ, pipe organ', 'n03857828': 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'n03866082': 'overskirt', 'n03868242': 'oxcart', 'n03868863': 'oxygen mask', 'n03871628': 'packet', 'n03873416': 'paddle, boat paddle', 'n03874293': 'paddlewheel, paddle wheel', 'n03874599': 'padlock', 'n03876231': 'paintbrush', 'n03877472': "pajama, pyjama, pj's, jammies", 'n03877845': 'palace', 'n03884397': 'panpipe, pandean pipe, syrinx', 'n03887697': 'paper towel', 'n03888257': 'parachute, chute', 'n03888605': 'parallel bars, bars', 'n03891251': 'park bench', 'n03891332': 'parking meter', 'n03895866': 'passenger car, coach, carriage', 'n03899768': 'patio, terrace', 'n03902125': 'pay-phone, pay-station', 'n03903868': 'pedestal, plinth, footstall', 'n03908618': 'pencil box, pencil case', 'n03908714': 'pencil sharpener', 'n03916031': 'perfume, essence', 'n03920288': 'Petri dish', 'n03924679': 'photocopier', 'n03929660': 'pick, plectrum, plectron', 'n03929855': 'pickelhaube', 'n03930313': 'picket fence, paling', 'n03930630': 'pickup, pickup truck', 'n03933933': 'pier', 'n03935335': 'piggy bank, penny bank', 'n03937543': 'pill bottle', 'n03938244': 'pillow', 'n03942813': 'ping-pong ball', 'n03944341': 'pinwheel', 'n03947888': 'pirate, pirate ship', 'n03950228': 'pitcher, ewer', 'n03954731': "plane, carpenter's plane, woodworking plane", 'n03956157': 'planetarium', 'n03958227': 'plastic bag', 'n03961711': 'plate rack', 'n03967562': 'plow, plough', 'n03970156': "plunger, plumber's helper", 'n03976467': 'Polaroid camera, Polaroid Land camera', 'n03976657': 'pole', 'n03977966': 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'n03980874': 'poncho', 'n03982430': 'pool table, billiard table, snooker table', 'n03983396': 'pop bottle, soda bottle', 'n03991062': 'pot, flowerpot', 'n03992509': "potter's wheel", 'n03995372': 'power drill', 'n03998194': 'prayer rug, prayer mat', 'n04004767': 'printer', 'n04005630': 'prison, prison house', 'n04008634': 'projectile, missile', 'n04009552': 'projector', 'n04019541': 'puck, hockey puck', 'n04023962': 'punching bag, punch bag, punching ball, punchball', 'n04026417': 'purse', 'n04033901': 'quill, quill pen', 'n04033995': 'quilt, comforter, comfort, puff', 'n04037443': 'racer, race car, racing car', 'n04039381': 'racket, racquet', 'n04040759': 'radiator', 'n04041544': 'radio, wireless', 'n04044716': 'radio telescope, radio reflector', 'n04049303': 'rain barrel', 'n04065272': 'recreational vehicle, RV, R.V.', 'n04067472': 'reel', 'n04069434': 'reflex camera', 'n04070727': 'refrigerator, icebox', 'n04074963': 'remote control, remote', 'n04081281': 'restaurant, eating house, eating place, eatery', 'n04086273': 'revolver, six-gun, six-shooter', 'n04090263': 'rifle', 'n04099969': 'rocking chair, rocker', 'n04111531': 'rotisserie', 'n04116512': 'rubber eraser, rubber, pencil eraser', 'n04118538': 'rugby ball', 'n04118776': 'rule, ruler', 'n04120489': 'running shoe', 'n04125021': 'safe', 'n04127249': 'safety pin', 'n04131690': 'saltshaker, salt shaker', 'n04133789': 'sandal', 'n04136333': 'sarong', 'n04141076': 'sax, saxophone', 'n04141327': 'scabbard', 'n04141975': 'scale, weighing machine', 'n04146614': 'school bus', 'n04147183': 'schooner', 'n04149813': 'scoreboard', 'n04152593': 'screen, CRT screen', 'n04153751': 'screw', 'n04154565': 'screwdriver', 'n04162706': 'seat belt, seatbelt', 'n04179913': 'sewing machine', 'n04192698': 'shield, buckler', 'n04200800': 'shoe shop, shoe-shop, shoe store', 'n04201297': 'shoji', 'n04204238': 'shopping basket', 'n04204347': 'shopping cart', 'n04208210': 'shovel', 'n04209133': 'shower cap', 'n04209239': 'shower curtain', 'n04228054': 'ski', 'n04229816': 'ski mask', 'n04235860': 'sleeping bag', 'n04238763': 'slide rule, slipstick', 'n04239074': 'sliding door', 'n04243546': 'slot, one-armed bandit', 'n04251144': 'snorkel', 'n04252077': 'snowmobile', 'n04252225': 'snowplow, snowplough', 'n04254120': 'soap dispenser', 'n04254680': 'soccer ball', 'n04254777': 'sock', 'n04258138': 'solar dish, solar collector, solar furnace', 'n04259630': 'sombrero', 'n04263257': 'soup bowl', 'n04264628': 'space bar', 'n04265275': 'space heater', 'n04266014': 'space shuttle', 'n04270147': 'spatula', 'n04273569': 'speedboat', 'n04275548': "spider web, spider's web", 'n04277352': 'spindle', 'n04285008': 'sports car, sport car', 'n04286575': 'spotlight, spot', 'n04296562': 'stage', 'n04310018': 'steam locomotive', 'n04311004': 'steel arch bridge', 'n04311174': 'steel drum', 'n04317175': 'stethoscope', 'n04325704': 'stole', 'n04326547': 'stone wall', 'n04328186': 'stopwatch, stop watch', 'n04330267': 'stove', 'n04332243': 'strainer', 'n04335435': 'streetcar, tram, tramcar, trolley, trolley car', 'n04336792': 'stretcher', 'n04344873': 'studio couch, day bed', 'n04346328': 'stupa, tope', 'n04347754': 'submarine, pigboat, sub, U-boat', 'n04350905': 'suit, suit of clothes', 'n04355338': 'sundial', 'n04355933': 'sunglass', 'n04356056': 'sunglasses, dark glasses, shades', 'n04357314': 'sunscreen, sunblock, sun blocker', 'n04366367': 'suspension bridge', 'n04367480': 'swab, swob, mop', 'n04370456': 'sweatshirt', 'n04371430': 'swimming trunks, bathing trunks', 'n04371774': 'swing', 'n04372370': 'switch, electric switch, electrical switch', 'n04376876': 'syringe', 'n04380533': 'table lamp', 'n04389033': 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'n04392985': 'tape player', 'n04398044': 'teapot', 'n04399382': 'teddy, teddy bear', 'n04404412': 'television, television system', 'n04409515': 'tennis ball', 'n04417672': 'thatch, thatched roof', 'n04418357': 'theater curtain, theatre curtain', 'n04423845': 'thimble', 'n04428191': 'thresher, thrasher, threshing machine', 'n04429376': 'throne', 'n04435653': 'tile roof', 'n04442312': 'toaster', 'n04443257': 'tobacco shop, tobacconist shop, tobacconist', 'n04447861': 'toilet seat', 'n04456115': 'torch', 'n04458633': 'totem pole', 'n04461696': 'tow truck, tow car, wrecker', 'n04462240': 'toyshop', 'n04465501': 'tractor', 'n04467665': 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'n04476259': 'tray', 'n04479046': 'trench coat', 'n04482393': 'tricycle, trike, velocipede', 'n04483307': 'trimaran', 'n04485082': 'tripod', 'n04486054': 'triumphal arch', 'n04487081': 'trolleybus, trolley coach, trackless trolley', 'n04487394': 'trombone', 'n04493381': 'tub, vat', 'n04501370': 'turnstile', 'n04505470': 'typewriter keyboard', 'n04507155': 'umbrella', 'n04509417': 'unicycle, monocycle', 'n04515003': 'upright, upright piano', 'n04517823': 'vacuum, vacuum cleaner', 'n04522168': 'vase', 'n04523525': 'vault', 'n04525038': 'velvet', 'n04525305': 'vending machine', 'n04532106': 'vestment', 'n04532670': 'viaduct', 'n04536866': 'violin, fiddle', 'n04540053': 'volleyball', 'n04542943': 'waffle iron', 'n04548280': 'wall clock', 'n04548362': 'wallet, billfold, notecase, pocketbook', 'n04550184': 'wardrobe, closet, press', 'n04552348': 'warplane, military plane', 'n04553703': 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'n04554684': 'washer, automatic washer, washing machine', 'n04557648': 'water bottle', 'n04560804': 'water jug', 'n04562935': 'water tower', 'n04579145': 'whiskey jug', 'n04579432': 'whistle', 'n04584207': 'wig', 'n04589890': 'window screen', 'n04590129': 'window shade', 'n04591157': 'Windsor tie', 'n04591713': 'wine bottle', 'n04592741': 'wing', 'n04596742': 'wok', 'n04597913': 'wooden spoon', 'n04599235': 'wool, woolen, woollen', 'n04604644': 'worm fence, snake fence, snake-rail fence, Virginia fence', 'n04606251': 'wreck', 'n04612504': 'yawl', 'n04613696': 'yurt', 'n06359193': 'web site, website, internet site, site', 'n06596364': 'comic book', 'n06785654': 'crossword puzzle, crossword', 'n06794110': 'street sign', 'n06874185': 'traffic light, traffic signal, stoplight', 'n07248320': 'book jacket, dust cover, dust jacket, dust wrapper', 'n07565083': 'menu', 'n07579787': 'plate', 'n07583066': 'guacamole', 'n07584110': 'consomme', 'n07590611': 'hot pot, hotpot', 'n07613480': 'trifle', 'n07614500': 'ice cream, icecream', 'n07615774': 'ice lolly, lolly, lollipop, popsicle', 'n07684084': 'French loaf', 'n07693725': 'bagel, beigel', 'n07695742': 'pretzel', 'n07697313': 'cheeseburger', 'n07697537': 'hotdog, hot dog, red hot', 'n07711569': 'mashed potato', 'n07714571': 'head cabbage', 'n07714990': 'broccoli', 'n07715103': 'cauliflower', 'n07716358': 'zucchini, courgette', 'n07716906': 'spaghetti squash', 'n07717410': 'acorn squash', 'n07717556': 'butternut squash', 'n07718472': 'cucumber, cuke', 'n07718747': 'artichoke, globe artichoke', 'n07720875': 'bell pepper', 'n07730033': 'cardoon', 'n07734744': 'mushroom', 'n07742313': 'Granny Smith', 'n07745940': 'strawberry', 'n07747607': 'orange', 'n07749582': 'lemon', 'n07753113': 'fig', 'n07753275': 'pineapple, ananas', 'n07753592': 'banana', 'n07754684': 'jackfruit, jak, jack', 'n07760859': 'custard apple', 'n07768694': 'pomegranate', 'n07802026': 'hay', 'n07831146': 'carbonara', 'n07836838': 'chocolate sauce, chocolate syrup', 'n07860988': 'dough', 'n07871810': 'meat loaf, meatloaf', 'n07873807': 'pizza, pizza pie', 'n07875152': 'potpie', 'n07880968': 'burrito', 'n07892512': 'red wine', 'n07920052': 'espresso', 'n07930864': 'cup', 'n07932039': 'eggnog', 'n09193705': 'alp', 'n09229709': 'bubble', 'n09246464': 'cliff, drop, drop-off', 'n09256479': 'coral reef', 'n09288635': 'geyser', 'n09332890': 'lakeside, lakeshore', 'n09399592': 'promontory, headland, head, foreland', 'n09421951': 'sandbar, sand bar', 'n09428293': 'seashore, coast, seacoast, sea-coast', 'n09468604': 'valley, vale', 'n09472597': 'volcano', 'n09835506': 'ballplayer, baseball player', 'n10148035': 'groom, bridegroom', 'n10565667': 'scuba diver', 'n11879895': 'rapeseed', 'n11939491': 'daisy', 'n12057211': "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'n12144580': 'corn', 'n12267677': 'acorn', 'n12620546': 'hip, rose hip, rosehip', 'n12768682': 'buckeye, horse chestnut, conker', 'n12985857': 'coral fungus', 'n12998815': 'agaric', 'n13037406': 'gyromitra', 'n13040303': 'stinkhorn, carrion fungus', 'n13044778': 'earthstar', 'n13052670': 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'n13054560': 'bolete', 'n13133613': 'ear, spike, capitulum', 'n15075141': 'toilet tissue, toilet paper, bathroom tissue'})


IMAGENET2012_LABELS = OrderedDict({key: i for i, key in enumerate(IMAGENET2012_CLASSES.keys())})


class GeneratorV3(nn.Module):

    def __init__(self, channel: 'int', k: 'List[int]', denseNorm: 'bool', loadFrom: 'str', qk_norm: 'bool', norm_eps: 'float', *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, denseNorm)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items() if '_lpips' not in k})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        logging.debug('Start loading clip...')
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        self.text_tokenizer = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        logging.debug('Loaded clip text model from %s.', 'openai/clip-vit-base-patch32')
        for params in self.text_encoder.parameters():
            params.requires_grad_(False)
        clip_text_channels = self.text_encoder.text_model.config.hidden_size
        self.next_residual_predictor = AnyRes_L([1, 2, 4, 8, 16], [[4096, 32] for _ in [1, 2, 4, 8, 16]], qk_norm=qk_norm, norm_eps=norm_eps)
        logging.debug('Created any-res transformer.')
        self.class_tokens = nn.Parameter(nn.init.trunc_normal_(torch.empty(len(IMAGENET2012_LABELS), 8, self.next_residual_predictor.hidden_size), std=math.sqrt(2 / (5 * self.next_residual_predictor.hidden_size))))
        self.compressor.eval()

    def train(self, mode: 'bool'=True):
        retValue = super().train(mode)
        self.compressor.eval()
        return retValue

    def forward(self, image, condition: 'torch.Tensor'):
        if self.training:
            with torch.no_grad():
                all_forwards_for_residual = list()
                codes = self.compressor.encode(image)
                formerLevel = None
                for level, code in enumerate(codes[:-1]):
                    all_forwards_for_residual.append(self.compressor.residual_forward(code, formerLevel, level))
                    formerLevel = all_forwards_for_residual[-1]
            codes = [c.squeeze(1) for c in codes]
            rawPredictions = self.next_residual_predictor([None, *all_forwards_for_residual], self.class_tokens[condition].mean(1), self.class_tokens[condition])
            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx:curIdx + h * w]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append(F.cross_entropy(pre, gt, reduction='none'))
                curIdx += h * w
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            with torch.no_grad():
                restored = self.compressor.decode(restoredCodes)
            return [*predictions], sum([(l.sum() / len(image)) for l in loss]), codes, restored, [l.mean() for l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                input_ids = batch_encoding.input_ids
                attention_mask = batch_encoding.attention_mask
                text_embedding: 'transformers.modeling_outputs.BaseModelOutputWithPooling' = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
                first_level = self.next_residual_predictor(f'{len(text_embedding)},2,2', text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask.bool())
                formerLevel = self.compressor.residual_forward(first_level.unsqueeze(1), None, 0)
                predictions = list()
                for i in range(1, len(self.compressor.Codebooks)):
                    predictions.append(self.next_residual_predictor((formerLevel, i), text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask.bool()))
                    formerLevel = self.compressor.residual_forward(predictions[-1].unsqueeze(1), formerLevel, i)
                predictions.insert(0, first_level)
                predictions = [p.unsqueeze(1) for p in predictions]
                restored = self.compressor.decode(predictions)
                return predictions, restored


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, dim: 'int', n_heads: 'int', n_kv_heads: 'Optional[int]', qk_norm: 'bool'):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.base_seqlen = None
        self.proportional_attn = False

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):

        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        return query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', pos_embed: 'torch.Tensor') ->torch.Tensor:
        """

        Args:
            x:
            y_feat:
            y_mask:
            pos_embed:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        pos_embed = pos_embed.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq + pos_embed
        xk = xk + pos_embed
        xq, xk = xq, xk
        output = F.scaled_dot_product_attention(xq.permute(0, 2, 1, 3), xk.permute(0, 2, 1, 3), xv.permute(0, 2, 1, 3), attn_mask=x_mask.expand(bsz, 1, -1, -1) if self.training else None).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int'):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first
                layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class Transformer(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """

    def __init__(self, input_dim, canvas_size, cap_dim, hidden_size=1152, depth=28, num_heads=16, vocab_size=4096, norm_eps=1e-05, qk_norm=False):
        super().__init__()
        self.in_channels = input_dim
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.final_layer = checkpoint_wrapper(FinalLayer(hidden_size, vocab_size))
        self.token_embedder = nn.Sequential(nn.LayerNorm(input_dim, norm_eps), nn.Linear(input_dim, hidden_size, bias=True))
        self.num_patches = canvas_size * canvas_size * 64
        self.pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_patches, hidden_size), std=2 / (6 * self.hidden_size)), requires_grad=False)
        self.blocks = nn.ModuleList([checkpoint_wrapper(TransformerBlock(idx, hidden_size, num_heads, num_heads, norm_eps, qk_norm)) for idx in range(depth)])
        self.proj_layer = checkpoint_wrapper(ProjLayer(hidden_size, scale_factor=1))

    def random_pos_embed(self, bs, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        result = list()
        for i in range(bs):
            random_up = random.randint(0, H - h)
            random_left = random.randint(0, W - w)
            result.append(pos_embed[random_up:random_up + h, random_left:random_left + w].reshape(h * w, -1))
        return torch.stack(result)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start + h, left_start:left_start + w].reshape(h * w, -1)

    def unpatchify(self, x, h, w):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, hw, dim = x.shape
        return x.permute(0, 2, 1).reshape(bs, dim, h, w).contiguous()
        return self.pixel_shuffle(x)

    def forward(self, x, x_mask, cap_pooled):
        bs = len(x)
        x = self.token_embedder(x)
        selected_pos_embed = self.pos_embed[:x.shape[1]].expand(bs, x.shape[1], -1)
        for block in self.blocks:
            x = block(x, x_mask, None, selected_pos_embed)
        prediction = self.final_layer(x, None)
        return prediction


class AnyResolutionModel(nn.Module):

    def __init__(self, canvas_size: 'List[int]', codebooks, hidden_size=1152, depth=28, num_heads=16, qk_norm=False, norm_eps=1e-05):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.model = Transformer(hidden_size, canvas_size[-1], hidden_size, hidden_size, depth, num_heads, codebooks[0][0], norm_eps, qk_norm)
        self.input_transform = nn.Sequential(nn.LayerNorm(codebooks[0][1], norm_eps), nn.Linear(codebooks[0][1], hidden_size), nn.LayerNorm(hidden_size, norm_eps))
        self.first_level_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, canvas_size[-1] * canvas_size[-1], self.hidden_size), std=math.sqrt(2 / 5)))
        self.cap_to_first_token = nn.Sequential(nn.LayerNorm(hidden_size, norm_eps), nn.Linear(hidden_size, hidden_size, bias=True))
        self.level_indicator_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(len(canvas_size), self.hidden_size), std=math.sqrt(2 / (5 * self.hidden_size))))
        self.register_buffer('input_mask', self.prepare_input_mask(canvas_size), False)

    def prepare_input_mask(self, canvas_size):
        lengths = list()
        for c in canvas_size:
            lengths.append(c * c)
        attention_mask = torch.tril(torch.ones([sum(lengths), sum(lengths)]))
        curDiag = 0
        for l in lengths:
            attention_mask[curDiag:curDiag + l, curDiag:curDiag + l] = 1
            curDiag += l
        return attention_mask

    def random_pos_embed(self, bs, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        result = list()
        for i in range(bs):
            random_up = random.randint(0, H - h)
            random_left = random.randint(0, W - w)
            result.append(pos_embed[random_up:random_up + h, random_left:random_left + w].reshape(h * w, -1))
        return torch.stack(result)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start + h, left_start:left_start + w].reshape(h * w, -1)

    def forward(self, all_forwards_for_residual, cap_pooled):
        if self.training:
            if not isinstance(all_forwards_for_residual, list):
                raise RuntimeError('The given training input is not a list.')
            results = list()
            total = list()
            for level, current in enumerate(all_forwards_for_residual):
                if level == 0:
                    if current is not None:
                        raise RuntimeError('The first level input should be None.')
                    bs, _, h, w = all_forwards_for_residual[1].shape
                    h, w = 1, 1
                    if self.training:
                        selected_pos_embed = self.center_pos_embed(h, w)
                    else:
                        selected_pos_embed = self.center_pos_embed(h, w)
                    current = selected_pos_embed.expand(bs, h * w, -1)
                    current = selected_pos_embed + self.cap_to_first_token(cap_pooled)[:, None, ...]
                else:
                    bs, _, h, w = current.shape
                    current = self.input_transform(current.permute(0, 2, 3, 1).reshape(bs, h * w, -1))
                level_emb = self.level_indicator_pos_embed[level]
                current = current + level_emb
                total.append(current)
            total = torch.cat(total, dim=1)
            results = self.model(total, self.input_mask, cap_pooled)
            return results
        else:
            if not isinstance(all_forwards_for_residual, tuple):
                raise RuntimeError('The given training input is not a tuple.')
            current, level = all_forwards_for_residual
            if level == 0:
                h, w = 1, 1
                bs, dim = cap_pooled.shape
                selected_pos_embed = self.center_pos_embed(h, w)
                current = selected_pos_embed.expand(bs, h * w, self.token_dim)
                current = current + self.cap_to_first_token(cap_pooled)[:, None, ...]
                current = current.permute(0, 2, 1).reshape(bs, self.token_dim, h, w).contiguous()
                bs, _, h, w = current.shape
                level_emb = self.level_indicator_pos_embed[level]
                current = current.permute(0, 2, 3, 1).reshape(bs, h * w, -1) + level_emb
                logits = self.model(current, None, cap_pooled)
                return logits.argmax(dim=-1)
            else:
                current, level = all_forwards_for_residual
                current_feat = current[-1]
                bs, _, h, w = current_feat.shape
                level_emb = self.level_indicator_pos_embed[level]
                current_feat = current_feat.permute(0, 2, 3, 1).reshape(bs, h * w, -1) + level_emb
                logits = self.model(current_feat, None, cap_pooled)
                return logits.argmax(dim=-1)


class GeneratorV3SelfAttention(nn.Module):

    def __init__(self, channel: 'int', k: 'int', size: 'List[int]', denseNorm: 'bool', loadFrom: 'str', qk_norm: 'bool', norm_eps: 'float', *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, size, denseNorm)
        self.next_residual_predictor = checkpoint_wrapper(AnyRes_L(size[::-1], [[k, channel] for _ in size], qk_norm=qk_norm, norm_eps=norm_eps))
        logging.debug('Created any-res transformer.')
        self.class_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(len(IMAGENET2012_LABELS), self.next_residual_predictor.hidden_size), std=math.sqrt(2 / (5 * self.next_residual_predictor.hidden_size))))
        decoders = list()
        dequantizers = list()
        codebook = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, k, self.next_residual_predictor.hidden_size), std=math.sqrt(2 / (5 * self.next_residual_predictor.hidden_size))))
        lastSize = size[0] * 2
        for i, thisSize in enumerate(size):
            if thisSize == lastSize // 2:
                dequantizer = _multiCodebookDeQuantization(codebook)
                restoreHead = pixelShuffle3x3(self.next_residual_predictor.hidden_size, self.next_residual_predictor.hidden_size, 2)
            elif thisSize == lastSize:
                dequantizer = _multiCodebookDeQuantization(codebook)
                restoreHead = conv3x3(self.next_residual_predictor.hidden_size, self.next_residual_predictor.hidden_size)
            else:
                raise ValueError('The given size sequence does not half or equal to from left to right.')
            lastSize = thisSize
            decoders.append(restoreHead)
            dequantizers.append(dequantizer)
        self._decoders: 'nn.ModuleList' = nn.ModuleList(decoders)
        self._dequantizers: 'nn.ModuleList' = nn.ModuleList(dequantizers)
        self.init_weights(self.next_residual_predictor.hidden_size)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items() if '_lpips' not in k})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        self.compressor.eval()

    def residual_forward(self, code: 'torch.Tensor', formerLevel: 'torch.Tensor', level: 'int'):
        if formerLevel is None and level > 0:
            raise RuntimeError('For reconstruction after level-0, you should provide not None formerLevel as input.')
        if formerLevel is not None and level == 0:
            raise RuntimeError('For reconstruction at level-0, you should provide None formerLevel as input.')
        decoder, dequantizer = self._decoders[-(level + 1)], self._dequantizers[-(level + 1)]
        quantized = dequantizer.decode(code)
        return decoder(quantized + formerLevel) if formerLevel is not None else decoder(quantized)

    def init_weights(self, hidden_size, init_adaln=0.5, init_adaln_gamma=1e-05, init_head=0.02, init_std=-1, conv_std_or_gain=0.02):
        if init_std < 0:
            init_std = (2 / (5 * hidden_size)) ** 0.5
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()
        if init_head >= 0:
            self.next_residual_predictor.model.final_layer.linear.weight.data.mul_(init_head)
            self.next_residual_predictor.model.final_layer.linear.bias.data.zero_()
        self.next_residual_predictor.model.final_layer.adaLN_modulation[-1].weight.data.mul_(init_adaln)
        self.next_residual_predictor.model.final_layer.adaLN_modulation[-1].bias.data.mul_(init_adaln_gamma)
        depth = self.next_residual_predictor.depth
        for block_idx, transformerBlock in enumerate(self.next_residual_predictor.model.blocks):
            transformerBlock: 'TransformerBlock'
            transformerBlock.attention.wo.weight.data.div_(math.sqrt(2 * depth))
            transformerBlock.ffn.w2.weight.data.div_(math.sqrt(2 * depth))
        self.next_residual_predictor.model.adaLN_modulation[-1].weight.data.mul_(init_adaln)
        self.next_residual_predictor.model.adaLN_modulation[-1].bias.data.mul_(init_adaln_gamma)

    def train(self, mode: 'bool'=True):
        retValue = super().train(mode)
        self.compressor.eval()
        return retValue

    def forward(self, image, condition: 'torch.Tensor'):
        if self.training:
            with torch.autocast('cuda', enabled=False):
                with torch.no_grad():
                    codes = self.compressor.encode(image.float())
                all_forwards_for_residual = list()
                formerLevel = None
                for level, code in enumerate(codes[:-1]):
                    all_forwards_for_residual.append(self.residual_forward(code, formerLevel, level))
                    formerLevel = all_forwards_for_residual[-1]
            codes = [c.squeeze(1) for c in codes]
            all_forwards_for_residual = [x for x in all_forwards_for_residual]
            rawPredictions = self.next_residual_predictor([None, *all_forwards_for_residual], self.class_pos_embed[condition])
            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx:curIdx + h * w]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append((h * w, F.cross_entropy(pre, gt, reduction='none', label_smoothing=0.1)))
                curIdx += h * w
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                restored = self.compressor.decode(restoredCodes)
            return [*predictions], sum([(hw * l).sum() for hw, l in loss]) / len(image) / (curIdx + 1), codes, restored, [l.mean() for _, l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                class_embed = self.class_pos_embed[condition]
                h, w = 1, 1
                bs, hidden_size = class_embed.shape
                first_level_token = self.next_residual_predictor((None, 0), class_embed)
                first_level_token = first_level_token.unsqueeze(dim=1)
                first_level_token = first_level_token.permute(0, 2, 1).reshape(bs, -1, h, w)
                first_scale_feat = self.compressor.residual_forward(first_level_token, None, 0)
                predictions = [first_level_token]
                input_feats = [first_scale_feat]
                former_level_feat = first_scale_feat.clone()
                for i in range(1, len(self.compressor.Codebooks)):
                    next_level_token = self.next_residual_predictor((input_feats, i), class_embed)
                    scale = int(math.sqrt(next_level_token.shape[-1]))
                    h, w = scale, scale
                    next_level_token = next_level_token.reshape(bs, h, w).unsqueeze(1)
                    next_scale_feat = self.compressor.residual_forward(next_level_token, former_level_feat, i)
                    former_level_feat = next_scale_feat.clone()
                    predictions.append(next_level_token)
                    input_feats.append(next_scale_feat)
                restored = self.compressor.decode(predictions)
                return predictions, restored


class GeneratorV3SelfAttentionNoAda(nn.Module):

    def __init__(self, channel: 'int', k: 'int', size: 'List[int]', denseNorm: 'bool', loadFrom: 'str', qk_norm: 'bool', norm_eps: 'float', *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, size, denseNorm)
        self.next_residual_predictor = AnyRes_L(size[::-1], [[codebook.shape[1], codebook.shape[2]] for codebook in self.compressor.Codebooks[::-1]], qk_norm=qk_norm, norm_eps=norm_eps)
        logging.debug('Created any-res transformer.')
        self.class_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.empty(len(IMAGENET2012_LABELS), self.next_residual_predictor.hidden_size), std=math.sqrt(2 / 5)))
        self.init_weights(self.next_residual_predictor.hidden_size)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items() if '_lpips' not in k})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        self.compressor.eval()

    def init_weights(self, hidden_size, init_adaln=0.5, init_adaln_gamma=1e-05, init_head=0.02, init_std=-1, conv_std_or_gain=0.02):
        if init_std < 0:
            init_std = (2 / (5 * hidden_size)) ** 0.5
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()
        if init_head >= 0:
            self.next_residual_predictor.model.final_layer.linear.weight.data.mul_(init_head)
            self.next_residual_predictor.model.final_layer.linear.bias.data.zero_()
        depth = self.next_residual_predictor.depth
        for block_idx, transformerBlock in enumerate(self.next_residual_predictor.model.blocks):
            transformerBlock: 'TransformerBlock'
            transformerBlock.attention.wo.weight.data.div_(math.sqrt(2 * depth))
            transformerBlock.ffn.w2.weight.data.div_(math.sqrt(2 * depth))

    def train(self, mode: 'bool'=True):
        retValue = super().train(mode)
        self.compressor.eval()
        return retValue

    def forward(self, image, condition: 'torch.Tensor'):
        if self.training:
            with torch.no_grad():
                all_forwards_for_residual = list()
                codes = self.compressor.encode(image)
                formerLevel = None
                for level, code in enumerate(codes[:-1]):
                    all_forwards_for_residual.append(self.compressor.residual_forward(code, formerLevel, level))
                    formerLevel = all_forwards_for_residual[-1]
            codes = [c.squeeze(1) for c in codes]
            rawPredictions = self.next_residual_predictor([None, *all_forwards_for_residual], self.class_pos_embed[condition])
            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx:curIdx + h * w]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append(F.cross_entropy(pre, gt, reduction='none', label_smoothing=0.1))
                curIdx += h * w
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            with torch.no_grad():
                restored = self.compressor.decode(restoredCodes)
            return [*predictions], sum([l.sum() for l in loss]) / len(image), codes, restored, [l.mean() for l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                class_embed = self.class_pos_embed[condition]
                h, w = 1, 1
                bs, hidden_size = class_embed.shape
                first_level_token = self.next_residual_predictor((None, 0), class_embed)
                first_level_token = first_level_token.unsqueeze(dim=1)
                first_level_token = first_level_token.permute(0, 2, 1).reshape(bs, -1, h, w)
                first_scale_feat = self.compressor.residual_forward(first_level_token, None, 0)
                predictions = [first_level_token]
                input_feats = [first_scale_feat]
                former_level_feat = first_scale_feat.clone()
                for i in range(1, len(self.compressor.Codebooks)):
                    next_level_token = self.next_residual_predictor((input_feats, i), class_embed)
                    scale = int(math.sqrt(next_level_token.shape[-1]))
                    h, w = scale, scale
                    next_level_token = next_level_token.reshape(bs, h, w).unsqueeze(1)
                    next_scale_feat = self.compressor.residual_forward(next_level_token, former_level_feat, i)
                    former_level_feat = next_scale_feat.clone()
                    predictions.append(next_level_token)
                    input_feats.append(next_scale_feat)
                restored = self.compressor.decode(predictions)
                return predictions, restored


class AdaLNBeforeHead(nn.Module):

    def __init__(self, C, D, norm_layer):
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: 'torch.Tensor', cond_BD: 'torch.Tensor'):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False, scale_by_keep: 'bool'=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):

    def __init__(self, drop_prob: 'float'=0.0, scale_by_keep: 'bool'=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'(drop_prob=...)'


dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias, activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0, heuristic=0, process_group=None))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) ->str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):

    def __init__(self, block_idx, embed_dim=768, num_heads=12, attn_drop=0.0, proj_drop=0.0, attn_l2_norm=False, flash_if_available=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: 'float' = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: 'bool'):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_bias):
        B, L, C = x.shape
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q, k, v, attn_bias=None if attn_bias is None else attn_bias.expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) ->str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):

    def __init__(self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: 'bool', norm_layer, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, attn_l2_norm=False, flash_if_available=False, fused_if_available=True):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim ** 0.5)
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        self.fused_add_norm_fn = None

    def forward(self, x, cond_BD, attn_bias):
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path(self.attn(self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        x = x + self.drop_path(self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
        return x

    def extra_repr(self) ->str:
        return f'shared_aln={self.shared_aln}'


class SharedAdaLin(nn.Linear):

    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)


def gumbel_softmax_with_rng(logits: 'torch.Tensor', tau: 'float'=1, hard: 'bool'=False, eps: 'float'=1e-10, dim: 'int'=-1, rng: 'torch.Generator'=None) ->torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def sample_with_top_k_top_p_(logits_BlV: 'torch.Tensor', top_k: 'int'=0, top_p: 'float'=0.0, rng=None, num_samples=1) ->torch.Tensor:
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= 1 - top_p
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


class VAR(nn.Module):

    def __init__(self, codebook_size, num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_eps=1e-06, shared_aln=False, cond_drop_rate=0.1, attn_l2_norm=False, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), flash_if_available=True, fused_if_available=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = codebook_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1
        self.patch_nums: 'Tuple[int]' = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=torch.cuda.current_device())
        self.word_embed = nn.Linear(self.Cvae, self.C)
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=torch.cuda.current_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([AdaLNSelfAttn(cond_dim=self.D, shared_aln=shared_aln, block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1], attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available) for block_idx in range(depth)])
        fused_add_norm_fns = [(b.fused_add_norm_fn is not None) for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        None
        d: 'torch.Tensor' = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: 'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]', cond_BD: 'Optional[torch.Tensor]'):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(self, B: 'int', label_B: 'Optional[Union[int, torch.LongTensor]]', g_seed: 'Optional[int]'=None, cfg=1.5, top_k=0, top_p=0.0, more_smooth=False) ->torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)
        for b in self.blocks:
            b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    def forward(self, label_B: 'torch.LongTensor', x_BLCv_wo_first_l: 'torch.Tensor') ->torch.Tensor:
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC
        cond_BD_or_gss = cond_BD_or_gss
        attn_bias = attn_bias
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-05, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5
        None
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: 'AdaLNSelfAttn'
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-05)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class GeneratorVAR(nn.Module):

    def __init__(self, channel: 'int', k: 'int', size: 'List[int]', denseNorm: 'bool', loadFrom: 'str', **var_args):
        super().__init__()
        self.compressor = Neon(channel, k, size, denseNorm)
        depth = 24
        self.next_residual_predictor: 'VAR' = checkpoint_wrapper(VAR((8, 4096), len(IMAGENET2012_LABELS), depth=depth, embed_dim=1536, num_heads=16, norm_eps=1e-06, attn_l2_norm=True, patch_nums=size[::-1], drop_path_rate=0.1 * depth / 24))
        logging.debug('Created any-res transformer.')
        self.next_residual_predictor.init_weights(init_adaln=0.5, init_adaln_gamma=1e-05, init_head=0.02, init_std=-1)
        self.input_transform = nn.Identity()
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items() if '_lpips' not in k})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)
        self.compressor.eval()

    def residual_forward(self, code: 'torch.Tensor', formerLevel: 'torch.Tensor', level: 'int'):
        if formerLevel is None and level > 0:
            raise RuntimeError('For reconstruction after level-0, you should provide not None formerLevel as input.')
        if formerLevel is not None and level == 0:
            raise RuntimeError('For reconstruction at level-0, you should provide None formerLevel as input.')
        decoder, dequantizer = self._decoders[-(level + 1)], self._dequantizers[-(level + 1)]
        quantized = dequantizer.decode(code)
        return decoder(quantized + formerLevel) if formerLevel is not None else decoder(quantized)

    def train(self, mode: 'bool'=True):
        retValue = super().train(mode)
        self.compressor.eval()
        return retValue

    def forward(self, image, condition: 'torch.Tensor'):
        if self.training:
            with torch.autocast('cuda', enabled=False):
                with torch.no_grad():
                    codes = self.compressor.encode(image.float())
                    all_forwards_for_residual = list()
                    formerLevel = None
                    for level, code in enumerate(codes[:-1]):
                        all_forwards_for_residual.append(self.compressor.residual_forward(code, formerLevel, level))
                        formerLevel = all_forwards_for_residual[-1]
            codes = [c.squeeze(1) for c in codes]
            new_all_forwards_for_residual = list()
            for x in all_forwards_for_residual:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n, h * w, -1)
                new_all_forwards_for_residual.append(x)
            new_all_forwards_for_residual = torch.cat(new_all_forwards_for_residual, 1)
            new_all_forwards_for_residual.requires_grad_()
            rawPredictions = self.next_residual_predictor(condition, self.input_transform(new_all_forwards_for_residual))
            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx:curIdx + h * w]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append((h * w, F.cross_entropy(pre, gt, reduction='none', label_smoothing=0.0)))
                curIdx += h * w
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                restored = self.compressor.decode(restoredCodes)
            return [*predictions], sum([l.sum() for hw, l in loss]) / len(image), codes, restored, [l.mean() for _, l in loss]
        else:
            with torch.no_grad():
                device = next(self.parameters()).device
                class_embed = self.class_pos_embed[condition]
                h, w = 1, 1
                bs, hidden_size = class_embed.shape
                first_level_token = self.next_residual_predictor((None, 0), class_embed)
                first_level_token = first_level_token.unsqueeze(dim=1)
                first_level_token = first_level_token.permute(0, 2, 1).reshape(bs, -1, h, w)
                first_scale_feat = self.compressor.residual_forward(first_level_token, None, 0)
                predictions = [first_level_token]
                input_feats = [first_scale_feat]
                former_level_feat = first_scale_feat.clone()
                for i in range(1, len(self.compressor.Codebooks)):
                    next_level_token = self.next_residual_predictor((input_feats, i), class_embed)
                    scale = int(math.sqrt(next_level_token.shape[-1]))
                    h, w = scale, scale
                    next_level_token = next_level_token.reshape(bs, h, w).unsqueeze(1)
                    next_scale_feat = self.compressor.residual_forward(next_level_token, former_level_feat, i)
                    former_level_feat = next_scale_feat.clone()
                    predictions.append(next_level_token)
                    input_feats.append(next_scale_feat)
                restored = self.compressor.decode(predictions)
                return predictions, restored


class Phi(nn.Conv2d):

    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiNonShared(nn.ModuleList):

    def __init__(self, qresi: 'List'):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: 'float') ->Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) ->str:
        return f'ticks={self.ticks}'


class PhiPartiallyShared(nn.Module):

    def __init__(self, qresi_ls: 'nn.ModuleList'):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: 'float') ->Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) ->str:
        return f'ticks={self.ticks}'


class PhiShared(nn.Module):

    def __init__(self, qresi: 'Phi'):
        super().__init__()
        self.qresi: 'Phi' = qresi

    def __getitem__(self, _) ->Phi:
        return self.qresi


__initialized = False


def initialized():
    return __initialized


class VectorQuantizer2(nn.Module):

    def __init__(self, vocab_size, Cvae, using_znorm, beta: 'float'=0.25, default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4):
        super().__init__()
        self.vocab_size: 'int' = vocab_size
        self.Cvae: 'int' = Cvae
        self.using_znorm: 'bool' = using_znorm
        self.v_patch_nums: 'Tuple[int]' = v_patch_nums
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-06 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-06 else nn.Identity())
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-06 else nn.Identity()) for _ in range(share_quant_resi)]))
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        self.beta: 'float' = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        self.prog_si = -1

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) ->str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    def forward(self, f_BChw: 'torch.Tensor', ret_usages=False) ->Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        with torch.amp.autocast(enabled=False):
            mean_vq_loss: 'torch.Tensor' = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums):
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if si != SN - 1 else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if si != SN - 1 else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if initialized():
                        handler = tdist.all_reduce(hit_V, async_op=True)
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if si != SN - 1 else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                if self.training and initialized():
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        if ret_usages:
            usages = [((self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100) for si, pn in enumerate(self.v_patch_nums)]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    def embed_to_fhat(self, ms_h_BChw: 'List[torch.Tensor]', all_to_max_scale=True, last_one=False) ->Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)
        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self, f_BChw: 'torch.Tensor', to_fhat: 'bool', v_patch_nums: 'Optional[Sequence[Union[int, Tuple[int, int]]]]'=None) ->List[Union[torch.Tensor, torch.LongTensor]]:
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        f_hat_or_idx_Bl: 'List[torch.Tensor]' = []
        patch_hws = [((pn, pn) if isinstance(pn, int) else (pn[0], pn[1])) for pn in v_patch_nums or self.v_patch_nums]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'patch_hws[-1]={patch_hws[-1]!r} != (H={H!r}, W={W!r})'
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            if 0 <= self.prog_si < si:
                break
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if si != SN - 1 else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if si != SN - 1 else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))
        return f_hat_or_idx_Bl

    def idxBl_to_var_input(self, gt_ms_idx_Bl: 'List[torch.Tensor]') ->torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: 'int' = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or 0 <= self.prog_si - 1 < si:
                break
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return next_scales

    def get_next_autoregressive_input(self, si: 'int', SN: 'int', f_hat: 'torch.Tensor', h_BChw: 'torch.Tensor') ->Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Upsample2x(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-06, affine=True)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-06 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** -0.5
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()
        k = k.view(B, C, H * W).contiguous()
        w = torch.bmm(q, k).mul_(self.w_ratio)
        w = F.softmax(w, dim=2)
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()
        h = torch.bmm(v, w)
        h = h.view(B, C, H, W).contiguous()
        return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):

    def __init__(self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.0, in_channels=3, z_channels, double_z=False, using_sa=True, using_mid_sa=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


class Decoder(nn.Module):

    def __init__(self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.0, in_channels=3, z_channels, using_sa=True, using_mid_sa=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


class VQVAE(nn.Module):

    def __init__(self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0, beta=0.25, using_znorm=False, quant_conv_ks=3, quant_resi=0.5, share_quant_resi=4, default_qresi_counts=0, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), test_mode=True):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        ddconfig = dict(dropout=dropout, ch=ch, z_channels=z_channels, in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, using_sa=True, using_mid_sa=True)
        ddconfig.pop('double_z', None)
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.quantize: 'VectorQuantizer2' = VectorQuantizer2(vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta, default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi)
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def forward(self, inp, ret_usages=False):
        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss

    def fhat_to_img(self, f_hat: 'torch.Tensor'):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(self, inp_img_no_grad: 'torch.Tensor', v_patch_nums: 'Optional[Sequence[Union[int, Tuple[int, int]]]]'=None) ->List[torch.LongTensor]:
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(self, ms_idx_Bl: 'List[torch.Tensor]', same_shape: 'bool', last_one=False) ->Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_img(self, ms_h_BChw: 'List[torch.Tensor]', all_to_max_scale: 'bool', last_one=False) ->Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]

    def img_to_reconstructed_img(self, x, v_patch_nums: 'Optional[Sequence[Union[int, Tuple[int, int]]]]'=None, last_one=False) ->List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    def load_state_dict(self, state_dict: 'Dict[str, Any]', strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


class _logExpMinusOne(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return (x.exp() - 1 + torch.finfo(x.dtype).eps).log()

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        passThroughIf = x > bound
        remaining = ~passThroughIf
        return passThroughIf * grad_output + remaining * grad_output * x.exp() / (x.exp() - 1 + torch.finfo(x.dtype).eps), None


class LogExpMinusOne(nn.Module):

    def __init__(self):
        super().__init__()
        eps = torch.tensor(torch.finfo(torch.float).eps)
        self.register_buffer('_bound', ((1 + eps) / eps).log())

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return _logExpMinusOne.apply(x, self._bound)


class NonLocalBlock(nn.Module):

    def __init__(self, N, groups=1):
        super().__init__()
        self._c = N // 2
        self._q = conv1x1(N, N // 2, groups=groups)
        self._k = conv1x1(N, N // 2, groups=groups)
        self._v = conv1x1(N, N // 2, groups=groups)
        self._z = conv1x1(N // 2, N, groups=groups)

    def forward(self, x: 'torch.Tensor'):
        n, c, h, w = x.shape
        hw = h * w
        scale = sqrt(hw)
        q = self._q(x).reshape(n, self._c, hw)
        k = self._k(x).reshape(n, self._c, hw)
        v = self._v(x).reshape(n, self._c, hw).permute(0, 2, 1)
        qkLogits = torch.matmul(q.transpose(-1, -2), k) / scale
        randomMask = torch.rand((n, hw, hw), device=qkLogits.device) < 0.1
        qkLogits = qkLogits.masked_fill(randomMask, -1000000000.0)
        weights = torch.softmax(qkLogits, -1)
        z = torch.matmul(weights, v).permute(0, 2, 1).reshape(n, self._c, h, w)
        z = self._z(z)
        return x + z


_planckian_coeffs = torch.tensor([[0.6743, 0.4029, 0.0013], [0.6281, 0.4241, 0.1665], [0.5919, 0.4372, 0.2513], [0.5623, 0.4457, 0.3154], [0.5376, 0.4515, 0.3672], [0.5163, 0.4555, 0.4103], [0.4979, 0.4584, 0.4468], [0.4816, 0.4604, 0.4782], [0.4672, 0.4619, 0.5053], [0.4542, 0.463, 0.5289], [0.4426, 0.4638, 0.5497], [0.432, 0.4644, 0.5681], [0.4223, 0.4648, 0.5844], [0.4135, 0.4651, 0.599], [0.4054, 0.4653, 0.6121], [0.398, 0.4654, 0.6239], [0.3911, 0.4655, 0.6346], [0.3847, 0.4656, 0.6444], [0.3787, 0.4656, 0.6532], [0.3732, 0.4656, 0.6613], [0.368, 0.4655, 0.6688], [0.3632, 0.4655, 0.6756], [0.3586, 0.4655, 0.682], [0.3544, 0.4654, 0.6878], [0.3503, 0.4653, 0.6933], [0.5829, 0.4421, 0.2288], [0.551, 0.4514, 0.2948], [0.5246, 0.4576, 0.3488], [0.5021, 0.4618, 0.3941], [0.4826, 0.4646, 0.4325], [0.4654, 0.4667, 0.4654], [0.4502, 0.4681, 0.4938], [0.4364, 0.4692, 0.5186], [0.424, 0.47, 0.5403], [0.4127, 0.4705, 0.5594], [0.4023, 0.4709, 0.5763], [0.3928, 0.4713, 0.5914], [0.3839, 0.4715, 0.6049], [0.3757, 0.4716, 0.6171], [0.3681, 0.4717, 0.6281], [0.3609, 0.4718, 0.638], [0.3543, 0.4719, 0.6472], [0.348, 0.4719, 0.6555], [0.3421, 0.4719, 0.6631], [0.3365, 0.4719, 0.6702], [0.3313, 0.4719, 0.6766], [0.3263, 0.4719, 0.6826], [0.3217, 0.4719, 0.6882]])


_planckian_coeffs_ratio = torch.stack((_planckian_coeffs[:, 0] / _planckian_coeffs[:, 1], _planckian_coeffs[:, 2] / _planckian_coeffs[:, 1]), 1)


class RandomPlanckianJitter(nn.Module):
    pl: 'torch.Tensor'

    def __init__(self, p: 'float'=0.5) ->None:
        super().__init__()
        self.register_buffer('pl', _planckian_coeffs_ratio)
        self.p = p

    def forward(self, x: 'torch.Tensor'):
        needsApply = torch.rand(x.shape[0]) < self.p
        coeffs = self.pl[torch.randint(len(self.pl), (needsApply.sum(),))]
        r_w = coeffs[:, 0][..., None, None]
        b_w = coeffs[:, 1][..., None, None]
        willbeApplied = x[needsApply]
        willbeApplied[..., 0, :, :].mul_(r_w)
        willbeApplied[..., 2, :, :].mul_(b_w)
        return x.clamp_(0.0, 1.0)


def linearToSrgb(x: 'torch.Tensor'):
    return torch.where(x < 0.04045, x / 12.92, torch.pow(torch.abs(x + 0.055) / 1.055, 2.4))


def randomGamma(x: 'torch.Tensor', randomGammas: 'torch.Tensor'):
    x = torch.pow(x.clamp_(0), randomGammas)
    return x.clamp_(0.0, 1.0)


def srgbToLinear(x: 'torch.Tensor'):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * torch.pow(torch.abs(x), 1 / 2.4) - 0.055)


class RandomGamma(nn.Module):
    _fns = [srgbToLinear, linearToSrgb, lambda x: randomGamma(x, torch.rand((), device=x.device) * 1.95 + 0.05), lambda x: x]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return random.choice(self._fns)(x)


class DeTransform(nn.Module):
    _eps = 0.001
    _maxVal = 255

    def __init__(self, minValue: 'float'=-1.0, maxValue: 'float'=1.0):
        super().__init__()
        self._min = float(minValue)
        self._max = float(maxValue)

    def forward(self, x):
        x = (x - self._min) / (self._max - self._min)
        return (x * (self._maxVal + 1.0 - self._eps)).clamp(0.0, 255.0).byte()


class RandomHorizontalFlip(nn.Module):
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    Args:
        p (float): probability of an image being flipped.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        flipped = torch.rand(tensor.shape[0]) < self.p
        tensor[flipped].copy_(tensor[flipped].flip(-1))
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(nn.Module):
    """Applies the :class:`~torchvision.transforms.RandomVerticalFlip` transform to a batch of images.
    Args:
        p (float): probability of an image being flipped.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        flipped = torch.rand(tensor.shape[0]) < self.p
        tensor[flipped].copy_(tensor[flipped].flip(-2))
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAutocontrast(torch.nn.Module):
    """Autocontrast the pixels of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be autocontrasted.
        Returns:
            PIL Image or Tensor: Randomly autocontrasted image.
        """
        picked = torch.rand(img.shape[0]) < self.p
        img[picked].copy_(F.autocontrast(img[picked]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class Masking(nn.Module):

    def __init__(self):
        super().__init__()
        self._categorical = Categorical(torch.Tensor([0.85, 0.15]))

    def forward(self, images: 'torch.Tensor'):
        n, _, h, w = images.shape
        zeros = torch.zeros_like(images)
        mask = self._categorical.sample((n, 1, h, w)).byte()
        return (mask == 0) * images + (mask == 1) * zeros


class PatchWiseErasing(nn.Module):
    permutePattern: 'torch.Tensor'

    def __init__(self, grids: 'Tuple[int, int]'=(16, 16), p: 'float'=0.75) ->None:
        super().__init__()
        self.p = p
        self.grids = 1, 1, *grids
        permutePattern = torch.ones(grids).flatten()
        permuteAmount = round(permutePattern.numel() * p)
        permutePattern[:permuteAmount] = 0
        self.register_buffer('permutePattern', permutePattern)

    def forward(self, x: 'torch.Tensor'):
        shape = x.shape
        h, w = shape[-2], shape[-1]
        randIdx = torch.randperm(len(self.permutePattern), device=x.device)
        self.permutePattern.copy_(self.permutePattern[randIdx])
        permutePattern = self.permutePattern.reshape(self.grids)
        eraseArea = tf.interpolate(permutePattern, (h, w))
        return x * eraseArea


def ssim(X, Y, win, data_range=255, sizeAverage=True, K=(0.01, 0.03), nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    assert len(X.shape) in (4, 5), f'Input images should be 4-d or 5-d tensors, but got {X.shape}'
    assert X.type() == Y.type(), 'Input images should have the same dtype.'
    win_size = win.shape[-1]
    assert win_size % 2 == 1, 'Window size should be odd.'
    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    if sizeAverage:
        return ssim_per_channel.mean()
    return ssim_per_channel.mean(1)


class Ssim(nn.Module):

    def __init__(self, data_range=255, sizeAverage=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """
        super().__init__()
        self.register_buffer('win', _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims))
        self.sizeAverage = sizeAverage
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return 1.0 - ssim(X, Y, self.win, data_range=self.data_range, sizeAverage=self.sizeAverage, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


class Decibel(nn.Module):

    def __init__(self, upperBound: 'float'):
        super().__init__()
        self._upperBound = upperBound ** 2

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return -10 * (x / self._upperBound).log10()


class EMATracker(nn.Module):

    def __init__(self, size: 'Union[torch.Size, List[int], Tuple[int, ...]]', momentum: 'float'=0.9):
        super().__init__()
        self._shadow: 'torch.Tensor'
        self._decay = 1 - momentum
        self.register_buffer('_shadow', torch.empty(size) * torch.nan)

    @torch.no_grad()
    def forward(self, x: 'torch.Tensor'):
        if torch.all(torch.isnan(self._shadow)):
            self._shadow.copy_(x)
            return self._shadow
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaLNBeforeHead,
     lambda: ([], {'C': 4, 'D': 4, 'norm_layer': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 64, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (AlignedCrop,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttentionBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicRate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DeTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Decibel,
     lambda: ([], {'upperBound': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Director,
     lambda: ([], {'channel': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Director5x5,
     lambda: ([], {'channel': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EncoderHead,
     lambda: ([], {'channel': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EncoderHead5x5,
     lambda: ([], {'channel': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFN,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FinalLayer,
     lambda: ([], {'hidden_size': 4, 'prediction_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LogExpMinusOne,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LowerBound,
     lambda: ([], {'bound': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Masking,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocalBlock,
     lambda: ([], {'N': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonNegativeParametrizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PSNR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PatchWiseErasing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Phi,
     lambda: ([], {'embed_dim': 4, 'quant_resi': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PixelCNN,
     lambda: ([], {'m': 4, 'k': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProjLayer,
     lambda: ([], {'hidden_size': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomAutocontrast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (RandomGamma,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomHorizontalFlip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomPlanckianJitter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomVerticalFlip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'inChannels': 4, 'outChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlockMasked,
     lambda: ([], {'inChannels': 4, 'outChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'hidden_size': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (_multiCodebookDeQuantization,
     lambda: ([], {'codebook': torch.rand([4, 4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (_quantizerDecoder,
     lambda: ([], {'dequantizer': torch.nn.ReLU(), 'dequantizationHead': torch.nn.ReLU(), 'sideHead': torch.nn.ReLU(), 'restoreHead': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (_quantizerEncoder,
     lambda: ([], {'quantizer': torch.nn.ReLU(), 'dequantizer': torch.nn.ReLU(), 'latentStageEncoder': torch.nn.ReLU(), 'quantizationHead': torch.nn.ReLU(), 'latentHead': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_residulBlock,
     lambda: ([], {'act1': torch.nn.ReLU(), 'conv1': torch.nn.ReLU(), 'act2': torch.nn.ReLU(), 'conv2': torch.nn.ReLU(), 'skip': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

