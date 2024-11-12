
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


from typing import Union


import numpy as np


from numpy.typing import NDArray


from torch.utils.data import DataLoader


from typing import Optional


from typing import Sequence


import torch.nn as nn


from torch import Tensor


from torchvision.utils import make_grid


from functools import wraps


from scipy import ndimage


import inspect


from abc import ABC


from abc import abstractmethod


import matplotlib.pyplot as plt


from torch.utils.data import Dataset


import random


from typing import TypedDict


from matplotlib import pyplot as plt


import warnings


from torch.utils.cpp_extension import load


from collections import OrderedDict


import torch.nn.functional as F


from typing import Callable


import torch.optim.lr_scheduler as sched


from functools import partial


from typing import Any


from typing import Iterable


from typing import Type


from typing import TypeVar


from torch import nn


from torch import optim


from sklearn.decomposition import PCA


from torch.optim.lr_scheduler import _LRScheduler


from enum import Enum


from typing import Dict


from typing import List


from typing import Tuple


from torchvision.transforms import InterpolationMode


from torchvision.transforms import functional as F


import re


import collections


import copy


import time


from typing import Generator


import torch.utils.data._utils.collate


class ChamferDistanceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)
        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1
            dist2 = dist2
            idx1 = idx1
            idx2 = idx2
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1
            gradxyz2 = gradxyz2
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):

    def __init__(self):
        super().__init__()
        if cd is None:
            raise RuntimeError(f'Chamfer Distance module unavailable')

    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


class DenseL1Error(nn.Module):
    """Dense L1 loss averaged over channels."""

    def forward(self, pred, target):
        return (pred - target).abs().mean(dim=1, keepdim=True)


class DenseL2Error(nn.Module):
    """Dense L2 distance."""

    def forward(self, pred, target):
        return (pred - target).pow(2).sum(dim=1, keepdim=True).clamp(min=ops.eps(pred)).sqrt()


class SSIMError(nn.Module):
    """Structural similarity error."""

    def __init__(self):
        super().__init__()
        self.pool: 'nn.Module' = nn.AvgPool2d(kernel_size=3, stride=1)
        self.refl: 'nn.Module' = nn.ReflectionPad2d(padding=1)
        self.eps1: 'float' = 0.01 ** 2
        self.eps2: 'float' = 0.03 ** 2

    def forward(self, pred: 'Tensor', target: 'Tensor') ->Tensor:
        """Compute the structural similarity error between two images.

        :param pred: (Tensor) (b, c, h, w) Predicted reconstructed images.
        :param target: (Tensor) (b, c, h, w) Target images to reconstruct.
        :return: (Tensor) (b, c, h, w) Structural similarity error.
        """
        x, y = self.refl(pred), self.refl(target)
        mu_x, mu_y = self.pool(x), self.pool(y)
        sig_x = self.pool(x ** 2) - mu_x ** 2
        sig_y = self.pool(y ** 2) - mu_y ** 2
        sig_xy = self.pool(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.eps1) * (2 * sig_xy + self.eps2)
        den = (mu_x ** 2 + mu_y ** 2 + self.eps1) * (sig_x + sig_y + self.eps2)
        loss = ((1 - num / den) / 2).clamp(min=0, max=1)
        return loss


class PhotoError(nn.Module):
    """Class for computing the photometric error.
    From Monodepth (https://arxiv.org/abs/1609.03677)

    The SSIMLoss can be deactivated by setting `weight_ssim=0`.
    The L1Loss can be deactivated by setting `weight_ssim=1`.
    Otherwise, the loss is a weighted combination of both.

    Attributes:
    :param weight_ssim: (float) Weight controlling the contribution of the SSIMLoss. L1 weight is `1 - ssim_weight`.
    """

    def __init__(self, weight_ssim: 'float'=0.85):
        super().__init__()
        if weight_ssim < 0 or weight_ssim > 1:
            raise ValueError(f'Invalid SSIM weight. ({weight_ssim} vs. [0, 1])')
        self.weight_ssim: 'float' = weight_ssim
        self.weight_l1: 'float' = 1 - self.weight_ssim
        self.ssim: 'Optional[nn.Module]' = SSIMError() if self.weight_ssim > 0 else None
        self.l1: 'Optional[nn.Module]' = DenseL1Error() if self.weight_l1 > 0 else None

    def forward(self, pred: 'Tensor', target: 'Tensor') ->Tensor:
        """Compute the photometric error between two images.

        :param pred: (Tensor) (b, c, h, w) Predicted reconstructed images.
        :param target: (Tensor) (b, c, h, w) Target images to reconstruct.
        :return: (Tensor) (b, 1, h, w) Photometric error.
        """
        b, _, h, w = pred.shape
        loss = pred.new_zeros((b, 1, h, w))
        if self.ssim:
            loss += self.weight_ssim * self.ssim(pred, target).mean(dim=1, keepdim=True)
        if self.l1:
            loss += self.weight_l1 * self.l1(pred, target)
        return loss


TensorDict = dict[Union[str, int], Tensor]


LossData = tuple[torch.Tensor, TensorDict]


class ReconstructionLoss(nn.Module):
    """Class to compute the reconstruction loss when synthesising new views.

    Contributions:
        - Min reconstruction error: From Monodepth2 (https://arxiv.org/abs/1806.01260)
        - Static pixel automasking: From Monodepth2 (https://arxiv.org/abs/1806.01260)
        - Explainability mask: From SfM-Learner (https://arxiv.org/abs/1704.07813)
        - Uncertainty mask: From Klodt (https://openaccess.thecvf.com/content_ECCV_2018/papers/Maria_Klodt_Supervising_the_new_ECCV_2018_paper.pdf)

    :param loss_name: (str) Loss type to use.
    :param use_min: (bool) If `True`, take the final loss as the minimum across all available views.
    :param use_automask: (bool) If `True`, mask pixels where the original support image has a lower loss than the warped counterpart.
    :param mask_name: (Optional[str]) Weighting mask used. {'explainability', 'uncertainty', None}
    """

    def __init__(self, loss_name: 'str'='ssim', use_min: 'bool'=False, use_automask: 'bool'=False, mask_name: 'Optional[str]'=None):
        super().__init__()
        self.loss_name = loss_name
        self.use_min = use_min
        self.use_automask = use_automask
        self.mask_name = mask_name
        if self.mask_name not in {'explainability', 'uncertainty', None}:
            raise ValueError(f'Invalid mask type: {self.mask_name}')
        self._photo = {'ssim': PhotoError(weight_ssim=0.85), 'l1': DenseL1Error(), 'l2': DenseL2Error()}[self.loss_name]

    def apply_mask(self, err: 'Tensor', mask: 'Optional[Tensor]'=None) ->Tensor:
        """Apply a weighting mask to a photometric loss error.

        :param err: (Tensor) (b, n, h, w) Photometric error to mask.
        :param mask: (Optional[Tensor]) (b, n, h, w) Optional weighting mask to apply.
        :return: (Tensor) (b, n, h, w) The weighted photometric error.
        """
        if self.mask_name and mask is None:
            raise ValueError('Must provide a "mask" when masking...')
        if self.mask_name == 'explainability':
            err *= mask
        elif self.mask_name == 'uncertainty':
            err = err * (-mask).exp() + mask
        return err

    def apply_automask(self, err: 'Tensor', source: 'Tensor', target: 'Tensor', mask: 'Optional[Tensor]'=None) ->tuple[Tensor, Tensor]:
        """Compute and apply an automask based on the identity reconstruction error.

        :param err: (Tensor) (b, 1, h, w) The photometric error for between target and warped support frames.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param source: (Optional[Tensor]) (*n, b, 3, h, w) Original support images.
        :param mask: (Optional[Tensor]) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (
            err: (Tensor) (b, 1, h, w) The automasked photometric error.
            automask: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by the automasking procedure.
        )
        """
        err_static = self.compute_photo(source, target, mask=mask)
        err_static += ops.eps(err_static) * torch.randn_like(err_static)
        err = torch.cat((err, err_static), dim=1)
        err, idxs = torch.min(err, dim=1, keepdim=True)
        automask = idxs == 0
        return err, automask

    def compute_photo(self, pred: 'Tensor', target: 'Tensor', mask: 'Optional[Tensor]'=None) ->Tensor:
        """Compute the dense photometric between multiple predictions and a single target.

        :param pred: (Tensor) (*n, b, 3, h, w) Synthesized warped support images.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param mask: (Optional[Tensor]) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (Tensor) (b, 1, h, w) The reduced photometric error.
        """
        if pred.ndim == 4:
            err = self._photo(pred, target)
        else:
            target = target[None].expand_as(pred)
            err = self._photo(pred.flatten(0, 1), target.flatten(0, 1))
            err = err.squeeze(1).unflatten(0, pred.shape[:2]).permute(1, 0, 2, 3)
        err = self.apply_mask(err, mask)
        err = err.min(dim=1, keepdim=True)[0] if self.use_min else err.mean(dim=1, keepdim=True)
        return err

    def forward(self, pred: 'Tensor', target: 'Tensor', source: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None) ->LossData:
        """Compute the reconstruction loss between two images.

        :param pred: (Tensor) (*n, b, 3, h, w) Synthesized warped support images.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param source: (Optional[Tensor]) (*n, b, 3, h, w) Original support images.
        :param mask: (Optional[Tensor]) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (
            loss: (Tensor) (,) Scalar loss.
            loss_dict: {
                (Optional) (If using automasking)
                automask: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by the automasking procedure.
            }
        )
        """
        ld = {}
        err = self.compute_photo(pred, target, mask)
        if self.use_automask:
            if source is None:
                raise ValueError('Must provide the original "source" images when automasking...')
            err, automask = self.apply_automask(err, source, target, mask)
            ld['automask'] = automask
        loss = err.mean()
        return loss, ld


def l1_loss(pred: 'Tensor', target: 'Tensor') ->Tensor:
    """Dense L1 loss."""
    loss = (pred - target).abs()
    return loss


def berhu_loss(pred: 'Tensor', target: 'Tensor', delta: 'float'=0.2, dynamic: 'bool'=True) ->Tensor:
    """Dense berHu loss.

    :param pred: (Tensor) Network prediction.
    :param target: (Tensor) Ground-truth target.
    :param delta: (float) Threshold above which the loss switches from L1.
    :param dynamic: (bool) If `True`, set threshold dynamically, using `delta` as the max error percentage.
    :return: (Tensor) The computed `berhu` loss.
    """
    diff = l1_loss(pred, target)
    delta = delta if not dynamic else delta * diff.max()
    diff_delta = (diff.pow(2) + delta.pow(2)) / (2 * delta + ops.eps(pred))
    loss = torch.where(diff <= delta, diff, diff_delta)
    return loss


def log_l1_loss(pred: 'Tensor', target: 'Tensor') ->Tensor:
    """Dense Log L1 loss."""
    loss = (1 + l1_loss(pred, target)).log()
    return loss


class RegressionLoss(nn.Module):
    """Class implementing a supervised regression loss.

    NOTE: The DepthHints automask is not computed here. Instead, we rely on the `MonoDepthModule` to compute it.
    Probably not the best way of doing it, but it keeps this loss clean...

    Contributions:
        - Virtual stereo consistency: From Monodepth (https://arxiv.org/abs/1609.03677)
        - Proxy berHu regression: From Kuznietsov (https://arxiv.org/abs/1702.02706)
        - Proxy LogL1 regression: From Depth Hints (https://arxiv.org/abs/1909.09051)
        - Proxy loss automasking: From Depth Hints/Monodepth2 (https://arxiv.org/abs/1909.09051)

    :param loss_name: (str) Loss type to use. {l1, log_l1, berhu}
    :param use_automask: (bool) If `True`, use DepthHints automask based on the pred/hints errors.
    """

    def __init__(self, loss_name: 'str'='berhu', use_automask: 'bool'=False):
        super().__init__()
        self.loss_name = loss_name
        self.use_automask = use_automask
        self.criterion = {'l1': l1_loss, 'log_l1': log_l1_loss, 'berhu': berhu_loss}[self.loss_name]

    def forward(self, pred: 'Tensor', target: 'Tensor', mask: 'Optional[Tensor]'=None) ->LossData:
        if mask is None:
            mask = torch.ones_like(target)
        err = mask * self.criterion(pred, target)
        loss = err.sum() / mask.sum()
        return loss, {'err_regr': err, 'mask_regr': mask}


ACT = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(inplace=True), 'none': nn.Identity(), None: nn.Identity()}


def conv3x3(in_ch: 'int', out_ch: 'int', bias: 'bool'=True) ->nn.Conv2d:
    """Layer to pad and convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=bias)


class DetailEmphasis(nn.Module):
    """Detail Emphasis Module.

    :param ch: (int) Number of input/output channels.
    """

    def __init__(self, ch: 'int'):
        super().__init__()
        self.conv = nn.Sequential(conv3x3(ch, ch), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = x + x * self.att(x)
        return x


class StructurePerception(nn.Module):
    """Self-attention Structure Perception Module."""

    def forward(self, x):
        b, c, h, w = x.shape
        value = x.view(b, c, -1)
        query = value
        key = value.permute(0, 2, 1)
        att = query @ key
        att = att.max(dim=-1, keepdim=True)[0] - att
        out = att.softmax(dim=-1) @ value
        out = x + out.view(b, c, h, w)
        return out


def conv_block(in_ch: 'int', out_ch: 'int') ->nn.Module:
    """Layer to perform a convolution followed by ELU."""
    return nn.Sequential(OrderedDict({'conv': conv3x3(in_ch, out_ch), 'act': nn.ELU(inplace=True)}))


class CADepthDecoder(nn.Module):
    """From CADepth (https://arxiv.org/abs/2112.13047)

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = conv_block(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)
            self.convs[f'detail_emphasis_{i}'] = DetailEmphasis(num_ch_in)
        for i in self.out_sc:
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.out_ch)
        self.structure_perception = StructurePerception()
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = ACT[self.out_act]

    def forward(self, enc_features):
        out = {}
        x = self.structure_perception(enc_features[-1])
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_{0}'](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                x += [enc_features[idx]]
            x = torch.cat(x, 1)
            x = self.convs[f'detail_emphasis_{i}'](x)
            x = self.convs[f'upconv_{i}_{1}'](x)
            if i in self.out_sc:
                out[i] = self.activation(self.convs[f'outconv_{i}'](x))
        return out


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block.

    :param ch: (int) Number of input/output channels.
    """

    def __init__(self, ch):
        super().__init__()
        self.query_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.key_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.value_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1, padding=0), nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.shape
        query = self.query_conv(x).flatten(-2, -1)
        key = self.key_conv(x).flatten(-2, -1).permute(0, 2, 1)
        value = self.value_conv(x).flatten(-2, -1)
        att = query @ key
        out = att.softmax(dim=-1) @ value
        out = out.view(b, c, h, w)
        return out


def get_discrete_bins(n: 'int', mode: 'str'='linear') ->Tensor:
    """Get the discretized disparity value depending on number of bins and quantization mode.

    All modes assume that we are quantizing sigmoid disparity, and therefore are in range [0, 1].
    Quantization modes:
        - linear: Evenly spaces out all bins.
        - exp: Spaces bins out exponentially, providing finer detail at low disparity values, ie higher depth values.

    :param n: (int) Number of bins to use.
    :param mode: (str) Quantization mode. {linear, exp}
    :return: (Tensor) (1, n, 1, 1) Computed discrete disparity bins.
    """
    bins = torch.arange(n) / n
    if mode == 'linear':
        pass
    elif mode == 'exp':
        max_depth = Tensor(200)
        bins = torch.exp(torch.log(max_depth) * (bins - 1))
    else:
        raise ValueError(f'Invalid discretization mode. "{mode}"')
    return bins.view(1, n, 1, 1)


class DDVNetDecoder(nn.Module):
    """From DDVNet (https://arxiv.org/abs/2003.13951)

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.num_bins = 128
        self.bins = nn.Parameter(get_discrete_bins(self.num_bins, mode='linear'))
        self.convs = OrderedDict()
        self.convs['att'] = SelfAttentionBlock(self.num_ch_enc[-1])
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = conv_block(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)
        for i in self.out_sc:
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.num_bins * self.out_ch)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = ACT[self.out_act]
        self.logits = {}

    def expected_disparity(self, logits: 'Tensor') ->Tensor:
        """Maps discrete disparity logits into the expected weighted disparity.

        :param logits: (Tensor) (b, n, h, w) Raw unnormalized predicted probabilities.
        :return: (Tensor) (b, 1, h, w) Expected disparity map.
        """
        probs = logits.softmax(dim=1)
        disp = (probs * self.bins).sum(dim=1, keepdim=True)
        return disp

    def argmax_disparity(self, logits: 'Tensor') ->Tensor:
        idx = logits.argmax(dim=1)
        one_hot = F.one_hot(idx, self.num_bins).permute(0, 3, 1, 2)
        disp = (one_hot * self.bins).sum(dim=1, keepdim=True)
        return disp

    def forward(self, enc_features: 'Sequence[Tensor]') ->dict[int, Tensor]:
        out = {}
        x = self.convs['att'](enc_features[-1])
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_{0}'](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                x += [enc_features[idx]]
            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)
            if i in self.out_sc:
                logits = self.convs[f'outconv_{i}'](x)
                self.logits[i] = logits
                out[i] = torch.cat([self.expected_disparity(l) for l in logits.chunk(self.out_ch, dim=1)], dim=1)
        return out


class ChannelAttention(nn.Module):
    """Channel Attention Module incorporating Squeeze & Exicitation.

    :param in_ch: (int) Number of input channels.
    :param ratio: (int) Channels reduction ratio in bottleneck.
    """

    def __init__(self, in_ch: 'int', ratio: 'int'=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_ch, in_ch // ratio, bias=False), nn.ReLU(inplace=True), nn.Linear(in_ch // ratio, in_ch, bias=False))
        self.init_weights()

    def init_weights(self):
        """Kaiming weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        att = self.avg_pool(x)
        att = self.fc(att.squeeze()).sigmoid()
        return x * att[..., None, None]


class AttentionBlock(nn.Module):
    """Attention Block incorporating channel attention.

    :param in_ch: (int) Number of input channels.
    :param skip_ch: (int) Number of channels in skip connection features.
    :param out_ch: (Optional[int]) Number of output channels.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    """

    def __init__(self, in_ch: 'int', skip_ch: 'int', out_ch: 'Optional[int]'=None, upsample_mode: 'str'='nearest'):
        super().__init__()
        self.in_ch = in_ch + skip_ch
        self.out_ch = out_ch or in_ch
        self.upsample_mode = upsample_mode
        self.layers = nn.Sequential(ChannelAttention(self.in_ch), conv3x3(self.in_ch, self.out_ch), nn.ReLU(inplace=True))

    def forward(self, x, x_skip):
        return self.layers(torch.cat((F.interpolate(x, scale_factor=2, mode=self.upsample_mode), x_skip), dim=1))


def upsample_block(in_ch: 'int', out_ch: 'int', upsample_mode: 'str'='nearest') ->nn.Module:
    """Layer to upsample the input by a factor of 2 without skip connections."""
    return nn.Sequential(conv_block(in_ch, out_ch), nn.Upsample(scale_factor=2, mode=upsample_mode), conv_block(out_ch, out_ch))


class DiffNetDecoder(nn.Module):
    """From DiffNet (https://arxiv.org/abs/2110.09482)

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_skip = self.num_ch_enc[idx]
                self.convs[f'upconv_{i}'] = AttentionBlock(num_ch_in, num_ch_skip, num_ch_out, self.upsample_mode)
            else:
                self.convs[f'upconv_{i}'] = upsample_block(num_ch_in, num_ch_out, self.upsample_mode)
        for i in range(4):
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.out_ch)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = ACT[self.out_act]

    def forward(self, enc_features):
        out = {}
        x = enc_features[-1]
        for i in range(4, -1, -1):
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                x = self.convs[f'upconv_{i}'](x, enc_features[idx])
            else:
                x = self.convs[f'upconv_{i}'](x)
            if i in self.out_sc:
                out[i] = self.activation(self.convs[f'outconv_{i}'](x))
        return out


def conv1x1(in_ch: 'int', out_ch: 'int', bias: 'bool'=True) ->nn.Conv2d:
    """Layer to convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=bias)


class FSEBlock(nn.Module):

    def __init__(self, in_ch: 'int', skip_ch: 'int', out_ch: 'Optional[int]'=None, upsample_mode: 'str'='nearest'):
        super().__init__()
        self.in_ch = in_ch + skip_ch
        self.out_ch = out_ch or in_ch
        self.upsample_mode = upsample_mode
        self.reduction = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(self.in_ch, self.in_ch // self.reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(self.in_ch // self.reduction, self.in_ch, bias=False))
        self.conv = nn.Sequential(conv1x1(self.in_ch, self.out_ch, bias=True), nn.ReLU(inplace=True))

    def forward(self, x: 'Tensor', xs_skip: 'Sequence[Tensor]') ->Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x, *xs_skip], dim=1)
        y = self.avg_pool(x).squeeze()
        y = self.se(y).sigmoid()
        y = y[..., None, None].expand_as(x)
        x = self.conv(x * y)
        return x


class HRDepthDecoder(nn.Module):
    """From HRDepth (https://arxiv.org/pdf/2012.07356.pdf)

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if not self.use_skip:
            raise ValueError('HRDepth decoder must use skip connections.')
        if len(self.enc_sc) == 4:
            warnings.warn('HRDepth requires 5 scales, but the provided backbone has only 4. The first scale will be duplicated and upsampled!')
            self.enc_sc = [self.enc_sc[0] // 2] + self.enc_sc
            self.num_ch_enc = [self.num_ch_enc[0]] + self.num_ch_enc
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.activation = ACT[self.out_act]
        self.num_ch_dec = [(ch // 2) for ch in self.num_ch_enc[1:]]
        self.num_ch_dec = [self.num_ch_dec[0] // 2] + self.num_ch_dec
        self.all_idx = ['01', '11', '21', '31', '02', '12', '22', '03', '13', '04']
        self.att_idx = ['31', '22', '13', '04']
        self.non_att_idx = ['01', '11', '21', '02', '12', '03']
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                ch_in = self.num_ch_enc[i]
                if i == 0 and j != 0:
                    ch_in //= 2
                if i == 0 and j == 4:
                    ch_in = self.num_ch_enc[i + 1] // 2
                ch_out = ch_in // 2
                self.convs[f'{i}{j}_conv_0'] = conv_block(ch_in, ch_out)
                if i == 0 and j == 4:
                    ch_in = ch_out
                    ch_out = self.num_ch_dec[i]
                    self.convs[f'{i}{j}_conv_1'] = conv_block(ch_in, ch_out)
        for idx in self.att_idx:
            row, col = int(idx[0]), int(idx[1])
            self.convs[f'{idx}_att'] = FSEBlock(in_ch=self.num_ch_enc[row + 1] // 2, skip_ch=self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1), upsample_mode=self.upsample_mode)
        for idx in self.non_att_idx:
            row, col = int(idx[0]), int(idx[1])
            if col == 1:
                self.convs[f'{row + 1}{col - 1}_conv_1'] = conv_block(in_ch=self.num_ch_enc[row + 1] // 2 + self.num_ch_enc[row], out_ch=self.num_ch_dec[row + 1])
            else:
                self.convs[f'{idx}_down'] = conv1x1(in_ch=self.num_ch_enc[row + 1] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1), out_ch=2 * self.num_ch_dec[row + 1], bias=False)
                self.convs[f'{row + 1}{col - 1}_conv_1'] = conv_block(in_ch=2 * self.num_ch_dec[row + 1], out_ch=self.num_ch_dec[row + 1])
        channels = self.num_ch_dec
        for i, c in enumerate(channels):
            if i in self.out_sc:
                self.convs[f'outconv_{i}'] = nn.Sequential(conv3x3(c, self.out_ch), self.activation)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def nested_conv(self, convs: 'Sequence[nn.Module]', x: 'Tensor', xs_skip: 'Sequence[Tensor]') ->Tensor:
        x = F.interpolate(convs[0](x), scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x, *xs_skip], dim=1)
        if len(convs) == 3:
            x = convs[2](x)
        x = convs[1](x)
        return x

    def forward(self, enc_features: 'Sequence[Tensor]') ->dict[int, Tensor]:
        if len(enc_features) == 4:
            enc_features = [F.interpolate(enc_features[0], scale_factor=2, mode=self.upsample_mode)] + enc_features
        feat = {f'{i}0': f for i, f in enumerate(enc_features)}
        for idx in self.all_idx:
            row, col = int(idx[0]), int(idx[1])
            xs_skip = [feat[f'{row}{i}'] for i in range(col)]
            if idx in self.att_idx:
                feat[f'{idx}'] = self.convs[f'{idx}_att'](self.convs[f'{row + 1}{col - 1}_conv_0'](feat[f'{row + 1}{col - 1}']), xs_skip)
            elif idx in self.non_att_idx:
                conv = [self.convs[f'{row + 1}{col - 1}_conv_0'], self.convs[f'{row + 1}{col - 1}_conv_1']]
                if col != 1:
                    conv.append(self.convs[f'{idx}_down'])
                feat[f'{idx}'] = self.nested_conv(conv, feat[f'{row + 1}{col - 1}'], xs_skip)
        x = feat['04']
        x = self.convs['04_conv_0'](x)
        x = self.convs['04_conv_1'](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        out_feat = [x, feat['04'], feat['13'], feat['22']]
        out = {i: self.convs[f'outconv_{i}'](f) for i, f in enumerate(out_feat) if i in self.out_sc}
        return out


class MonodepthDecoder(nn.Module):
    """From Monodepth(2) (https://arxiv.org/abs/1806.01260)

    Generic convolutional decoder incorporating multi-scale predictions and skip connections.

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = conv_block(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            sf = 2 ** i
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)
        for i in self.out_sc:
            self.convs[f'outconv_{i}'] = conv3x3(self.num_ch_dec[i], self.out_ch)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.act = ACT[self.out_act]

    def forward(self, enc_feat: 'Sequence[Tensor]') ->TensorDict:
        out = {}
        x = enc_feat[-1]
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_{0}'](x)
            x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
            sf = 2 ** i
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                x += [enc_feat[idx]]
            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)
            if i in self.out_sc:
                out[i] = self.act(self.convs[f'outconv_{i}'](x))
        return out


class SubPixelConv(nn.Module):

    def __init__(self, ch_in: 'int', up_factor: 'int'):
        super().__init__()
        ch_out = ch_in * up_factor ** 2
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), groups=ch_in, padding=1)
        self.shuffle = nn.PixelShuffle(up_factor)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.conv.bias)
        self.conv.weight = nn.Parameter(self.conv.weight[::4].repeat_interleave(4, 0))

    def forward(self, x):
        return self.shuffle(self.conv(x))


class SuperdepthDecoder(nn.Module):
    """From SuperDepth (https://arxiv.org/abs/1806.01260)

    Generic convolutional decoder incorporating multi-scale predictions and skip connections.

    :param num_ch_enc: (Sequence[int]) List of channels per encoder stage.
    :param enc_sc: (Sequence[int]) List of downsampling factor per encoder stage.
    :param upsample_mode: (str) Torch upsampling mode. {'nearest', 'bilinear'...}
    :param use_skip: (bool) If `True`, add skip connections from corresponding encoder stage.
    :param out_sc: (Sequence[int]) List of multi-scale output downsampling factor as 2**s.
    :param out_ch: (int) Number of output channels.
    :param out_act: (str) Activation to apply to each output stage.
    """

    def __init__(self, num_ch_enc: 'Sequence[int]', enc_sc: 'Sequence[int]', upsample_mode: 'str'='nearest', use_skip: 'bool'=True, out_sc: 'Sequence[int]'=(0, 1, 2, 3), out_ch: 'int'=1, out_act: 'str'='sigmoid'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act
        if self.out_act not in ACT:
            raise KeyError(f'Invalid activation key. ({self.out_act} vs. {tuple(ACT.keys())}')
        self.activation = ACT[self.out_act]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{0}'] = nn.Sequential(conv_block(num_ch_in, num_ch_out), SubPixelConv(num_ch_out, up_factor=2), nn.ReLU(inplace=True))
            num_ch_in = self.num_ch_dec[i]
            scale_factor = 2 ** i
            if self.use_skip and scale_factor in self.enc_sc:
                idx = self.enc_sc.index(scale_factor)
                num_ch_in += self.num_ch_enc[idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_{1}'] = conv_block(num_ch_in, num_ch_out)
        for i in self.out_sc:
            if i == 0:
                self.convs[f'outconv_{i}'] = nn.Sequential(conv3x3(self.num_ch_dec[i], self.out_ch), self.activation)
            else:
                self.convs[f'outconv_{i}'] = nn.Sequential(conv_block(self.num_ch_dec[i], self.out_ch), SubPixelConv(self.out_ch, up_factor=2 ** i), self.activation)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, feat: 'Sequence[Tensor]') ->dict[int, Tensor]:
        out = {}
        x = feat[-1]
        for i in range(4, -1, -1):
            x = [self.convs[f'upconv_{i}_{0}'](x)]
            sf = 2 ** i
            if self.use_skip and sf in self.enc_sc:
                idx = self.enc_sc.index(sf)
                x += [feat[idx]]
            x = torch.cat(x, 1)
            x = self.convs[f'upconv_{i}_{1}'](x)
            if i in self.out_sc:
                out[i] = self.convs[f'outconv_{i}'](x)
        return out


DECODERS = {'monodepth': MonodepthDecoder, 'hrdepth': HRDepthDecoder, 'superdepth': SuperdepthDecoder, 'cadepth': CADepthDecoder, 'diffnet': DiffNetDecoder, 'ddvnet': DDVNetDecoder}


class AutoencoderNet(nn.Module):
    """Image autoencoder network.
    From FeatDepth (https://arxiv.org/abs/2007.10603)

    Heavily based on the Depth network with some changes:
        - Single decoder
        - Produces 3 sigmoid channels (RGB)
        - No skip connections, it's an autoencoder!

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    :param dec_name: (str) Custom decoder type to use.
    :param out_scales: (Sequence[int]) List of multi-scale output downsampling factor as `2**s.`
    """

    def __init__(self, enc_name: 'str'='resnet18', pretrained: 'bool'=True, dec_name: 'str'='monodepth', out_scales: 'Union[int, Sequence[int]]'=(0, 1, 2, 3)):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.dec_name = dec_name
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales
        if self.dec_name not in DECODERS:
            raise KeyError(f'Invalid decoder key. ({self.dec_name} vs. {DECODERS.keys()}')
        self.encoder = timm.create_model(self.enc_name, features_only=True, pretrained=pretrained)
        self.num_ch_enc = self.encoder.feature_info.channels()
        self.enc_sc = self.encoder.feature_info.reduction()
        self.decoder = DECODERS[self.dec_name](num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc, upsample_mode='nearest', use_skip=False, out_sc=self.out_scales, out_ch=3, out_act='sigmoid')

    def forward(self, x: 'Tensor') ->TensorDict:
        """Image autoencoder forward pass.

        :param x: (Tensor) (b, 3, h, w) Input image.
        :return: {
            autoenc_feats: (list(Tensor)) Autoencoder encoder multi-scale features.
            autoenc_imgs: (TensorDict) (b, 1, h/2**s, w/2**s) Dict mapping from scales to image reconstructions.
        }
        """
        feat = self.encoder(x)
        out = {'autoenc_feats': feat}
        k = 'autoenc_imgs'
        out[k] = self.decoder(feat)
        out[k] = {k2: out[k][k2] for k2 in sorted(out[k])}
        return out


MASKS = {'explainability': 'sigmoid', 'uncertainty': 'relu', None: None}


def sort_dict(d: 'MutableMapping'):
    """Return a dict with sorted keys."""
    return {k: d[k] for k in sorted(d)}


class PoseNet(nn.Module):
    """Relative pose prediction network.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)

    This network predicts the relative pose between two images, concatenated channelwise.
    It consists of a ResNet encoder (with duplicated and scaled input weights) and a simple regression decoder.
    Pose is predicted as axis-angle rotation and a translation vector.

    The objective is to predict the relative pose between two images.
    The network consists of a ResNet encoder (with duplicated weights and scaled for the input images), plus a simple
    regression decoder.
    Pose is predicted as an axis-angle rotation and a translation vector.

    NOTE: Translation is not in metric scale unless training with stereo + mono.

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, returns an encoder pretrained on ImageNet.
    """

    def __init__(self, enc_name: 'str'='resnet18', pretrained: 'bool'=False):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.n_imgs = 2
        self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
        self.n_chenc = self.encoder.feature_info.channels()
        self.squeeze = self.block(self.n_chenc[-1], 256, kernel_size=1)
        self.decoder = nn.Sequential(self.block(256, 256, kernel_size=3, stride=1, padding=1), self.block(256, 256, kernel_size=3, stride=1, padding=1), nn.Conv2d(256, 6 * self.n_imgs, kernel_size=1))

    @staticmethod
    def block(in_ch: 'int', out_ch: 'int', kernel_size: 'int', stride: 'int'=1, padding: 'int'=0) ->nn.Module:
        """Conv + ReLU."""
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding), nn.ReLU(inplace=True))

    def forward(self, x: 'torch.Tensor') ->TensorDict:
        """Pose network forward pass.

        :param x: (Tensor) (b, 2*3, h, w) Channel-wise concatenated input images.
        :return: (dict[str, Tensor]) {
            R: (b, 2, 3) Predicted rotation in axis-angle (direction=axis, magnitude=angle).
            t: (b, 2, 3) Predicted translation.
        }
        """
        feat = self.encoder(x)
        out = self.decoder(self.squeeze(feat[-1]))
        out = 0.01 * out.mean(dim=(2, 3)).view(-1, self.n_imgs, 6)
        return {'R': out[..., :3], 't': out[..., 3:]}


class MaskReg(nn.Module):
    """Class implementing photometric loss masking regularization.
    From SfM-Learner (https://arxiv.org/abs/1704.07813)

    Based on the `explainability` mask, which predicts a weighting factor for each pixel in the photometric loss.
    To avoid the degenerate solution where all pixels are ignored, this regularization pushes all values towards 1
    using binary cross-entropy.
    """

    def forward(self, x: 'Tensor') ->LossData:
        """Mask regularization forward pass.

        :param x: (Tensor) (*) Input sigmoid explainability mask.
        :return: {
            loss: (Tensor) (,) Computed loss.
            loss_dict: (TensorDict) {}.
        }
        """
        loss = F.binary_cross_entropy(x, torch.ones_like(x))
        return loss, {}


class OccReg(nn.Module):
    """Class implementing disparity occlusion regularization.
    From DVSO (https://arxiv.org/abs/1807.02570)

    This regularization penalizes the overall disparity in the image, encouraging the network to select background
    disparities.

    NOTE: In this case we CANNOT apply mean normalization to the input disparity. By definition, this fixes the mean of
    all elements to 1, meaning the loss is impossible to minimize.

    NOTE: The benefits of applying this regularization to purely monocular supervision are unclear,
    since the loss could simply be optimized by making all disparities smaller.

    :param invert: (bool) If `True`, encourage foreground disparities instead of background.
    """

    def __init__(self, invert: 'bool'=False):
        super().__init__()
        self.invert = invert
        self._sign = nn.Parameter(torch.tensor(-1 if self.invert else 1), requires_grad=False)

    def forward(self, x: 'Tensor') ->LossData:
        """Occlusion regularization forward pass.

        :param x: (Tensor) (*) Input sigmoid disparities.
        :return: {
            loss: (Tensor) (,) Computed loss.
            loss_dict: (TensorDict) {}.
        }
        """
        loss = self._sign * x.mean()
        return loss, {}


def compute_grad(x: Tensor, /, use_blur: 'bool'=False, ch_mean: 'bool'=False) ->tuple[Tensor, Tensor]:
    """Compute absolute spatial gradients in `(x, y)` directions.

    :param x: (Tensor) (b, c, h, w) Input to compute spatial gradients for.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    :param ch_mean: (bool) If `True`, return the mean gradient across channels, i.e. `c=1`.
    :return: (Tensor, Tensor) (b, (c|1), h, w) Spatial gradients in `(x, y)`.
    """
    b, c, h, w = x.shape
    if use_blur:
        x = gaussian_blur2d(x, kernel_size=(3, 3), sigma=(1, 1))
    dx = (x[..., :, :-1] - x[..., :, 1:]).abs()
    dx = torch.cat((dx, x.new_zeros((b, c, h, 1))), dim=-1)
    dy = (x[..., :-1, :] - x[..., 1:, :]).abs()
    dy = torch.cat((dy, x.new_zeros((b, c, 1, w))), dim=-2)
    if ch_mean:
        dx, dy = dx.mean(dim=1, keepdim=True), dy.mean(dim=1, keepdim=True)
    return dx, dy


def compute_laplacian(x: Tensor, /, use_blur: 'bool'=False, ch_mean: 'bool'=False) ->tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute absolute second-order spatial gradients in (xx, yy, xy, yx) directions.

    :param x: (Tensor) (b, c, h, w) Input to compute spatial gradients for.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    :param ch_mean: (bool) If `True`, return the mean gradient across channels, i.e. `c=1`.
    :return: (Tensor, Tensor, Tensor, Tensor) (b, (c|1), h, w) Second-order spatial gradients in `(xx, yy, xy, yx)`.
    """
    dx, dy = compute_grad(x, use_blur)
    dxx, dxy = compute_grad(dx, use_blur)
    dyx, dyy = compute_grad(dy, use_blur)
    if ch_mean:
        dxx, dxy = dxx.mean(dim=1, keepdim=True), dxy.mean(dim=1, keepdim=True)
        dyx, dyy = dyx.mean(dim=1, keepdim=True), dyy.mean(dim=1, keepdim=True)
    return dxx, dyy, dxy, dyx


class SmoothReg(nn.Module):
    """Class implementing a disparity smoothness regularization.

    - Base: From Garg (https://arxiv.org/abs/1603.04992).
    - Edge-aware: From Monodepth (https://arxiv.org/abs/1609.03677)
    - Edge-aware + Laplacian: From DVSO (https://arxiv.org/abs/1807.02570)

    :param use_edges: (bool) If `True`, do not penalize disparity gradients aligned with image gradients.
    :param use_laplacian: (bool) If `True`, compute second-order gradients instead of first-order.
    :param use_blur: (bool) If `True`, apply Gaussian blurring to the input.
    """

    def __init__(self, use_edges: 'bool'=False, use_laplacian: 'bool'=False, use_blur: 'bool'=False) ->None:
        super().__init__()
        self.use_edges = use_edges
        self.use_laplacian = use_laplacian
        self.use_blur = use_blur
        self._fn = compute_laplacian if self.use_laplacian else compute_grad

    def forward(self, disp: 'Tensor', img: 'Tensor') ->LossData:
        """Smoothness regularization forward pass.

        :param disp: (Tensor) (b, 1, h, w) Input sigmoid disparity.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        disp = ops.mean_normalize(disp)
        disp_dx, disp_dy = self._fn(disp, use_blur=self.use_blur)[:2]
        disp_grad = (disp_dx.pow(2) + disp_dy.pow(2)).clamp(min=ops.eps(disp)).sqrt()
        img_dx, img_dy = self._fn(img, use_blur=self.use_blur, ch_mean=True)[:2]
        img_grad = (img_dx.pow(2) + img_dy.pow(2)).clamp(min=ops.eps(disp)).sqrt()
        if self.use_edges:
            disp_dx *= (-img_dx).exp()
            disp_dy *= (-img_dy).exp()
        loss = disp_dx.mean() + disp_dy.mean()
        return loss, {'disp_grad': disp_grad, 'image_grad': img_grad}


class FeatPeakReg(nn.Module):
    """Class implementing feature gradient peakiness regularization.
    From Feat-Depth (https://arxiv.org/abs/2007.10603).

    Objective is to learn a feature representation discriminative in smooth image regions, by encouraging
    first-order gradients.

    :param use_edges: (bool) If `True`, penalize feature gradient smoothness aligned with image gradients.
    """

    def __init__(self, use_edges: 'bool'=False):
        super().__init__()
        self.use_edges = use_edges

    def forward(self, feat: 'Tensor', img: 'Tensor') ->LossData:
        """Feature peakiness regularization forward pass.

        :param feat: (Tensor) (b, c, h, w) Input feature maps.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        feat_dx, feat_dy = compute_grad(feat)
        feat_grad = (feat_dx.pow(2) + feat_dy.pow(2)).clamp(min=ops.eps(feat)).sqrt()
        if self.use_edges:
            dx, dy = compute_grad(img, ch_mean=True)
            feat_dx *= (-dx).exp()
            feat_dy *= (-dy).exp()
        loss = -(feat_dx.mean() + feat_dy.mean())
        return loss, {'feat_grad': feat_grad}


class FeatSmoothReg(nn.Module):
    """Class implementing second-order feature gradient smoothness regularization.
    From Feat-Depth (https://arxiv.org/abs/2007.10603).

    Objective is to learn a feature representation with smooth second-order gradients to make optimization easier.

    :param use_edges: (bool) If `True`, penalize feature gradient smoothness aligned with image gradients.
    """

    def __init__(self, use_edges: 'bool'=False) ->None:
        super().__init__()
        self.edge_aware = use_edges

    def forward(self, feat: 'Tensor', img: 'Tensor') ->LossData:
        """Second-order feature smoothness regularization forward pass.

        :param feat: (Tensor) (b, c, h, w) Input feature maps.
        :param img: (Tensor) (b, 3, h, w) Corresponding image.
        :return:
        """
        feat_dxx, feat_dyy, feat_dxy, feat_dyx = compute_laplacian(feat)
        feat_grad = (feat_dxx.pow(2) + feat_dyy.pow(2)).clamp(min=ops.eps(feat)).sqrt()
        if self.edge_aware:
            dxx, dyy, dxy, dyx = compute_laplacian(img, ch_mean=True)
            feat_dxx *= (-dxx).exp()
            feat_dyy *= (-dyy).exp()
            feat_dxy *= (-dxy).exp()
            feat_dyx *= (-dyx).exp()
        loss = feat_dxx.mean() + feat_dyy.mean() + feat_dxy.mean() + feat_dyx.mean()
        return loss, {'feat_grad': feat_grad}


class BackprojectDepth(nn.Module):
    """Module to backproject a depth map into a pointcloud.

    :param shape: (tuple[int, int]) Depth map shape as (height, width).
    """

    def __init__(self, shape: 'tuple[int, int]'):
        super().__init__()
        self.h, self.w = shape
        self.ones = nn.Parameter(torch.ones(1, 1, self.h * self.w), requires_grad=False)
        grid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing='xy')
        pix = torch.stack(grid).view(2, -1)[None]
        pix = torch.cat((pix, self.ones), dim=1)
        self.pix = nn.Parameter(pix, requires_grad=False)

    def forward(self, depth: 'Tensor', K_inv: 'Tensor') ->Tensor:
        """Backproject a depth map into a pointcloud.

        Camera is assumed to be at the origin.

        :param depth: (Tensor) (b, 1, h, w) Depth map to backproject.
        :param K_inv: (Tensor) (b, 4, 4) Inverse camera intrinsic parameters.
        :return: (Tensor) (b, 4, h*w) Backprojected 3-D points as (x, y, z, homo).
        """
        b = depth.shape[0]
        pts = K_inv[:, :3, :3] @ self.pix.repeat(b, 1, 1)
        pts *= depth.flatten(-2)
        pts = torch.cat((pts, self.ones.repeat(b, 1, 1)), dim=1)
        return pts


class ProjectPoints(nn.Module):
    """Convert a 3-D pointcloud into image grid sample locations.

    :param shape: (tuple[int, int]) Depth map shape as (height, width).
    """

    def __init__(self, shape: 'tuple[int, int]'):
        super().__init__()
        self.h, self.w = shape

    def forward(self, pts: 'Tensor', K: 'Tensor', T: 'Tensor'=None) ->tuple[Tensor, Tensor]:
        """Convert a 3-D pointcloud into image grid sample locations.

        :param pts: (Tensor) (b, 4, h*w) Pointcloud points to project.
        :param K:  (Tensor) (b, 4, 4) Camera intrinsic parameters.
        :param T: (Tensor) (b, 4, 4) Optional camera extrinsic parameters, i.e. additional transform to apply.
        :return: (
            pix_coords: (Tensor) (b, h, w, 2) Grid sample locations [-1, 1] as (x, y).
            cam_depth: (Tensor) (b, 1, h, w) Depth map in the transformed reference frame.
        )
        """
        if T is not None:
            pts = (T @ pts)[:, :3]
        depth = pts[:, 2:].clamp(min=ops.eps(pts))
        pix = (K[:, :3, :3] @ (pts / depth.clamp(min=0.1)))[:, :2]
        depth = depth.view(-1, 1, self.h, self.w)
        grid = pix.view(-1, 2, self.h, self.w).permute(0, 2, 3, 1)
        grid[..., 0] /= self.w - 1
        grid[..., 1] /= self.h - 1
        grid = (grid - 0.5) * 2
        return grid, depth


class ViewSynth(nn.Module):
    """Warp an image according to depth and pose information.

    :param shape: (tuple[int, int]) Depth map shape as (h, w).
    """

    def __init__(self, shape: 'tuple[int, int]'):
        super().__init__()
        self.shape = shape
        self.backproj = BackprojectDepth(shape)
        self.proj = ProjectPoints(shape)
        self.sample = partial(F.grid_sample, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, input: 'Tensor', depth: 'Tensor', T: 'Tensor', K: 'Tensor', K_inv: 'Optional[Tensor]'=None) ->tuple[Tensor, Tensor, Tensor]:
        """Warp the input image.

        :param input: (Tensor) (b, *, h, w) Input tensor to warp.
        :param depth: (Tensor) (b, 1, h, w) Predicted depth for the source image.
        :param T: (Tensor) (b, 4, 4) Pose of the target image relative to the source image.
        :param K: (Tensor) (b, 4, 4) The target image camera intrinsics.
        :param K_inv: (Optional[Tensor]) (b, 4, 4) The source image inverse camera intrinsics. (default: `K.inverse()`)
        :return: (Tensor) (b, *, h, w) The synthesized warped image.
        """
        if K_inv is None:
            K_inv = K.inverse()
        pts = self.backproj(depth, K_inv)
        grid, depth_warp = self.proj(pts, K, T)
        mask_valid = (grid.abs() < 1).all(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        input_warp = self.sample(input=input, grid=grid)
        return input_warp, depth_warp, mask_valid


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different data.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    IMAGENET = 'imagenet'
    CIFAR10 = 'cifar10'
    SVHN = 'svhn'


def _apply_op(img: 'Tensor', op_name: 'str', magnitude: 'float', interpolation: 'InterpolationMode', fill: 'Optional[List[float]]'):
    if op_name == 'ShearX':
        raise ValueError(f'Attempted geometric transformation "{op_name}"')
    elif op_name == 'ShearY':
        raise ValueError(f'Attempted geometric transformation "{op_name}"')
    elif op_name == 'TranslateX':
        raise ValueError(f'Attempted geometric transformation "{op_name}"')
    elif op_name == 'TranslateY':
        raise ValueError(f'Attempted geometric transformation "{op_name}"')
    elif op_name == 'Rotate':
        raise ValueError(f'Attempted geometric transformation "{op_name}"')
    elif op_name == 'Brightness':
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == 'Color':
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == 'Contrast':
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == 'Sharpness':
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == 'Posterize':
        img = F.posterize(img, int(magnitude))
    elif op_name == 'Solarize':
        img = F.solarize(img, magnitude)
    elif op_name == 'AutoContrast':
        img = F.autocontrast(img)
    elif op_name == 'Equalize':
        img = F.equalize(img)
    elif op_name == 'Invert':
        img = F.invert(img)
    elif op_name == 'Identity':
        pass
    else:
        raise ValueError('The provided operator {} is not recognized.'.format(op_name))
    return img


class AutoAugment(torch.nn.Module):
    """AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, policy: 'AutoAugmentPolicy'=AutoAugmentPolicy.IMAGENET, interpolation: 'InterpolationMode'=InterpolationMode.NEAREST, fill: 'Optional[List[float]]'=None) ->None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(self, policy: 'AutoAugmentPolicy') ->List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [(('Posterize', 0.4, 8), ('Rotate', 0.6, 9)), (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)), (('Equalize', 0.8, None), ('Equalize', 0.6, None)), (('Posterize', 0.6, 7), ('Posterize', 0.6, 6)), (('Equalize', 0.4, None), ('Solarize', 0.2, 4)), (('Equalize', 0.4, None), ('Rotate', 0.8, 8)), (('Solarize', 0.6, 3), ('Equalize', 0.6, None)), (('Posterize', 0.8, 5), ('Equalize', 1.0, None)), (('Equalize', 0.6, None), ('Posterize', 0.4, 6)), (('Equalize', 0.0, None), ('Equalize', 0.8, None)), (('Invert', 0.6, None), ('Equalize', 1.0, None)), (('Color', 0.6, 4), ('Contrast', 1.0, 8)), (('Color', 0.8, 8), ('Solarize', 0.8, 7)), (('Sharpness', 0.4, 7), ('Invert', 0.6, None)), (('Color', 0.4, 0), ('Equalize', 0.6, None)), (('Equalize', 0.4, None), ('Solarize', 0.2, 4)), (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)), (('Invert', 0.6, None), ('Equalize', 1.0, None)), (('Color', 0.6, 4), ('Contrast', 1.0, 8)), (('Equalize', 0.8, None), ('Equalize', 0.6, None))]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [(('Invert', 0.1, None), ('Contrast', 0.2, 6)), (('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)), (('AutoContrast', 0.5, None), ('Equalize', 0.9, None)), (('Color', 0.4, 3), ('Brightness', 0.6, 7)), (('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)), (('Equalize', 0.6, None), ('Equalize', 0.5, None)), (('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)), (('Color', 0.7, 7), ('TranslateX', 0.5, 8)), (('Equalize', 0.3, None), ('AutoContrast', 0.4, None)), (('Brightness', 0.9, 6), ('Color', 0.2, 8)), (('Solarize', 0.5, 2), ('Invert', 0.0, None)), (('Equalize', 0.2, None), ('AutoContrast', 0.6, None)), (('Equalize', 0.2, None), ('Equalize', 0.6, None)), (('Color', 0.9, 9), ('Equalize', 0.6, None)), (('AutoContrast', 0.8, None), ('Solarize', 0.2, 8)), (('Brightness', 0.1, 3), ('Color', 0.7, 0)), (('Solarize', 0.4, 5), ('AutoContrast', 0.9, None)), (('AutoContrast', 0.9, None), ('Solarize', 0.8, 3)), (('Equalize', 0.8, None), ('Invert', 0.1, None))]
        elif policy == AutoAugmentPolicy.SVHN:
            return [(('Equalize', 0.6, None), ('Solarize', 0.6, 6)), (('Invert', 0.9, None), ('Equalize', 0.6, None)), (('Equalize', 0.6, None), ('Rotate', 0.9, 3)), (('Invert', 0.9, None), ('AutoContrast', 0.8, None)), (('Equalize', 0.6, None), ('Rotate', 0.9, 3)), (('Equalize', 0.9, None), ('TranslateY', 0.6, 6)), (('Invert', 0.9, None), ('Equalize', 0.6, None)), (('Contrast', 0.3, 3), ('Rotate', 0.8, 4)), (('Invert', 0.8, None), ('TranslateY', 0.0, 2)), (('ShearY', 0.7, 6), ('Solarize', 0.4, 8)), (('Invert', 0.6, None), ('Rotate', 0.8, 4)), (('Solarize', 0.7, 2), ('TranslateY', 0.6, 7))]
        else:
            raise ValueError('The provided policy {} is not recognized.'.format(policy))

    def _augmentation_space(self, num_bins: 'int', image_size: 'List[int]') ->Dict[str, Tuple[Tensor, bool]]:
        return {'Brightness': (torch.linspace(0.0, 0.9, num_bins), True), 'Color': (torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (torch.linspace(0.0, 0.9, num_bins), True), 'Posterize': (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (torch.linspace(255.0, 0.0, num_bins), False), 'AutoContrast': (torch.tensor(0.0), False), 'Equalize': (torch.tensor(0.0), False), 'Invert': (torch.tensor(0.0), False)}

    @staticmethod
    def get_params(transform_num: 'int') ->Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))
        return policy_id, probs, signs

    def forward(self, img: 'Tensor') ->Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]
        transform_id, probs, signs = self.get_params(len(self.policies))
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                op_meta = self._augmentation_space(10, F.get_image_size(img))
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        return img

    def __repr__(self) ->str:
        return self.__class__.__name__ + '(policy={}, fill={})'.format(self.policy, self.fill)


class RandAugment(torch.nn.Module):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_ops: 'int'=2, magnitude: 'int'=9, num_magnitude_bins: 'int'=31, interpolation: 'InterpolationMode'=InterpolationMode.NEAREST, fill: 'Optional[List[float]]'=None) ->None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: 'int', image_size: 'List[int]') ->Dict[str, Tuple[Tensor, bool]]:
        return {'Identity': (torch.tensor(0.0), False), 'Brightness': (torch.linspace(0.0, 0.9, num_bins), True), 'Color': (torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (torch.linspace(0.0, 0.9, num_bins), True), 'Posterize': (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (torch.linspace(255.0, 0.0, num_bins), False), 'AutoContrast': (torch.tensor(0.0), False), 'Equalize': (torch.tensor(0.0), False)}

    def forward(self, img: 'Tensor') ->Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]
        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        return img

    def __repr__(self) ->str:
        s = self.__class__.__name__ + '('
        s += 'num_ops={num_ops}'
        s += ', magnitude={magnitude}'
        s += ', num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


class TrivialAugmentWide(torch.nn.Module):
    """Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: 'int'=31, interpolation: 'InterpolationMode'=InterpolationMode.NEAREST, fill: 'Optional[List[float]]'=None) ->None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: 'int') ->Dict[str, Tuple[Tensor, bool]]:
        return {'Brightness': (torch.linspace(0.0, 0.99, num_bins), True), 'Color': (torch.linspace(0.0, 0.99, num_bins), True), 'Contrast': (torch.linspace(0.0, 0.99, num_bins), True), 'Sharpness': (torch.linspace(0.0, 0.99, num_bins), True), 'Posterize': (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False), 'Solarize': (torch.linspace(255.0, 0.0, num_bins), False), 'AutoContrast': (torch.tensor(0.0), False), 'Equalize': (torch.tensor(0.0), False)}

    def forward(self, img: 'Tensor') ->Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]
        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) ->str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ChannelAttention,
     lambda: ([], {'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseL1Error,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DetailEmphasis,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OccReg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PhotoError,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ReconstructionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SSIMError,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SelfAttentionBlock,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StructurePerception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SubPixelConv,
     lambda: ([], {'ch_in': 4, 'up_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

