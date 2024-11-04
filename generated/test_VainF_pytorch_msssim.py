import sys
_module = sys.modules[__name__]
del sys
pytorch_msssim = _module
ssim = _module
setup = _module
datasets = _module
image_dataset = _module
models = _module
autoencoder = _module
gdn = _module
train = _module
tests_comparisons_2d3d = _module
tests_comparisons_skimage = _module
tests_comparisons_tf_skimage = _module
tests_cuda = _module
tests_loss = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import warnings


import torch


import torch.nn.functional as F


from torch.utils import data


import torch.nn as nn


import math


from torch.autograd import Function


import numpy as np


from torchvision import transforms


import time


import tensorflow as tf


from torch.autograd import Variable


from torch import optim


def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """ Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([(ws == 1) for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)
    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(f'Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}')
    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03), nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f'Input images should have the same dimensions, but got {X.shape} and {Y.shape}.')
    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)
    if len(X.shape) not in (4, 5):
        raise ValueError(f'Input images should be 4-d or 5-d tensors, but got {X.shape}')
    if not X.type() == Y.type():
        raise ValueError(f'Input images should have the same dtype, but got {X.type()} and {Y.type()}.')
    if win is not None:
        win_size = win.shape[-1]
    if not win_size % 2 == 1:
        raise ValueError('Window size should be odd.')
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)
    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


class SSIM(torch.nn.Module):

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


def ms_ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f'Input images should have the same dimensions, but got {X.shape} and {Y.shape}.')
    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)
    if not X.type() == Y.type():
        raise ValueError(f'Input images should have the same dtype, but got {X.type()} and {Y.type()}.')
    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f'Input images should be 4-d or 5-d tensors, but got {X.shape}')
    if win is not None:
        win_size = win.shape[-1]
    if not win_size % 2 == 1:
        raise ValueError('Window size should be odd.')
    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * 2 ** 4, 'Image size should be larger than %d due to the 4 downsamplings in ms-ssim' % ((win_size - 1) * 2 ** 4)
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [(s % 2) for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)
    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class MS_SSIM(torch.nn.Module):

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        """ class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, weights=self.weights, K=self.K)


class LowerBound(Function):

    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
        return inputs.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors
        pass_through_1 = inputs >= bound
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):

    def __init__(self, num_features, inverse=False, gamma_init=0.1, beta_bound=1e-06, gamma_bound=0.0, reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset ** 2
        beta_init = torch.sqrt(torch.ones(num_features, dtype=torch.float) + self.pedestal)
        gama_init = torch.sqrt(torch.full((num_features, num_features), fill_value=gamma_init, dtype=torch.float) * torch.eye(num_features, dtype=torch.float) + self.pedestal)
        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gama_init)
        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = LowerBound.apply(var, bound)
        return var ** 2 - self.pedestal

    def forward(self, x):
        gamma = self._reparam(self.gamma, self.gamma_bound).view(self.num_features, self.num_features, 1, 1)
        beta = self._reparam(self.beta, self.beta_bound)
        norm_pool = F.conv2d(x ** 2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)
        if self._inverse:
            norm_pool = x * norm_pool
        else:
            norm_pool = x / norm_pool
        return norm_pool


class Decoder(nn.Module):
    """ Decoder
    """

    def __init__(self, C=32, M=128, out_chan=3):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=C, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False), GDN(M, inverse=True), nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False), GDN(M, inverse=True), nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False), GDN(M, inverse=True), nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False))

    def forward(self, q):
        return torch.sigmoid(self.dec(q))


class Encoder(nn.Module):
    """ Encoder
    """

    def __init__(self, C=32, M=128, in_chan=3):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False), GDN(M), nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False), GDN(M), nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False), GDN(M), nn.Conv2d(in_channels=M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False))

    def forward(self, x):
        return self.enc(x)


class AutoEncoder(nn.Module):

    def __init__(self, C=128, M=128, in_chan=3, out_chan=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(C=C, M=M, in_chan=in_chan)
        self.decoder = Decoder(C=C, M=M, out_chan=out_chan)

    def forward(self, x, **kargs):
        code = self.encoder(x)
        out = self.decoder(code)
        return out


class MS_SSIM_Loss(MS_SSIM):

    def forward(self, img1, img2):
        return 100 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):

    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AutoEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (GDN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_VainF_pytorch_msssim(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

