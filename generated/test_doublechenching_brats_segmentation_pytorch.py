
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


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import random


from torch.nn import functional as F


import torch.nn as nn


import torch.nn.functional as F


from torch.optim import lr_scheduler


import logging


import time


from torch import nn


from torch.autograd import Variable


import torch.utils.data


from torchvision.models.inception import inception_v3


from scipy.stats import entropy


from math import exp


from collections import defaultdict


from collections import deque


import warnings


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul
        return x


class VAEBranch(nn.Module):

    def __init__(self, input_shape, init_channels, out_channels, squeeze_channels=None):
        super(VAEBranch, self).__init__()
        self.input_shape = input_shape
        if squeeze_channels:
            self.squeeze_channels = squeeze_channels
        else:
            self.squeeze_channels = init_channels * 4
        self.hidden_conv = nn.Sequential(nn.GroupNorm(8, init_channels * 8), nn.ReLU(inplace=True), nn.Conv3d(init_channels * 8, self.squeeze_channels, (3, 3, 3), padding=(1, 1, 1)), nn.AdaptiveAvgPool3d(1))
        self.mu_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)
        self.logvar_fc = nn.Linear(self.squeeze_channels // 2, self.squeeze_channels // 2)
        recon_shape = np.prod(self.input_shape) // 16 ** 3
        self.reconstraction = nn.Sequential(nn.Linear(self.squeeze_channels // 2, init_channels * 8 * recon_shape), nn.ReLU(inplace=True))
        self.vconv4 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 8, (1, 1, 1)), nn.Upsample(scale_factor=2))
        self.vconv3 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 4, (3, 3, 3), padding=(1, 1, 1)), nn.Upsample(scale_factor=2), BasicBlock(init_channels * 4, init_channels * 4))
        self.vconv2 = nn.Sequential(nn.Conv3d(init_channels * 4, init_channels * 2, (3, 3, 3), padding=(1, 1, 1)), nn.Upsample(scale_factor=2), BasicBlock(init_channels * 2, init_channels * 2))
        self.vconv1 = nn.Sequential(nn.Conv3d(init_channels * 2, init_channels, (3, 3, 3), padding=(1, 1, 1)), nn.Upsample(scale_factor=2), BasicBlock(init_channels, init_channels))
        self.vconv0 = nn.Conv3d(init_channels, out_channels, (1, 1, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.hidden_conv(x)
        batch_size = x.size()[0]
        x = x.view((batch_size, -1))
        mu = x[:, :self.squeeze_channels // 2]
        mu = self.mu_fc(mu)
        logvar = x[:, self.squeeze_channels // 2:]
        logvar = self.logvar_fc(logvar)
        z = self.reparameterize(mu, logvar)
        re_x = self.reconstraction(z)
        recon_shape = [batch_size, self.squeeze_channels // 2, self.input_shape[0] // 16, self.input_shape[1] // 16, self.input_shape[2] // 16]
        re_x = re_x.view(recon_shape)
        x = self.vconv4(re_x)
        x = self.vconv3(x)
        x = self.vconv2(x)
        x = self.vconv1(x)
        vout = self.vconv0(x)
        return vout, mu, logvar


class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)
        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)
        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)
        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)
        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)
        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)
        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)
        c4d = self.dropout(c4d)
        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)
        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)
        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)
        return uout, c4d


class UnetVAE3D(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UnetVAE3D, self).__init__()
        self.unet = UNet3D(input_shape, in_channels, out_channels, init_channels, p)
        self.vae_branch = VAEBranch(input_shape, init_channels, out_channels=in_channels)

    def forward(self, x):
        uout, c4d = self.unet(x)
        vout, mu, logvar = self.vae_branch(c4d)
        return uout, vout, mu, logvar


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

