
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


import torch.utils.data


import random


import numpy as np


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.transforms.functional as TF


from abc import ABC


from abc import abstractmethod


import torch


import warnings


from collections import OrderedDict


import torch.nn as nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


import matplotlib.pyplot as plt


import time


import pandas as pd


from scipy import sparse


from copy import deepcopy


from scipy.sparse import csr_matrix


import itertools


from typing import List


from typing import Union


class Identity(nn.Module):

    def forward(self, x):
        return x


class UnetSkipConnectionBlock3d(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock3d, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        interp = 'trilinear'
        transp_conv = False
        if outermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                upconv = [upsamp, conv]
            down = [downconv]
            up = [uprelu, *upconv, nn.ReLU()]
            model = down + [submodule] + up
        elif innermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]
            down = [downrelu, downconv]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]
            down = [downrelu, downconv, downnorm]
            up = [uprelu, *upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator3d(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a 3D Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator3d, self).__init__()
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3d(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.BN(self.relu(self.conv2(self.BN(self.relu(self.conv1(x))))))


class Encoder(nn.Module):

    def __init__(self, in_ch, chs=(32, 64, 128, 256, 512)):
        super().__init__()
        chs = (in_ch,) + chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class ResizeConv(nn.Module):

    def __init__(self, in_ch, out_ch, interp='nearest'):
        super().__init__()
        self.upsamp = nn.Upsample(scale_factor=2, mode=interp)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsamp(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([ResizeConv(chs[i], chs[i + 1], interp='trilinear') for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet3D(nn.Module):

    def __init__(self, in_ch=1, enc_chs=(32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), out_ch=1):
        super().__init__()
        self.encoder = Encoder(in_ch, enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv3d(dec_chs[-1], out_ch, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        activation = nn.ReLU()
        out = activation(out)
        return out


class DVHLoss(nn.Module):

    def __init__(self):
        super(DVHLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]
        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            hist[:, i] = num / vols
        return self.loss(hist, target_hist)


class BhattLoss(nn.Module):

    def __init__(self):
        super(BhattLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, target_hist, target_bins, oar):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0, 1] - target_bins[0, 0]
        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:, i]) / bin_w)
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            hist[:, i] = num / vols
        None
        None
        histprod = torch.sqrt(hist * target_hist)
        None
        bhattdist = torch.sum(histprod, axis=(1, 2))
        bhattdist = torch.clamp(bhattdist, 0.001)
        None
        bhattloss = torch.mean(-torch.log(bhattdist))
        return bhattloss


class MomentLoss(nn.Module):

    def __init__(self):
        super(MomentLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, oar, dose):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            target hist (tensor)    -- [N, n_bins, n_oars]
            target bins (tensor)    -- [N, n_bins]
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """
        vols = torch.sum(oar, axis=(2, 3, 4))
        keys = ['Eso', 'Cord', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
        momentOfStructure = {'Eso': {'moments': [1, 2, 10], 'weights': [1, 1, 1]}, 'Cord': {'moments': [1, 2, 10], 'weights': [1, 1, 1]}, 'Heart': {'moments': [1, 2, 10], 'weights': [1, 1, 1]}, 'Lung_L': {'moments': [1, 2, 10], 'weights': [1, 1, 1]}, 'Lung_R': {'moments': [1, 2, 10], 'weights': [1, 1, 1]}, 'PTV': {'moments': [2, 4, 6], 'weights': [1, 1, 1]}}
        oarPredMoment = []
        oarRealMoment = []
        pres = 60
        epsilon = 1e-05
        for i in range(oar.shape[1]):
            moments = momentOfStructure[keys[i]]['moments']
            weights = momentOfStructure[keys[i]]['weights']
            for j in range(len(moments)):
                gEUDa = moments[j]
                weight = weights[j]
                oarpreddose = predicted_dose * oar[:, i, :, :, :]
                oarRealDose = dose * oar[:, i, :, :, :]
                if i < oar.shape[1] - 1:
                    oarPredMomenta = weight * torch.pow(1 / vols[0, i] * torch.sum(torch.pow(oarpreddose, gEUDa), axis=(2, 3, 4)) + epsilon, 1 / gEUDa)
                    oarRealMomenta = weight * torch.pow(1 / vols[0, i] * torch.sum(torch.pow(oarRealDose, gEUDa), axis=(2, 3, 4)) + epsilon, 1 / gEUDa)
                    oarPredMoment.append(oarPredMomenta)
                    oarRealMoment.append(oarRealMomenta)
        oarPredMoment = torch.stack(oarPredMoment)
        oarRealMoment = torch.stack(oarRealMoment)
        return self.loss(oarPredMoment, oarRealMoment)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResizeConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {})),
]

