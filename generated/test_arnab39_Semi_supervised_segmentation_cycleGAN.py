
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


from torch import nn


import torch


from torch.nn import functional as F


import functools


from torch.nn import init


import torch.nn as nn


import torch.nn.functional as F


from torchvision.transforms import *


import numpy as np


import random


import pandas as pd


import scipy.misc as m


import scipy.io as sio


from torch.utils.data import Dataset


import re


import itertools


from torch.autograd import Variable


import torchvision


import torchvision.datasets as dsets


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import copy


from torchvision import models


from collections import namedtuple


from torchvision import transforms


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), norm_layer(out_dim), nn.LeakyReLU(0.2, True))


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, norm_layer=norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, norm_layer=norm_layer, padding=1, bias=use_bias)]
        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
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


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)


def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1), conv_norm_relu(dim, dim, kernel_size=3, norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding, bias=bias), norm_layer(out_dim), nn.ReLU(True))


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6, softmax=False):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        res_model = [nn.ReflectionPad2d(3), conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias), conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias), conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]
        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]
        if softmax:
            res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7)]
        else:
            res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias), nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)


class DownsamplingBottleneck(nn.Module):
    """
    This is the Downsampling bottleneck layer to downsample the input
    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=0, return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        assert internal_ratio > 1 and internal_ratio < in_channels, 'The value of the internal_ratio is not correct'
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_max1 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.ext_reg = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_reg(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding
        main = torch.cat((main, padding), 1)
        out = main + ext
        if self.return_indices:
            return self.out_activation(out), max_indices
        else:
            return self.out_activation(out)


class InitialBlock(nn.Module):
    """
    This will be initial block of the network
    """

    def __init__(self, input_dim=3, output_dim=16, kernel_size=3, relu=True, padding=0, bias=False):
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_branch = nn.Conv2d(in_channels=input_dim, out_channels=output_dim - 3, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        self.ext_branch = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.activation = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out


class RegularBottleneck(nn.Module):
    """
    So in this also there are two branches:
    Main branch -->
        This is basically simply the input
    Extension Branch -->
        1. 1*1 conv to decrease the spatial dimensions to a internal_ratio specified by the user
        2. conv layer, which can be regular, atrous or asymmetric convolution
        3. 1*1 to increase the spatial dimension back from internal_ratio to the original size
        4. Regularizer which in this case is dropout
    """

    def __init__(self, channels, internal_ratio=4, kernel_size=3, dilation=1, padding=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        assert internal_ratio > 1 and internal_ratio < channels, 'The value of the internal_ratio is not correct'
        internal_channels = channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        if asymmetric:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), stride=(1, 1), padding=(padding, 0), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation, nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, padding), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        else:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(channels), activation)
        self.ext_regularizer = nn.Dropout2d(p=dropout_prob)
        self.out_relu = activation

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regularizer(ext)
        out = ext + main
        return self.out_relu(out)


class UpsamplingBottleneck(nn.Module):
    """
    This is the Upsampling bottleneck as is displayed in the paper
    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=0, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        assert internal_ratio > 1 and internal_ratio < in_channels, 'The value of the internal_ratio is not correct'
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias), nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.ext_reg = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation

    def forward(self, x, max_indices):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_reg(ext)
        out = main + ext
        return self.out_activation(out)


class ENet(nn.Module):
    """
    num_classes : The number of classes to segment the image into
    encoder_relu : whether to include relu or prelu for encoder part
    decoder_relu : whether to include relu or prelu for decoder part
    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        return x


class APN(nn.Module):
    """
    This is the Attention Pyramid Network including the whole upsampling block
    """

    def __init__(self, in_channels, out_channels, image_dim, bias=True, relu=True):
        """
        image_dim = dimension of the incoming image
        """
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.conv_33 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=bias), nn.BatchNorm2d(in_channels), activation)
        self.conv_55 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, bias=bias), nn.BatchNorm2d(in_channels), activation)
        self.conv_77 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=2, padding=3, bias=bias), nn.BatchNorm2d(in_channels), activation)
        self.conv_77_toC = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.conv_55_toC = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.conv_33_toC = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.conv_11 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.global_pool = nn.AvgPool2d(kernel_size=(image_dim, image_dim), stride=0, padding=0)
        self.pool_conv_toC = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), nn.BatchNorm2d(out_channels), activation)
        self.upsample_times2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_globalPool = nn.UpsamplingBilinear2d(size=(image_dim, image_dim))

    def forward(self, x):
        output_conv33 = self.conv_33(x)
        output_conv55 = self.conv_55(output_conv33)
        output_conv77 = self.conv_77(output_conv55)
        output_conv77_toC = self.conv_77_toC(output_conv77)
        output_conv77_upsample = self.upsample_times2(output_conv77_toC)
        output_conv55_toC = self.conv_55_toC(output_conv55)
        output_conv55_toC = output_conv55_toC + output_conv77_upsample
        output_conv55_upsample = self.upsample_times2(output_conv55_toC)
        output_conv33_toC = self.conv_33_toC(output_conv33)
        output_conv33_toC = output_conv33_toC + output_conv55_upsample
        output_conv33_upsample = self.upsample_times2(output_conv33_toC)
        output_conv11 = self.conv_11(x)
        output_global_pool = self.global_pool(x)
        output_pool_conv_toC = self.pool_conv_toC(output_global_pool)
        output_pool_upsample = self.upsample_globalPool(output_pool_conv_toC)
        output_conv11 = output_conv11 * output_conv33_upsample
        output_conv11 = output_conv11 + output_pool_upsample
        final_output = F.upsample_bilinear(output_conv11, scale_factor=8)
        return final_output


class Downsample_Block_led(nn.Module):
    """
    The downsampling block for the LEDNet

    So basically we will do two operations in parallel, conv and max pool and then concat both of them
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, bias=True, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_activation = activation

    def forward(self, x):
        main = self.conv(x)
        ext = self.maxpool(x)
        out = torch.cat((main, ext), 1)
        out = self.out_activation(out)
        return out


class Channel_Split(nn.Module):
    """
    Used to split the channels of a tensor
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        x_left = x[:, 0:self.channels // 2, :, :]
        x_right = x[:, self.channels // 2:, :, :]
        return x_left, x_right


class ShuffleBlock(nn.Module):

    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class SSnbt(nn.Module):
    """
    The class for the implementation split-shuffle-non-bottleneck block
    """

    def __init__(self, channels, kernel_size=3, padding=0, groups=4, dilation=1, bias=True, relu=False, drop_prob=0):
        """
        groups: parameter for determining the shuffling order in the shuffleBlock
        """
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        mid_channels = channels // 2
        self.split = Channel_Split(channels)
        self.l_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=1, bias=bias)
        self.l_conv2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=1, bias=bias), nn.BatchNorm2d(mid_channels), activation)
        self.r_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=1, bias=bias)
        self.r_conv2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=1, bias=bias), nn.BatchNorm2d(mid_channels), activation)
        self.l_conv3 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding + dilation - 1, 0), dilation=dilation, bias=bias), activation)
        self.l_conv4 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding + dilation - 1), dilation=dilation, bias=bias), nn.BatchNorm2d(mid_channels), activation)
        self.r_conv3 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding + dilation - 1), dilation=dilation, bias=bias), activation)
        self.r_conv4 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding + dilation - 1, 0), dilation=dilation, bias=bias), nn.BatchNorm2d(mid_channels), activation)
        self.regularizer = nn.Dropout2d(p=drop_prob)
        self.out_activation = activation
        self.shuffle = ShuffleBlock(groups=groups)

    def forward(self, x):
        main = x
        l_input, r_input = self.split(x)
        l_input = self.l_conv1(l_input)
        l_input = self.l_conv2(l_input)
        r_input = self.r_conv1(r_input)
        r_input = self.r_conv2(r_input)
        l_input = self.l_conv3(l_input)
        l_input = self.l_conv4(l_input)
        r_input = self.r_conv3(r_input)
        r_input = self.r_conv4(r_input)
        ext = torch.cat((l_input, r_input), 1)
        ext = self.regularizer(ext)
        out = main + ext
        out = self.out_activation(out)
        out = self.shuffle(out)
        return out


class LEDNet(nn.Module):
    """
    Implementation of the LEDNet network
    """

    def __init__(self, in_channels=3, n_classes=22, encoder_relu=False, decoder_relu=True, image_dim=128):
        super().__init__()
        self.downsample_1 = Downsample_Block_led(in_channels, 32, kernel_size=3, padding=1, relu=encoder_relu)
        self.ssnbt_1_1 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_1_2 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_1_3 = SSnbt(32, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.downsample_2 = Downsample_Block_led(32, 64, kernel_size=3, padding=1, relu=encoder_relu)
        self.ssnbt_2_1 = SSnbt(64, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.ssnbt_2_2 = SSnbt(64, kernel_size=3, padding=1, groups=4, relu=encoder_relu, drop_prob=0.01)
        self.downsample_3 = Downsample_Block_led(64, 128, kernel_size=3, padding=1, relu=encoder_relu)
        self.ssnbt_3_1 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=1, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_2 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=2, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_3 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=5, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_4 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=9, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_5 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=2, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_6 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=5, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_7 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=9, relu=encoder_relu, drop_prob=0.1)
        self.ssnbt_3_8 = SSnbt(128, kernel_size=3, padding=1, groups=8, dilation=17, relu=encoder_relu, drop_prob=0.1)
        self.apn = APN(128, n_classes, image_dim=image_dim // 8, relu=decoder_relu)

    def forward(self, x):
        x = self.downsample_1(x)
        x = self.ssnbt_1_1(x)
        x = self.ssnbt_1_2(x)
        x = self.ssnbt_1_3(x)
        x = self.downsample_2(x)
        x = self.ssnbt_2_1(x)
        x = self.ssnbt_2_2(x)
        x = self.downsample_3(x)
        x = self.ssnbt_3_1(x)
        x = self.ssnbt_3_2(x)
        x = self.ssnbt_3_3(x)
        x = self.ssnbt_3_4(x)
        x = self.ssnbt_3_5(x)
        x = self.ssnbt_3_6(x)
        x = self.ssnbt_3_7(x)
        x = self.ssnbt_3_8(x)
        out = self.apn(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNet(nn.Module):

    def __init__(self, in_channels, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate}, {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros(x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
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
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Channel_Split,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Classifier_Module,
     lambda: ([], {'dilation_series': [4, 4], 'padding_series': [4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {})),
    (ENet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FCDiscriminator,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (GaussianNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InitialBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'dim': 4, 'norm_layer': torch.nn.ReLU, 'use_dropout': 0.5, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ShuffleBlock,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

