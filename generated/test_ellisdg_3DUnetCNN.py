
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


import math


import torch


import torch.nn


from functools import partial


import numpy as np


import torch.nn as nn


from torch import nn


from torch.nn import functional as F


from torch import nn as nn


import torch.nn.functional as F


from itertools import permutations


import logging


import warnings


from copy import deepcopy


import pandas as pd


import time


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import random


import itertools


from collections.abc import Iterable


from scipy.ndimage.interpolation import map_coordinates


from scipy.ndimage.filters import gaussian_filter


from scipy.ndimage import binary_erosion


from torch.nn import ReplicationPad3d


class VariationalBlock(nn.Module):

    def __init__(self, in_size, n_features, out_size, return_parameters=False):
        super(VariationalBlock, self).__init__()
        self.n_features = n_features
        self.return_parameters = return_parameters
        self.dense1 = nn.Linear(in_size, out_features=n_features * 2)
        self.dense2 = nn.Linear(self.n_features, out_size)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.dense1(x)
        mu, logvar = torch.split(x, self.n_features, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.dense2(z)
        if self.return_parameters:
            return out, mu, logvar, z
        else:
            return out, mu, logvar


class MyronenkoLayer(nn.Module):

    def __init__(self, n_blocks, block, in_planes, planes, *args, dropout=None, kernel_size=3, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(block(in_planes, planes, *args, kernel_size=kernel_size, **kwargs))
            in_planes = planes
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, padding=None, dilation=1, kernel_size=3):
    """3x3x3 convolution with padding"""
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)


class MyronenkoConvolutionBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoConvolutionBlock, self).__init__()
        self.norm_groups = norm_groups
        if norm_layer is None:
            self.norm_layer = nn.GroupNorm
        else:
            self.norm_layer = norm_layer
        self.norm1 = self.create_norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3x3(in_planes, planes, stride, kernel_size=kernel_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

    def create_norm_layer(self, planes, error_on_non_divisible_norm_groups=False):
        if planes < self.norm_groups:
            return self.norm_layer(planes, planes)
        elif not error_on_non_divisible_norm_groups and planes % self.norm_groups > 0:
            None
            return self.norm_layer(planes, planes)
        else:
            return self.norm_layer(self.norm_groups, planes)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MyronenkoResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoResidualBlock, self).__init__()
        self.conv1 = MyronenkoConvolutionBlock(in_planes=in_planes, planes=planes, stride=stride, norm_layer=norm_layer, norm_groups=norm_groups, kernel_size=kernel_size)
        self.conv2 = MyronenkoConvolutionBlock(in_planes=planes, planes=planes, stride=stride, norm_layer=norm_layer, norm_groups=norm_groups, kernel_size=kernel_size)
        if in_planes != planes:
            self.sample = conv1x1x1(in_planes, planes)
        else:
            self.sample = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.sample is not None:
            identity = self.sample(identity)
        x += identity
        return x


class MirroredDecoder(nn.Module):

    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock, upsampling_scale=2, feature_reduction_scale=2, upsampling_mode='trilinear', align_corners=False, layer_widths=None, use_transposed_convolutions=False, kernel_size=3):
        super(MirroredDecoder, self).__init__()
        self.use_transposed_convolutions = use_transposed_convolutions
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)
            if depth != 0:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=in_width, kernel_size=kernel_size))
                if self.use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernel_size, stride=upsampling_scale, padding=1))
                else:
                    self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale, mode=upsampling_mode, align_corners=align_corners))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width, kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        if self.layer_widths is not None:
            out_width = self.layer_widths[depth]
            in_width = self.layer_widths[depth + 1]
        elif depth > 0:
            out_width = int(self.base_width * self.feature_reduction_scale ** (depth - 1))
            in_width = out_width * self.feature_reduction_scale
        else:
            out_width = self.base_width
            in_width = self.base_width
        return in_width, out_width

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1]):
            x = lay(x)
            x = pre(x)
            x = up(x)
        x = self.layers[-1](x)
        return x


class MyronenkoDecoder(nn.Module):

    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock, upsampling_scale=2, feature_reduction_scale=2, upsampling_mode='trilinear', align_corners=False, layer_widths=None, use_transposed_convolutions=False, kernal_size=3):
        super(MyronenkoDecoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 1, 1]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            if layer_widths is not None:
                out_width = layer_widths[depth]
                in_width = layer_widths[depth + 1]
            else:
                out_width = base_width * feature_reduction_scale ** depth
                in_width = out_width * feature_reduction_scale
            if use_transposed_convolutions:
                self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale, mode=upsampling_mode, align_corners=align_corners))
            else:
                self.pre_upsampling_blocks.append(nn.Sequential())
                self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernal_size, stride=upsampling_scale, padding=1))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=out_width, planes=out_width, kernal_size=kernal_size))

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers):
            x = pre(x)
            x = up(x)
            x = lay(x)
        return x


class MyronenkoEncoder(nn.Module):

    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock, feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_width = layer_widths[i]
            else:
                out_width = base_width * feature_dilation ** i
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width, dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride, kernel_size=kernel_size))
            None
            in_width = out_width

    def forward(self, x):
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            x = downsampling(x)
        x = self.layers[-1](x)
        return x


class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None, feature_dilation=2, downsampling_stride=2, interpolation_mode='trilinear', encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.base_width = base_width
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder_class(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks, feature_dilation=feature_dilation, downsampling_stride=downsampling_stride, layer_widths=layer_widths, kernel_size=kernel_size)
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks)
        self.decoder = decoder_class(base_width=base_width, layer_blocks=decoder_blocks, upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation, upsampling_mode=interpolation_mode, layer_widths=layer_widths, use_transposed_convolutions=use_transposed_convolutions, kernel_size=kernel_size)
        self.set_final_convolution(n_features)
        self.set_activation(activation=activation)

    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            if decoder_class is None:
                decoder_class = MirroredDecoder
        elif decoder_blocks is None:
            decoder_blocks = [1] * len(encoder_blocks)
            if decoder_class is None:
                decoder_class = MyronenkoDecoder
        return decoder_class, decoder_blocks

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MyronenkoVariationalLayer(nn.Module):

    def __init__(self, in_features, input_shape, reduced_features=16, latent_features=128, conv_block=MyronenkoConvolutionBlock, conv_stride=2, upsampling_mode='trilinear', align_corners_upsampling=False):
        super(MyronenkoVariationalLayer, self).__init__()
        self.in_conv = conv_block(in_planes=in_features, planes=reduced_features, stride=conv_stride)
        self.reduced_shape = tuple(np.asarray((reduced_features, *np.divide(input_shape, conv_stride)), dtype=np.int))
        self.in_size = np.prod(self.reduced_shape, dtype=np.int)
        self.var_block = VariationalBlock(in_size=self.in_size, out_size=self.in_size, n_features=latent_features)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = conv1x1x1(in_planes=reduced_features, out_planes=in_features, stride=1)
        self.upsample = partial(nn.functional.interpolate, scale_factor=conv_stride, mode=upsampling_mode, align_corners=align_corners_upsampling)

    def forward(self, x):
        x = self.in_conv(x).flatten(start_dim=1)
        x, mu, logvar = self.var_block(x)
        x = self.relu(x).view(-1, *self.reduced_shape)
        x = self.out_conv(x)
        x = self.upsample(x)
        return x, mu, logvar


class VariationalAutoEncoder(ConvolutionalAutoEncoder):

    def __init__(self, n_reduced_latent_feature_maps=16, vae_features=128, variational_layer=MyronenkoVariationalLayer, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None, feature_dilation=2, downsampling_stride=2, interpolation_mode='trilinear', encoder_class=MyronenkoEncoder, decoder_class=MyronenkoDecoder, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, var_layer_stride=2):
        super(VariationalAutoEncoder, self).__init__(input_shape=input_shape, n_features=n_features, base_width=base_width, encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks, feature_dilation=feature_dilation, downsampling_stride=downsampling_stride, interpolation_mode=interpolation_mode, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, layer_widths=layer_widths, decoder_mirrors_encoder=decoder_mirrors_encoder, activation=activation, use_transposed_convolutions=use_transposed_convolutions)
        if vae_features is not None:
            depth = len(encoder_blocks) - 1
            n_latent_feature_maps = base_width * feature_dilation ** depth
            latent_image_shape = np.divide(input_shape, downsampling_stride ** depth)
            self.var_layer = variational_layer(in_features=n_latent_feature_maps, input_shape=latent_image_shape, reduced_features=n_reduced_latent_feature_maps, latent_features=vae_features, upsampling_mode=interpolation_mode, conv_stride=var_layer_stride)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar

    def test(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(mu)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar


class LabeledVariationalAutoEncoder(VariationalAutoEncoder):

    def __init__(self, *args, n_outputs=None, base_width=32, **kwargs):
        super().__init__(*args, n_outputs=n_outputs, base_width=base_width, **kwargs)
        self.final_convolution = conv1x1x1(in_planes=base_width, out_planes=n_outputs, stride=1)


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(1 - self.decay, embed_onehot.sum(0))
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(inplace=True), nn.Conv3d(in_channel, channel, 3, padding=1), nn.ReLU(inplace=True), nn.Conv3d(channel, in_channel, 1))

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv3d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv3d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv3d(channel // 2, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv3d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([nn.ConvTranspose3d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose3d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose3d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):

    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv3d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv3d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose3d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 4, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 4, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 4, 1, 2, 3)
        diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 4, 1, 2, 3)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 4, 1, 2, 3)
        dec = self.decode(quant_t, quant_b)
        return dec


class RegularizedResNet(VariationalAutoEncoder):

    def __init__(self, n_outputs, *args, **kwargs):
        super(RegularizedResNet, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.var_layer.in_size, n_outputs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.var_layer.in_conv(x).flatten(start_dim=1)
        output = self.fc(x)
        x, mu, logvar = self.var_layer.var_block(x)
        x = self.var_layer.relu(x).view(-1, *self.var_layer.reduced_shape)
        x = self.var_layer.out_conv(x)
        x = self.var_layer.upsample(x)
        x = self.decoder(x)
        vae_output = self.final_convolution(x)
        return output, vae_output, mu, logvar


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm3d
        else:
            self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = self.create_norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = self.create_norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def create_norm_layer(self, *args, **kwargs):
        return self.norm_layer(*args, **kwargs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm3d
        else:
            self.norm_layer = norm_layer
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1x1(in_planes, width)
        self.bn1 = self.create_norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = self.create_norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = self.create_norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def create_norm_layer(self, *args, **kwargs):
        return self.norm_layer(*args, **kwargs)


class ResNet(nn.Module):

    def __init__(self, block, layers, n_outputs=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, n_features=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(n_features, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, n_outputs)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class _ResNetLatent(ResNet):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        latent = x
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, latent


class RegularizedBasicResNet(nn.Module):

    def __init__(self, n_features, upsampling_mode='trilinear', upsampling_scale=2, plane_dilation=2, decoding_layers=None, latent_planes=512, layer_block=BasicBlock, **encoder_kwargs):
        super(RegularizedBasicResNet, self).__init__()
        if decoding_layers is None:
            decoding_layers = [1, 1, 1, 1, 1, 1, 1]
        self.encoder = _ResNetLatent(block=layer_block, n_features=n_features, **encoder_kwargs)
        self.decoder = BasicDecoder(upsampling_scale=upsampling_scale, upsampling_mode=upsampling_mode, plane_dilation=plane_dilation, layers=decoding_layers, in_planes=latent_planes, block=layer_block)
        out_decoder_planes = int(latent_planes / plane_dilation ** len(decoding_layers))
        self.final_convolution = conv1x1x1(in_planes=out_decoder_planes, out_planes=n_features, stride=1)

    def forward(self, x):
        out, x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return out, x


class ResNetWithDecoder1D(nn.Module):

    def __init__(self, n_fc_outputs, n_outputs, initial_upsample=1024, blocks_per_layer=1, channel_decay=2, upsample_factor=2, resnet_block=BasicBlock, interpolation_mode='linear', interpolation_align_corners=True, **kwargs):
        super(ResNetWithDecoder1D, self).__init__()
        self.encoder = ResNet(n_outputs=n_fc_outputs, block=resnet_block, **kwargs)
        self.initial_upsample = initial_upsample
        _size = initial_upsample
        _channels = n_fc_outputs
        layer_blocks = list()
        layer_channels = list()
        while _size < n_outputs:
            _size = int(_size * upsample_factor)
            _channels = int(_channels / channel_decay)
            layer_blocks.append(blocks_per_layer)
            layer_channels.append(_channels)
        self.decoder = Decoder1D(input_features=n_fc_outputs, output_features=n_outputs, layer_blocks=layer_blocks, layer_channels=layer_channels, upsample_factor=upsample_factor, interpolation_mode=interpolation_mode, interpolation_align_corners=interpolation_align_corners)
        self.out_conv = nn.Conv1d(in_channels=layer_channels[-1], out_channels=1, kernel_size=3, bias=False)
        self.output_features = n_outputs
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x.flatten(start_dim=1)[..., None], size=(self.initial_upsample,))
        x = self.decoder(x)
        x = self.out_conv(x)
        return nn.functional.interpolate(x, size=(self.output_features,), mode=self.interpolation_mode, align_corners=self.interpolation_align_corners)


class BasicBlock1D(BasicBlock):

    def __init__(self, in_channels, channels, stride=1, downsample=None, kernel_size=3, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm1d
        else:
            self.norm_layer = norm_layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, stride=stride, kernel_size=kernel_size, bias=False, padding=1)
        self.bn1 = self.create_norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, stride=stride, kernel_size=kernel_size, bias=False, padding=1)
        self.bn2 = self.create_norm_layer(channels)
        self.downsample = downsample
        self.stride = stride


class FCN(nn.Module):

    def __init__(self, hidden_layers_list, n_inputs, n_outputs):
        super().__init__()
        _layers = list()
        _n_inputs = n_inputs
        for hidden_layer in hidden_layers_list:
            _layer = nn.Linear(_n_inputs, hidden_layer)
            _n_inputs = hidden_layer
            _layers.append(_layer)
            _layers.append(nn.ReLU())
        _layers.append(nn.Linear(_n_inputs, n_outputs))
        self.network = nn.Sequential(*_layers)

    def forward(self, x):
        return self.network(x)


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, adjacency_matrix_wrapper, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix_wrapper = adjacency_matrix_wrapper
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 6.0 / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjacency_matrix_wrapper.adjacency_matrix, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                output.append(spmm(self.adjacency_matrix_wrapper.adjacency_matrix, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphLinear(nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """

    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]


class GraphResBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, adjacency_matrix_wrapper):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, adjacency_matrix_wrapper)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm(out_channels // 2 // 8, out_channels // 2)
        self.norm2 = nn.GroupNorm(out_channels // 2 // 8, out_channels // 2)

    def forward(self, x):
        y = F.relu(self.pre_norm(x))
        y = self.lin1(y)
        y = F.relu(self.norm1(y))
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        y = F.relu(self.norm2(y))
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x + y


class AdjacencyMatrixWrapper(object):

    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix


def faces_to_edges(faces):
    edges = list()
    for face in faces:
        edges.extend(list(permutations(face, 2)))
    return torch.LongTensor(edges).t()


def faces_to_adjacency_matrix(faces, size):
    indices = faces_to_edges(faces)
    values = torch.zeros(indices.shape[1], dtype=torch.float)
    adjacency_matrix = torch.sparse.FloatTensor(indices, values, size=size)
    return adjacency_matrix


def load_surface(surface_filename):
    surface = nib.load(os.path.abspath(surface_filename))
    vertices = surface.darrays[0].data
    n_vertices = vertices.shape[0]
    faces = surface.darrays[1].data
    adjacency_matrix = faces_to_adjacency_matrix(faces, size=(n_vertices, n_vertices))
    return torch.FloatTensor(vertices).t(), adjacency_matrix


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet_18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


class GraphCMR(nn.Module):

    def __init__(self, n_outputs=None, ref_vertices=None, adjacency_matrix=None, n_layers=5, n_channels=256, output_features=3, encoder=resnet_18, encoder_outputs=512, reference_filename=None, **encoder_kwargs):
        super(GraphCMR, self).__init__()
        if reference_filename is not None and (ref_vertices is None or adjacency_matrix is None):
            ref_vertices, adjacency_matrix = load_surface(surface_filename=reference_filename)
        self.adjacency_matrix_wrapper = AdjacencyMatrixWrapper(adjacency_matrix)
        self.ref_vertices = ref_vertices
        self.encoder = encoder(n_outputs=encoder_outputs, **encoder_kwargs)
        self.encoder_outputs = encoder_outputs
        layers = [GraphLinear(3 + self.encoder_outputs, 2 * n_channels), GraphResBlock(2 * n_channels, n_channels, self.adjacency_matrix_wrapper)]
        for i in range(n_layers):
            layers.append(GraphResBlock(n_channels, n_channels, self.adjacency_matrix_wrapper))
        self.shape = nn.Sequential(GraphResBlock(n_channels, 64, self.adjacency_matrix_wrapper), GraphResBlock(64, 32, self.adjacency_matrix_wrapper), nn.GroupNorm(32 // 8, 32), nn.ReLU(inplace=True), GraphLinear(32, output_features))
        self.gc = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass
        Inputs:
            x: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = x.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        x = self.encoder(x)
        x = x.view(batch_size, self.encoder_outputs, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, x], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        return shape

    def cuda(self, *args, **kwargs):
        self.ref_vertices = self.ref_vertices
        self.adjacency_matrix_wrapper.adjacency_matrix = self.adjacency_matrix_wrapper.adjacency_matrix
        return super(GraphCMR, self)


class UNetEncoder(MyronenkoEncoder):

    def forward(self, x):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return outputs


class UNetDecoder(MirroredDecoder):

    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        None
        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0]
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            x = pre(x)
            x = up(x)
            diffZ = inputs[i + 1].size()[2] - x.size()[2]
            diffY = inputs[i + 1].size()[3] - x.size()[3]
            diffX = inputs[i + 1].size()[4] - x.size()[4]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat((x, inputs[i + 1]), 1)
        x = self.layers[-1](x)
        return x


class UNet3D(ConvolutionalAutoEncoder):

    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutocastUNet(UNet3D):

    def forward(self, *args, **kwargs):
        from torch.cuda.amp import autocast
        with autocast():
            output = super().forward(*args, **kwargs)
        return output


class AutoImplantUNet(UNet3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (BasicBlock1D,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Decoder,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'channel': 4, 'n_res_block': 4, 'n_res_channel': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FCN,
     lambda: ([], {'hidden_layers_list': [4, 4], 'n_inputs': 4, 'n_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GraphLinear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MyronenkoConvolutionBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MyronenkoResidualBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Quantize,
     lambda: ([], {'dim': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VariationalBlock,
     lambda: ([], {'in_size': 4, 'n_features': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

