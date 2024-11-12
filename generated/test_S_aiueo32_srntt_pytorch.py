
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


import functools


from torch.utils.data import Dataset


from torchvision import transforms


from torchvision.transforms import functional as TF


import random


import numpy as np


import torch


from collections import OrderedDict


from torchvision.utils import save_image


import torch.autograd as autograd


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import warnings


import torchvision.models.vgg as vgg


from torch.utils.data import DataLoader


import pandas as pd


from sklearn.model_selection import train_test_split


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter


class AdversarialLoss(nn.Module):
    """
    PyTorch module for GAN loss.
    This code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
    """

    def __init__(self, gan_mode='wgangp', target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).detach()

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class BackProjectionLoss(nn.Module):

    def __init__(self, scale_factor=4):
        super(BackProjectionLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, y):
        assert x.shape[2] == y.shape[2] * self.scale_factor
        assert x.shape[3] == y.shape[3] * self.scale_factor
        x = F.interpolate(x, y.size()[-2:], mode='bicubic', align_corners=True)
        return F.l1_loss(x, y)


class SSIM(nn.Module):

    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        if x.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
        if y.shape[1] == 3:
            y = kornia.color.rgb_to_grayscale(y)
        return 1 - kornia.losses.ssim(x, y, self.window_size, 'mean')


class PSNR(nn.Module):

    def __init__(self, max_val=1.0, mode='Y'):
        super(PSNR, self).__init__()
        self.max_val = max_val
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'Y' and x.shape[1] == 3 and y.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
            y = kornia.color.rgb_to_grayscale(y)
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr


NAMES = {'vgg11': ['conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg13': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg16': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], 'vgg19': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']}


def insert_bn(names: 'list'):
    """
    Inserts bn layer after each conv.

    Parameters
    ---
    names : list
        The list of layer names.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            pos = name.replace('conv', '')
            names_bn.append('bn' + pos)
    return names_bn


class VGG(nn.Module):
    """
    Creates any type of VGG models.

    Parameters
    ---
    model_type : str
        The model type you want to load.
    requires_grad : bool, optional
        Whethere compute gradients.
    """

    def __init__(self, model_type: 'str', requires_grad: 'bool'=False):
        super(VGG, self).__init__()
        features = getattr(vgg, model_type)(True).features
        self.names = NAMES[model_type.replace('_bn', '')]
        if 'bn' in model_type:
            self.names = insert_bn(self.names)
        self.net = nn.Sequential(OrderedDict([(k, v) for k, v in zip(self.names, features)]))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
        self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    def z_score(self, x):
        x = x.sub(self.vgg_mean.detach())
        x = x.div(self.vgg_std.detach())
        return x

    def forward(self, x: 'torch.Tensor', targets: 'list') ->dict:
        """
        Parameters
        ---
        x : torch.Tensor
            The input tensor normalized to [0, 1].
        target : list of str
            The layer names you want to pick up.
        Returns
        ---
        out_dict : dict of torch.Tensor
            The dictionary of tensors you specified.
            The elements are ordered by the original VGG order. 
        """
        assert all([(t in self.names) for t in targets]), 'Specified name does not exist.'
        if torch.all(x < 0.0) and torch.all(x > 1.0):
            warnings.warn('input tensor is not normalize to [0, 1].')
        x = self.z_score(x)
        out_dict = OrderedDict()
        for key, layer in self.net._modules.items():
            x = layer(x)
            if key in targets:
                out_dict.update({key: x})
            if len(out_dict) == len(targets):
                break
        return out_dict


class PerceptualLoss(nn.Module):
    """
    PyTorch module for perceptual loss.

    Parameters
    ---
    model_type : str
        select from [`vgg11`, `vgg11bn`, `vgg13`, `vgg13bn`,
                     `vgg16`, `vgg16bn`, `vgg19`, `vgg19bn`, ].
    target_layers : str
        the layer name you want to compare.
    norm_type : str
        the type of norm, select from ['mse', 'fro']
    """

    def __init__(self, model_type: 'str'='vgg19', target_layer: 'str'='relu5_1', norm_type: 'str'='fro'):
        super(PerceptualLoss, self).__init__()
        assert norm_type in ['mse', 'fro']
        self.model = VGG(model_type=model_type)
        self.target_layer = target_layer
        self.norm_type = norm_type

    def forward(self, x, y):
        x_feat, *_ = self.model(x, [self.target_layer]).values()
        y_feat, *_ = self.model(y, [self.target_layer]).values()
        if self.norm_type == 'mse':
            loss = F.mse_loss(x_feat, y_feat)
        elif self.norm_type == 'fro':
            loss = torch.norm(x_feat - y_feat, p='fro')
        return loss


def gram_matrix(features):
    N, C, H, W = features.size()
    feat_reshaped = features.view(N, C, -1)
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
    return gram


class TextureLoss(nn.Module):
    """
    creates a criterion to compute weighted gram loss.
    """

    def __init__(self, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights
        self.model = VGG(model_type='vgg19')
        self.register_buffer('a', torch.tensor(-20.0, requires_grad=False))
        self.register_buffer('b', torch.tensor(0.65, requires_grad=False))

    def forward(self, x, maps, weights):
        input_size = x.shape[-1]
        x_feat = self.model(x, ['relu1_1', 'relu2_1', 'relu3_1'])
        if self.use_weights:
            weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
            for idx, l in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                weights_scaled = F.interpolate(weights, None, 2 ** idx, 'bicubic', True)
                coeff = weights_scaled * self.a.detach() + self.b.detach()
                coeff = torch.sigmoid(coeff)
                maps[l] = maps[l] * coeff
                x_feat[l] = x_feat[l] * coeff
        loss_relu1_1 = torch.norm(gram_matrix(x_feat['relu1_1']) - gram_matrix(maps['relu1_1'])) / 4.0 / (input_size * input_size * 1024) ** 2
        loss_relu2_1 = torch.norm(gram_matrix(x_feat['relu2_1']) - gram_matrix(maps['relu2_1'])) / 4.0 / (input_size * input_size * 512) ** 2
        loss_relu3_1 = torch.norm(gram_matrix(x_feat['relu3_1']) - gram_matrix(maps['relu3_1'])) / 4.0 / (input_size * input_size * 256) ** 2
        loss = (loss_relu1_1 + loss_relu2_1 + loss_relu3_1) / 3.0
        return loss


class Discriminator(nn.Sequential):

    def __init__(self, ndf=32):

        def conv_block(in_channels, out_channels):
            block = [nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True), nn.Conv2d(out_channels, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True)]
            return block
        super(Discriminator, self).__init__(*conv_block(3, ndf), *conv_block(ndf, ndf * 2), *conv_block(ndf * 2, ndf * 4), *conv_block(ndf * 4, ndf * 8), *conv_block(ndf * 8, ndf * 16), nn.Conv2d(ndf * 16, 1024, kernel_size=1), nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1), nn.Sigmoid())
        models.init_weights(self, init_type='normal', init_gain=0.02)


class ImageDiscriminator(nn.Sequential):

    def __init__(self, ndf=32):

        def conv_block(in_channels, out_channels):
            block = [nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True), nn.Conv2d(out_channels, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True)]
            return block
        super(ImageDiscriminator, self).__init__(*conv_block(3, ndf), *conv_block(ndf, ndf * 2), *conv_block(ndf * 2, ndf * 4), *conv_block(ndf * 4, ndf * 8), *conv_block(ndf * 8, ndf * 16), nn.AdaptiveAvgPool2d(1), nn.Conv2d(ndf * 16, 1024, kernel_size=1), nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1), nn.Sigmoid())
        models.init_weights(self, init_type='normal', init_gain=0.02)


class ResBlock(nn.Module):
    """
    Basic residual block for SRNTT.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_filters, n_filters, 3, 1, 1), nn.ReLU(True), nn.Conv2d(n_filters, n_filters, 3, 1, 1))

    def forward(self, x):
        return self.body(x) + x


class ContentExtractor(nn.Module):
    """
    Content Extractor for SRNTT, which outputs maps before-and-after upscale.
    more detail: https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L73.
    Currently this module only supports `scale_factor=4`.

    Parameters
    ---
    ngf : int, optional
        a number of generator's features.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.body = nn.Sequential(*[ResBlock(ngf) for _ in range(n_blocks)])
        self.tail = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True), nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True), nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True), nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h


class TextureTransfer(nn.Module):
    """
    Conditional Texture Transfer for SRNTT,
        see https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L116.
    This module is devided 3 parts for each scales.

    Parameters
    ---
    ngf : int
        a number of generator's filters.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16, use_weights=False):
        super(TextureTransfer, self).__init__()
        self.head_small = nn.Sequential(nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.body_small = nn.Sequential(*[ResBlock(ngf) for _ in range(n_blocks)])
        self.tail_small = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
        self.head_medium = nn.Sequential(nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.body_medium = nn.Sequential(*[ResBlock(ngf) for _ in range(n_blocks)])
        self.tail_medium = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
        self.head_large = nn.Sequential(nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True))
        self.body_large = nn.Sequential(*[ResBlock(ngf) for _ in range(n_blocks)])
        self.tail_large = nn.Sequential(nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, True), nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))
        if use_weights:
            self.a = nn.Parameter(torch.ones(3), requires_grad=True)
            self.b = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, x, maps, weights=None):
        if hasattr(self, 'a') and weights is not None:
            for idx, layer in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                weights_scaled = F.interpolate(F.pad(weights, (1, 1, 1, 1), mode='replicate'), scale_factor=2 ** idx, mode='bicubic', align_corners=True) * self.a[idx] + self.b[idx]
                maps[layer] *= torch.sigmoid(weights_scaled)
        h = torch.cat([x, maps['relu3_1']], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)
        h = torch.cat([x, maps['relu2_1']], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)
        h = torch.cat([x, maps['relu1_1']], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)
        return x


class SRNTT(nn.Module):
    """
    PyTorch Module for SRNTT.
    Now x4 is only supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blucks : int, optional
        the number of residual blocks for each module.
    """

    def __init__(self, ngf=64, n_blocks=16, use_weights=False):
        super(SRNTT, self).__init__()
        self.content_extractor = ContentExtractor(ngf, n_blocks)
        self.texture_transfer = TextureTransfer(ngf, n_blocks, use_weights)
        models.init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x, maps, weights=None):
        """
        Parameters
        ---
        x : torch.Tensor
            the input image of SRNTT.
        maps : dict of torch.Tensor
            the swapped feature maps on relu3_1, relu2_1 and relu1_1.
            depths of the maps are 256, 128 and 64 respectively.
        """
        base = F.interpolate(x, None, 4, 'bilinear', False)
        upscale_plain, content_feat = self.content_extractor(x)
        if maps is not None:
            if hasattr(self.texture_transfer, 'a'):
                upscale_srntt = self.texture_transfer(content_feat, maps, weights)
            else:
                upscale_srntt = self.texture_transfer(content_feat, maps)
            return upscale_plain + base, upscale_srntt + base
        else:
            return upscale_plain + base, None


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdversarialLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
    (ContentExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PSNR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PerceptualLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (ResBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
]

