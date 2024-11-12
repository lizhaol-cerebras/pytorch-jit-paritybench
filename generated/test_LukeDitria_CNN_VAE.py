
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


import torch.nn.functional as F


import torch


import torch.nn as nn


import torch.utils.data


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torchvision.datasets as Datasets


import torchvision.transforms as transforms


import torchvision.utils as vutils


from collections import defaultdict


def get_norm_layer(channels, norm_type='bn'):
    if norm_type == 'bn':
        return nn.BatchNorm2d(channels, eps=0.0001)
    elif norm_type == 'gn':
        return nn.GroupNorm(8, channels, eps=0.0001)
    else:
        ValueError('norm_type must be bn or gn')


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type='bn'):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2 + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type='bn'):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, channel_in // 2 + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type='bn'):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        first_out = channel_in // 2 if channel_in == channel_out else channel_in // 2 + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type='bn', deep_model=False):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)
        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]
        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):
            if deep_model:
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))
        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu
        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type='bn', deep_model=False):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]
        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)
        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))
        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))
        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type='bn', deep_model=False):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels, num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels, num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var


class Res_down(nn.Module):

    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_down, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        x = F.rrelu(x + skip)
        return x


class Res_up(nn.Module):

    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_up, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.UpNN = nn.Upsample(scale_factor=scale, mode='nearest')

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        x = F.rrelu(x + skip)
        return x


class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)
        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.load_params_()

    def load_params_(self):
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for (name, source_param), target_param in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)
        return loss / 16


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Decoder,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 256, 8, 8])], {})),
    (Encoder,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResDown,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResUp,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Res_down,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Res_up,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VAE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

