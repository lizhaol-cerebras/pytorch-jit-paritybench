import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
model = _module
run = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
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


from torch.utils import data


from torchvision import transforms as T


from torchvision.datasets import ImageFolder


import torch


import random


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import math


import torch.optim as optim


from torch import cuda


import time


from torch.backends import cudnn


from scipy.linalg import block_diag


from torch.autograd import Variable


from random import *


from torch.optim import lr_scheduler


import torch.nn.init as init


from torchvision.utils import save_image


from torchvision.utils import make_grid


import torchvision.utils as vutils


class ConvBlock(nn.Module):

    def __init__(self, input_dim, output_dim, k, s, p, dilation=False, norm='in', n_group=32, activation='relu', pad_type='mirror', use_affine=True, use_bias=True):
        super(ConvBlock, self).__init__()
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim, affine=use_affine, track_running_stats=True)
        elif norm == 'ln':
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)
        elif norm == 'none':
            self.norm = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        if pad_type == 'mirror':
            self.pad = nn.ReflectionPad2d(p)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(p)
        if dilation:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, dilation=p, bias=use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, bias=use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim, norm='in', n_group=32, activation='relu', use_affine=True):
        super(ResidualBlock, self).__init__()
        layers = []
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation=activation, use_affine=use_affine)]
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation='none', use_affine=use_affine)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)


class Content_Encoder(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=4, norm='in', activation='relu'):
        super(Content_Encoder, self).__init__()
        layers = []
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm=norm, activation=activation)]
        curr_dim = conv_dim
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim * 2, 4, 2, 1, norm=norm, activation=activation)]
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm, activation=activation)]
        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class Style_Encoder(nn.Module):

    def __init__(self, conv_dim=64, n_group=32, norm='ln', activation='relu'):
        super(Style_Encoder, self).__init__()
        curr_dim = conv_dim
        layers = []
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm='none', n_group=n_group, activation=activation)]
        curr_dim = conv_dim
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim * 2, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
            curr_dim = curr_dim * 2
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm='ln', n_group=32, activation='relu', use_affine=True):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        if norm == 'ln':
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)
        elif norm == 'none':
            self.norm = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim, num_block=1, norm='none', n_group=32, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        curr_dim = dim
        layers += [LinearBlock(input_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]
        for _ in range(num_block):
            layers += [LinearBlock(curr_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]
        layers += [LinearBlock(curr_dim, output_dim, norm='none', activation='none')]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))


class Get(object):

    def __init__(self, s_CT, C, G, mask):
        self.s_CT = s_CT
        self.C = C
        self.G = G
        self.mask = mask
        self.n_mem = C // G

    def coloring(self):
        X = []
        U_arr = []
        for i in range(self.G):
            s_CT_i = self.s_CT[:, self.n_mem ** 2 * i:self.n_mem ** 2 * (i + 1)].unsqueeze(2).view(self.s_CT.size(0), self.n_mem, self.n_mem)
            D = torch.sum(s_CT_i ** 2, dim=1, keepdim=True) ** 0.5
            U_i = s_CT_i / D
            UDU_T_i = torch.bmm(s_CT_i, U_i.permute(0, 2, 1))
            X += [UDU_T_i]
            U_arr += [U_i]
        eigen_s = torch.cat(U_arr, dim=0)
        X = torch.cat(X, dim=1)
        X = X.repeat(1, 1, self.G)
        X = self.mask * X
        return X, eigen_s


class WCT(nn.Module):

    def __init__(self, n_group, device, input_dim, mlp_dim, bias_dim, mask, w_alpha=0.4):
        super(WCT, self).__init__()
        self.G = n_group
        self.device = device
        self.alpha = nn.Parameter(torch.ones(1) - w_alpha)
        self.mlp_CT = MLP(input_dim // n_group, (input_dim // n_group) ** 2, dim=mlp_dim, num_block=3, norm='none', n_group=n_group, activation='lrelu')
        self.mlp_mu = MLP(input_dim, bias_dim, dim=input_dim, num_block=1, norm='none', n_group=n_group, activation='lrelu')
        self.mask = mask

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def wct(self, c_A, s_B):
        """
        style_size torch.Size([1, 766])
        mask_size torch.Size([1, 1, 64, 64])
        content_size torch.Size([1, 256, 64, 64])
        W_size torch.size([1,256,256])
        """
        B, C, H, W = c_A.size()
        n_mem = C // self.G
        s_B_CT = self.mlp_CT(s_B.view(B * self.G, C // self.G, 1, 1)).view(B, -1)
        s_B_mu = self.mlp_mu(s_B).unsqueeze(2).unsqueeze(3)
        X_B, eigen_s = Get(s_B_CT, c_A.size(1), self.G, self.mask).coloring()
        eps = 1e-05
        c_A_ = c_A.permute(1, 0, 2, 3).contiguous().view(self.G, n_mem, -1)
        c_A_mean = torch.mean(c_A_, dim=2, keepdim=True)
        c_A_ = c_A_ - c_A_mean
        cov_c = torch.bmm(c_A_, c_A_.permute(0, 2, 1)).div(B * H * W - 1) + eps * torch.eye(n_mem).unsqueeze(0)
        whitend = c_A_.unsqueeze(0).contiguous().view(C, B, -1).permute(1, 0, 2)
        colored_B = torch.bmm(X_B, whitend).unsqueeze(3).view(B, C, H, -1)
        return self.alpha * (colored_B + s_B_mu) + (1 - self.alpha) * c_A, cov_c, eigen_s


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Decoder(nn.Module):

    def __init__(self, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4, norm='ln', device=None):
        super(Decoder, self).__init__()
        curr_dim = input_dim
        self.resblocks = nn.ModuleList([ResidualBlock(dim=curr_dim, norm='none', n_group=n_group) for i in range(repeat_num)])
        self.gdwct_modules = nn.ModuleList([WCT(n_group, device, input_dim, mlp_dim, bias_dim, mask) for i in range(repeat_num + 1)])
        layers = []
        for i in range(2):
            layers += [Upsample(scale_factor=2, mode='nearest')]
            layers += [ConvBlock(curr_dim, curr_dim // 2, 5, 1, 2, norm=norm, n_group=n_group)]
            curr_dim = curr_dim // 2
        layers += [ConvBlock(curr_dim, 3, 7, 1, 3, norm='none', activation='tanh')]
        self.main = nn.Sequential(*layers)

    def forward(self, c_A, s_B):
        whitening_reg = []
        coloring_reg = []
        for i, resblock in enumerate(self.resblocks):
            if i == 0:
                c_A, cov, eigen_s = self.gdwct_modules[i](c_A, s_B)
                whitening_reg += [cov]
                coloring_reg += [eigen_s]
            c_A = resblock(c_A)
            c_A, cov, eigen_s = self.gdwct_modules[i + 1](c_A, s_B)
            whitening_reg += [cov]
            coloring_reg += [eigen_s]
        return self.main(c_A), whitening_reg, coloring_reg


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=8, mask=None, n_group=16, mlp_dim=256, bias_dim=512, content_dim=256, device=None):
        super(Generator, self).__init__()
        self.c_encoder = Content_Encoder(conv_dim, repeat_num // 2, norm='in', activation='relu')
        self.s_encoder = Style_Encoder(conv_dim, n_group, norm='gn', activation='relu')
        self.decoder = Decoder(content_dim, mask, n_group, bias_dim, mlp_dim, repeat_num // 2, norm='ln', device=device)

    def forward(self, c_A, s_B_):
        return self.decoder(c_A, s_B_)


class Discriminator(nn.Module):

    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.n_layer = params['N_LAYER']
        self.gan_type = params['GAN_TYPE']
        self.dim = params['FIRST_DIM']
        self.norm = params['NORM']
        self.activ = params['ACTIVATION']
        self.num_scales = params['NUM_SCALES']
        self.pad_type = params['PAD_TYPE']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [ConvBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [ConvBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) + F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        outs0 = self.forward(input_fake)
        loss = 0
        for it, out0 in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, 'Unsupported GAN type: {}'.format(self.gan_type)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Content_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (Decoder,
     lambda: ([], {'input_dim': 4, 'mask': 4, 'n_group': 4, 'bias_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (LinearBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Style_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (WCT,
     lambda: ([], {'n_group': 4, 'device': 0, 'input_dim': 4, 'mlp_dim': 4, 'bias_dim': 4, 'mask': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_WonwoongCho_GDWCT(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

