
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


import matplotlib.pyplot as plt


import copy


import torch.nn as nn


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


from torch.nn import init


from torch.optim import lr_scheduler


from torchvision import transforms


from torch.utils.tensorboard import SummaryWriter


import logging


class Conv2d(nn.Module):

    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Norm2d(nn.Module):

    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):

    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class CNR2d(nn.Module):

    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()
        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True
        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if norm != []:
            layers += [Norm2d(nch_out, norm)]
        if relu != []:
            layers += [ReLU(relu)]
        if drop != []:
            layers += [nn.Dropout2d(drop)]
        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Deconv2d(nn.Module):

    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        return self.deconv(x)


class DECNR2d(nn.Module):

    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()
        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True
        layers = []
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]
        if norm != []:
            layers += [Norm2d(nch_out, norm)]
        if relu != []:
            layers += [ReLU(relu)]
        if drop != []:
            layers += [nn.Dropout2d(drop)]
        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


class Padding(nn.Module):

    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self.padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)


class ResBlock(nn.Module):

    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()
        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True
        layers = []
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]
        if drop != []:
            layers += [nn.Dropout2d(drop)]
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]
        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)


class CNR1d(nn.Module):

    def __init__(self, nch_in, nch_out, norm='bnorm', relu=0.0, drop=[]):
        super().__init__()
        if norm == 'bnorm':
            bias = False
        else:
            bias = True
        layers = []
        layers += [nn.Linear(nch_in, nch_out, bias=bias)]
        if norm != []:
            layers += [Norm2d(nch_out, norm)]
        if relu != []:
            layers += [ReLU(relu)]
        if drop != []:
            layers += [nn.Dropout2d(drop)]
        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Linear(nn.Module):

    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class Pooling2d(nn.Module):

    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()
        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):

    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()
        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest')
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])
        return torch.cat([x2, x1], dim=1)


class TV1dLoss(nn.Module):

    def __init__(self):
        super(TV1dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))
        return loss


class TV2dLoss(nn.Module):

    def __init__(self):
        super(TV2dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss


class SSIM2dLoss(nn.Module):

    def __init__(self):
        super(SSIM2dLoss, self).__init__()

    def forward(self, input, targer):
        loss = 0
        return loss


class UNet(nn.Module):

    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        """
        Encoder part
        """
        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.pool1 = Pooling2d(pool=2, type='avg')
        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.pool2 = Pooling2d(pool=2, type='avg')
        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.pool3 = Pooling2d(pool=2, type='avg')
        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.pool4 = Pooling2d(pool=2, type='avg')
        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        """
        Decoder part
        """
        self.dec5_1 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.unpool4 = UnPooling2d(pool=2, type='nearest')
        self.dec4_2 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4_1 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.unpool3 = UnPooling2d(pool=2, type='nearest')
        self.dec3_2 = DECNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3_1 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.unpool2 = UnPooling2d(pool=2, type='nearest')
        self.dec2_2 = DECNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2_1 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.unpool1 = UnPooling2d(pool=2, type='nearest')
        self.dec1_2 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1_1 = DECNR2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=3, stride=1, norm=[], relu=[], drop=[], bias=False)

    def forward(self, x):
        """
        Encoder part
        """
        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)
        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)
        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)
        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)
        enc5 = self.enc5_1(pool4)
        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)
        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))
        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))
        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))
        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))
        x = dec1
        return x


class ResNet(nn.Module):

    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=16):
        super(ResNet, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk
        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        self.enc1 = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=[], relu=0.0)
        res = []
        for i in range(self.nblk):
            res += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
        self.res = nn.Sequential(*res)
        self.dec1 = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=norm, relu=[])
        self.conv1 = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x0 = x
        x = self.res(x)
        x = self.dec1(x)
        x = x + x0
        x = self.conv1(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()
        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm
        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        self.dsc1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1, kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):
        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CNR1d,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CNR2d,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Concat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Conv2d,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DECNR2d,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Deconv2d,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Discriminator,
     lambda: ([], {'nch_in': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (Linear,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Padding,
     lambda: ([], {'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pooling2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReLU,
     lambda: ([], {'relu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SSIM2dLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TV1dLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TV2dLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'nch_in': 4, 'nch_out': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (UnPooling2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

