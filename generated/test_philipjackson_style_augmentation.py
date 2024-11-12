
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


from torchvision.transforms import ToTensor


from torchvision.transforms import ToPILImage


import matplotlib.pyplot as plt


import numpy as np


import torchvision


import torchvision.transforms as transforms


from torch.utils.data.dataloader import DataLoader


from torchvision.datasets import ImageFolder


from torchvision.transforms import Compose


from torchvision.transforms import Resize


import torch.nn as nn


import torch.nn.functional as F


class ConvInRelu(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        super(ConvInRelu, self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size / 2)))
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x


class UpsampleConvInRelu(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size, upsample, stride=1, activation=nn.ReLU):
        super(UpsampleConvInRelu, self).__init__()
        self.n_params = channels_out * 2
        self.upsample = upsample
        self.channels = channels_out
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.fc_beta = nn.Linear(100, channels_out)
        self.fc_gamma = nn.Linear(100, channels_out)
        if activation:
            self.activation = activation(inplace=False)
        else:
            self.activation = None

    def forward(self, x, style):
        beta = self.fc_beta(style).unsqueeze(2).unsqueeze(3)
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3)
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x
        x += beta
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.n_params = channels * 4
        self.channels = channels
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels)
        self.fc_beta1 = nn.Linear(100, channels)
        self.fc_gamma1 = nn.Linear(100, channels)
        self.fc_beta2 = nn.Linear(100, channels)
        self.fc_gamma2 = nn.Linear(100, channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=0)

    def forward(self, x, style):
        beta1 = self.fc_beta1(style).unsqueeze(2).unsqueeze(3)
        gamma1 = self.fc_gamma1(style).unsqueeze(2).unsqueeze(3)
        beta2 = self.fc_beta2(style).unsqueeze(2).unsqueeze(3)
        gamma2 = self.fc_gamma2(style).unsqueeze(2).unsqueeze(3)
        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        y = gamma1 * y
        y += beta1
        y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        y = gamma2 * y
        y += beta2
        return x + y


class Ghiasi(nn.Module):

    def __init__(self):
        super(Ghiasi, self).__init__()
        self.layers = nn.ModuleList([ConvInRelu(3, 32, 9, stride=1), ConvInRelu(32, 64, 3, stride=2), ConvInRelu(64, 128, 3, stride=2), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), ResidualBlock(128), UpsampleConvInRelu(128, 64, 3, upsample=2), UpsampleConvInRelu(64, 32, 3, upsample=2), UpsampleConvInRelu(32, 3, 9, upsample=None, activation=None)])
        self.n_params = sum([layer.n_params for layer in self.layers])

    def forward(self, x, styles):
        for i, layer in enumerate(self.layers):
            if i < 3:
                x = layer(x)
            else:
                x = layer(x, styles)
        return torch.sigmoid(x)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class StylePredictor(nn.Module):

    def __init__(self):
        super(StylePredictor, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.fc = nn.Linear(768, 100)

    def forward(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.255
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = x.mean(dim=3).mean(dim=2)
        x = self.fc(x)
        return x


class StyleAugmentor(nn.Module):

    def __init__(self):
        super(StyleAugmentor, self).__init__()
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()
        self.ghiasi
        self.stylePredictor
        checkpoint_ghiasi = torch.load(join(dirname(__file__), 'checkpoints/checkpoint_transformer.pth'))
        checkpoint_stylepredictor = torch.load(join(dirname(__file__), 'checkpoints/checkpoint_stylepredictor.pth'))
        checkpoint_embeddings = torch.load(join(dirname(__file__), 'checkpoints/checkpoint_embeddings.pth'))
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'], strict=False)
        self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'], strict=False)
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean']
        self.imagenet_embedding = self.imagenet_embedding
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        u, s, vh = np.linalg.svd(self.cov.numpy())
        self.A = np.matmul(u, np.diag(s ** 0.5))
        self.A = torch.tensor(self.A).float()

    def sample_embedding(self, n):
        embedding = torch.randn(n, 100)
        embedding = torch.mm(embedding, self.A.transpose(1, 0)) + self.mean
        return embedding

    def forward(self, x, alpha=0.5, downsamples=0, embedding=None, useStylePredictor=True):
        base = self.stylePredictor(x) if useStylePredictor else self.imagenet_embedding
        if downsamples:
            assert x.size(2) % 2 ** downsamples == 0
            assert x.size(3) % 2 ** downsamples == 0
            for i in range(downsamples):
                x = nn.functional.avg_pool2d(x, 2)
        if embedding is None:
            embedding = self.sample_embedding(x.size(0))
        embedding = alpha * embedding + (1 - alpha) * base
        restyled = self.ghiasi(x, embedding)
        if downsamples:
            restyled = nn.functional.upsample(restyled, scale_factor=2 ** downsamples, mode='bilinear')
        return restyled.detach()


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvInRelu,
     lambda: ([], {'channels_in': 4, 'channels_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionE,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StylePredictor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

