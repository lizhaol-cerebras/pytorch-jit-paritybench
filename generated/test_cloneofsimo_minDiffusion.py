
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


from typing import Dict


from typing import Tuple


import torch


import torch.nn as nn


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torchvision.datasets import MNIST


from torchvision import transforms


from torchvision.utils import save_image


from torchvision.utils import make_grid


from typing import Optional


from torchvision.datasets import ImageFolder


from torchvision.datasets import CIFAR10


def ddpm_schedules(beta1: 'float', beta2: 'float', T: 'int') ->Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, 'beta1 and beta2 must be in (0, 1)'
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {'alpha_t': alpha_t, 'oneover_sqrta': oneover_sqrta, 'sqrt_beta_t': sqrt_beta_t, 'alphabar_t': alphabar_t, 'sqrtab': sqrtab, 'sqrtmab': sqrtmab, 'mab_over_sqrtmab': mab_over_sqrtmab_inv}


class DDPM(nn.Module):

    def __init__(self, eps_model: 'nn.Module', betas: 'Tuple[float, float]', n_T: 'int', criterion: 'nn.Module'=nn.MSELoss()) ->None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        _ts = torch.randint(1, self.n_T, (x.shape[0],))
        eps = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * eps
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: 'int', size, device) ->torch.Tensor:
        x_i = torch.randn(n_sample, *size)
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i


class Conv3(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', is_res: 'bool'=False) ->None:
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.GroupNorm(8, out_channels), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.GroupNorm(8, out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.GroupNorm(8, out_channels), nn.ReLU())
        self.is_res = is_res

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int') ->None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int') ->None:
        super(UnetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 2, 2), Conv3(out_channels, out_channels), Conv3(out_channels, out_channels)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor', skip: 'torch.Tensor') ->torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class TimeSiren(nn.Module):

    def __init__(self, emb_dim: 'int') ->None:
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', n_feat: 'int'=256) ->None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.init_conv = Conv3(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
        self.timeembed = TimeSiren(2 * n_feat)
        self.up0 = nn.Sequential(nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4), nn.GroupNorm(8, 2 * n_feat), nn.ReLU())
        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor') ->torch.Tensor:
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        thro = self.to_vec(down3)
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)
        thro = self.up0(thro + temb)
        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


blk = lambda ic, oc: nn.Sequential(nn.Conv2d(ic, oc, 7, padding=3), nn.BatchNorm2d(oc), nn.LeakyReLU())


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: 'int') ->None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(blk(n_channel, 64), blk(64, 128), blk(128, 256), blk(256, 512), blk(512, 256), blk(256, 128), blk(128, 64), nn.Conv2d(64, n_channel, 3, padding=1))

    def forward(self, x, t) ->torch.Tensor:
        return self.conv(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DummyEpsModel,
     lambda: ([], {'n_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TimeSiren,
     lambda: ([], {'emb_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

