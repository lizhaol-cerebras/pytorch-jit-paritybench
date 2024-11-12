
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


import random


from torch.utils.data import Dataset


import warnings


import time


import torch.nn.functional as F


import logging


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


import math


from torch.optim.lr_scheduler import LambdaLR


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def xywhr2xywhrsigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5)
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution with shape (N, 2)
        wh (torch.Tensor): size of original bboxes
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution with shape (N, 2, 2)
    """
    _shape = xywhr.size()
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=0.0001, max=10000.0)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = (0.5 * torch.diag_embed(wh)).square()
    sigma = R.bmm(S).bmm(R.permute(0, 2, 1)).reshape((_shape[0], 2, 2))
    return xy, wh, r, sigma


class KFLoss(nn.Module):
    """Kalman filter based loss.
    ref: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'
        alpha (int, optional): coefficient to control the magnitude of kfiou.
            Defaults to 3.0
    Returns:
        loss (torch.Tensor)
    """

    def __init__(self, fun='exp', alpha=3.0):
        super(KFLoss, self).__init__()
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.alpha = alpha

    def forward(self, pred, target):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted convexes
            target (torch.Tensor): Corresponding gt convexes
        Returns:
            loss (torch.Tensor)
            KFIoU (torch.Tensor)
        """
        xy_p, wh_p, r_p, Sigma_p = xywhr2xywhrsigma(pred)
        xy_t, wh_t, r_t, Sigma_t = xywhr2xywhrsigma(target)
        diff = (xy_p - xy_t).unsqueeze(-1)
        xy_loss = torch.log(diff.permute(0, 2, 1).bmm(Sigma_t.inverse()).bmm(diff) + 1).sum(dim=-1)
        wp2, hp2 = wh_p[:, 0] ** 2, wh_p[:, 1] ** 2
        wt2, ht2 = wh_t[:, 0] ** 2, wh_t[:, 1] ** 2
        cos2dr, sin2dr = torch.cos(r_p - r_t) ** 2, torch.sin(r_p - r_t) ** 2
        A = torch.sqrt(1 + wp2 * hp2 / (wt2 * ht2) + (wp2 / wt2 + hp2 / ht2) * cos2dr + (wp2 / ht2 + hp2 / wt2) * sin2dr)
        B = torch.sqrt(1 + wt2 * ht2 / (wp2 * hp2) + (wt2 / wp2 + ht2 / hp2) * cos2dr + (wt2 / hp2 + ht2 / wp2) * sin2dr)
        KFIoU = (4 - self.alpha) / (A + B - self.alpha)
        if self.fun == 'ln':
            kf_loss = -torch.log(KFIoU + 1e-06)
        elif self.fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU
        loss = (xy_loss + kf_loss).clamp(0)
        return loss.mean(), KFIoU


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(nn.Mish())
        elif activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'swish':
            self.conv.append(nn.SiLU())
        elif activation == 'linear':
            pass
        else:
            raise NotImplementedError('Acativation function not found.')

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, e=0.5, act=None):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act)
        self.cv2 = Conv(c_, c2, 3, 1, act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super(CSP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1, 'mish')
        self.cv2 = Conv(c1, c_, 1, 1, 'mish')
        self.cv3 = Conv(c_, c_, 1, 1, 'mish')
        self.cv4 = Conv(2 * c_, c2, 1, 1, 'mish')
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0, act='mish') for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class SPP(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, 'leaky')
        self.cv2 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv3 = Conv(c1, c_, 1, 1, 'leaky')
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.cv4 = Conv(c_ * 4, c_, 1, 1, 'leaky')
        self.cv5 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv6 = Conv(c1, c2, 1, 1, 'leaky')

    def forward(self, x):
        x = self.cv3(self.cv2(self.cv1(x)))
        x = torch.cat([self.m3(x), self.m2(x), self.m1(x), x], dim=1)
        x = self.cv6(self.cv5(self.cv4(x)))
        return x


class Backbonev4(nn.Module):

    def __init__(self):
        super().__init__()
        self.cbm0 = Conv(3, 32, 3, 1, 'mish')
        self.cbm1 = Conv(32, 64, 3, 2, 'mish')
        self.csp1 = CSP(64, 64, 1)
        self.cbm2 = Conv(64, 128, 3, 2, 'mish')
        self.csp2 = CSP(128, 128, 2)
        self.cbm3 = Conv(128, 256, 3, 2, 'mish')
        self.csp3 = CSP(256, 256, 8)
        self.cbm4 = Conv(256, 512, 3, 2, 'mish')
        self.csp4 = CSP(512, 512, 8)
        self.cbm5 = Conv(512, 1024, 3, 2, 'mish')
        self.csp5 = CSP(1024, 1024, 4)
        self.spp = SPP(1024, 512)

    def forward(self, x):
        x = self.cbm0(x)
        x = self.csp1(self.cbm1(x))
        x = self.csp2(self.cbm2(x))
        d3 = self.csp3(self.cbm3(x))
        d4 = self.csp4(self.cbm4(d3))
        d5 = self.csp5(self.cbm5(d4))
        d5 = self.spp(d5)
        return d3, d4, d5


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1, 'swish')
        self.cv2 = Conv(c1, c_, 1, 1, 'swish')
        self.cv3 = Conv(2 * c_, c2, 1, 1, 'swish')
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0, act='swish') for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, 'swish')
        self.cv2 = Conv(c_ * 4, c2, 1, 1, 'swish')
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Backbonev5(nn.Module):

    def __init__(self):
        super().__init__()
        self.cbs0 = Conv(3, 64, 6, 2, 'swish')
        self.cbs1 = Conv(64, 128, 3, 2, 'swish')
        self.csp1 = C3(128, 128, 3)
        self.cbs2 = Conv(128, 256, 3, 2, 'swish')
        self.csp2 = C3(256, 256, 6)
        self.cbs3 = Conv(256, 512, 3, 2, 'swish')
        self.csp3 = C3(512, 512, 9)
        self.cbs4 = Conv(512, 1024, 3, 2, 'swish')
        self.csp4 = C3(1024, 1024, 3)
        self.spp = SPPF(1024, 1024)

    def forward(self, x):
        x = self.cbs0(x)
        x = self.csp1(self.cbs1(x))
        d3 = self.csp2(self.cbs2(x))
        d4 = self.csp3(self.cbs3(d3))
        d5 = self.csp4(self.cbs4(d4))
        d5 = self.spp(d5)
        return d3, d4, d5


class ELAN1(nn.Module):

    def __init__(self, c1, c2, e1=0.5, e2=0.5):
        super().__init__()
        h1 = int(c1 * e1)
        h2 = int(c1 * e2)
        self.cv1 = Conv(c1, h1, 1, 1, 'swish')
        self.cv2 = Conv(c1, h1, 1, 1, 'swish')
        self.cv3 = Conv(h1, h2, 3, 1, 'swish')
        self.cv4 = Conv(h1, h2, 3, 1, 'swish')
        self.cv5 = Conv(h2, h2, 3, 1, 'swish')
        self.cv6 = Conv(h2, h2, 3, 1, 'swish')
        self.cv7 = Conv((h1 + h2) * 2, c2, 1, 1, 'swish')

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv4(self.cv3(x2))
        x4 = self.cv6(self.cv5(x3))
        return self.cv7(torch.cat((x1, x2, x3, x4), dim=1))


class MaxConv(nn.Module):

    def __init__(self, c1, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(c1, c_, 1, 1, 'swish')
        self.cv2 = Conv(c1, c_, 1, 1, 'swish')
        self.cv3 = Conv(c_, c_, 3, 2, 'swish')

    def forward(self, x):
        x1 = self.cv1(self.m(x))
        x2 = self.cv3(self.cv2(x))
        return torch.cat((x1, x2), dim=1)


class SPPCSPC(nn.Module):

    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, 'swish')
        self.cv2 = Conv(c1, c_, 1, 1, 'swish')
        self.cv3 = Conv(c_, c_, 3, 1, 'swish')
        self.cv4 = Conv(c_, c_, 1, 1, 'swish')
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1, 'swish')
        self.cv6 = Conv(c_, c_, 3, 1, 'swish')
        self.cv7 = Conv(2 * c_, c2, 1, 1, 'swish')

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class Backbonev7(nn.Module):

    def __init__(self):
        super().__init__()
        self.cbs0 = Conv(3, 32, 3, 1, 'swish')
        self.cbs1 = Conv(32, 64, 3, 2, 'swish')
        self.cbs2 = Conv(64, 64, 3, 1, 'swish')
        self.cbs3 = Conv(64, 128, 3, 2, 'swish')
        self.elan1 = ELAN1(128, 256)
        self.mc1 = MaxConv(256)
        self.elan2 = ELAN1(256, 512)
        self.mc2 = MaxConv(512)
        self.elan3 = ELAN1(512, 1024)
        self.mc3 = MaxConv(1024)
        self.elan4 = ELAN1(1024, 1024, e1=0.25, e2=0.25)
        self.spp = SPPCSPC(1024, 512)

    def forward(self, x):
        x = self.cbs2(self.cbs1(self.cbs0(x)))
        x = self.elan1(self.cbs3(x))
        d3 = self.elan2(self.mc1(x))
        d4 = self.elan3(self.mc2(d3))
        d5 = self.elan4(self.mc3(d4))
        d5 = self.spp(d5)
        return d3, d4, d5


class C5(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1, 'leaky')
        self.cv2 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv3 = Conv(c1, c_, 1, 1, 'leaky')
        self.cv4 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv5 = Conv(c1, c2, 1, 1, 'leaky')

    def forward(self, x):
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))


class Neckv4(nn.Module):

    def __init__(self, output_ch):
        super().__init__()
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        self.conv9 = C5(512, 256)
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv15 = Conv(256, 128, 1, 1, 'leaky')
        self.conv16 = C5(256, 128)
        self.conv21 = Conv(128, 256, 3, 1, 'leaky')
        self.conv22 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv23 = Conv(128, 256, 3, 2, 'leaky')
        self.conv24 = C5(512, 256)
        self.conv29 = Conv(256, 512, 3, 1, 'leaky')
        self.conv30 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv31 = Conv(256, 512, 3, 2, 'leaky')
        self.conv32 = C5(1024, 512)
        self.conv37 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv38 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

    def forward(self, x1, x2, x3):
        up1 = self.up1(self.conv7(x1))
        x2 = self.conv8(x2)
        x2 = torch.cat([x2, up1], dim=1)
        x2 = self.conv9(x2)
        up2 = self.up2(self.conv14(x2))
        x3 = self.conv15(x3)
        x3 = torch.cat([x3, up2], dim=1)
        x3 = self.conv16(x3)
        x6 = self.conv22(self.conv21(x3))
        x3 = self.conv23(x3)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.conv24(x2)
        x5 = self.conv30(self.conv29(x2))
        x2 = self.conv31(x2)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv32(x1)
        x4 = self.conv38(self.conv37(x1))
        return x6, x5, x4


class Neckv5(nn.Module):

    def __init__(self, output_ch):
        super().__init__()
        self.conv7 = Conv(1024, 512, 1, 1, 'swish')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp1 = C3(1024, 512, 3, shortcut=False)
        self.conv14 = Conv(512, 256, 1, 1, 'swish')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp2 = C3(512, 256, 3, shortcut=False)
        self.conv15 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv16 = Conv(256, 256, 3, 2, 'swish')
        self.csp3 = C3(512, 512, 3, shortcut=False)
        self.conv17 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv18 = Conv(512, 512, 3, 2, 'swish')
        self.csp4 = C3(1024, 1024, 3, shortcut=False)
        self.conv19 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

    def forward(self, x1, x2, x3):
        x1 = self.conv7(x1)
        up1 = self.up1(x1)
        x2 = torch.cat([x2, up1], dim=1)
        x2 = self.csp1(x2)
        x2 = self.conv14(x2)
        up2 = self.up2(x2)
        x3 = torch.cat([x3, up2], dim=1)
        x3 = self.csp2(x3)
        x6 = self.conv15(x3)
        x3 = self.conv16(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.csp3(x2)
        x5 = self.conv17(x2)
        x2 = self.conv18(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.csp4(x1)
        x4 = self.conv19(x1)
        return x6, x5, x4


class ELAN2(nn.Module):

    def __init__(self, c1, c2, e1=0.5, e2=0.25):
        super().__init__()
        h1 = int(c1 * e1)
        h2 = int(c1 * e2)
        self.cv1 = Conv(c1, h1, 1, 1, 'swish')
        self.cv2 = Conv(c1, h1, 1, 1, 'swish')
        self.cv3 = Conv(h1, h2, 3, 1, 'swish')
        self.cv4 = Conv(h2, h2, 3, 1, 'swish')
        self.cv5 = Conv(h2, h2, 3, 1, 'swish')
        self.cv6 = Conv(h2, h2, 3, 1, 'swish')
        self.cv7 = Conv(h1 * 2 + h2 * 4, c2, 1, 1, 'swish')

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
        return self.cv7(torch.cat((x1, x2, x3, x4, x5, x6), dim=1))


class ImplicitA(nn.Module):

    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):

    def __init__(self, channel, mean=1.0, std=0.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class RepConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=1):
        super(RepConv, self).__init__()
        self.silu = nn.SiLU()
        self.rbr_identity = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
        self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, p, bias=False), nn.BatchNorm2d(num_features=c2))
        self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, 0, bias=False), nn.BatchNorm2d(num_features=c2))

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.silu(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


class Neckv7(nn.Module):

    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv(512, 256, 1, 1, 'swish')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.elan1 = ELAN2(512, 256)
        self.conv2 = Conv(256, 128, 1, 1, 'swish')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.elan2 = ELAN2(256, 128)
        self.conv3 = Conv(1024, 256, 1, 1, 'swish')
        self.conv4 = Conv(512, 128, 1, 1, 'swish')
        self.mc1 = MaxConv(128, e=1.0)
        self.elan3 = ELAN2(512, 256)
        self.mc2 = MaxConv(256, e=1.0)
        self.elan4 = ELAN2(1024, 512)
        self.repVgg1 = RepConv(128, 256)
        self.ia1 = ImplicitA(256)
        self.conv5 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.im1 = ImplicitM(output_ch)
        self.repVgg2 = RepConv(256, 512)
        self.ia2 = ImplicitA(512)
        self.conv6 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.im2 = ImplicitM(output_ch)
        self.repVgg3 = RepConv(512, 1024)
        self.ia3 = ImplicitA(1024)
        self.conv7 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.im3 = ImplicitM(output_ch)

    def forward(self, x1, x2, x3):
        x4 = self.up1(self.conv1(x1))
        x2 = self.conv3(x2)
        x2 = torch.cat([x2, x4], dim=1)
        x2 = self.elan1(x2)
        x5 = self.up2(self.conv2(x2))
        x3 = self.conv4(x3)
        x3 = torch.cat([x3, x5], dim=1)
        x3 = self.elan2(x3)
        x6 = self.im1(self.conv5(self.ia1(self.repVgg1(x3))))
        x3 = self.mc1(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.elan3(x2)
        x5 = self.im2(self.conv6(self.ia2(self.repVgg2(x2))))
        x2 = self.mc2(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.elan4(x1)
        x4 = self.im3(self.conv7(self.ia3(self.repVgg3(x1))))
        return x6, x5, x4


class YoloCSLLayer(nn.Module):

    def __init__(self, num_classes, anchors, stride):
        super(YoloCSLLayer, self).__init__()
        self.nc = num_classes
        self.anchors = anchors
        self.stride = stride

    def forward(self, out, training):
        device = out[0].device
        infer_out = []
        for i in range(3):
            bs, gs = out[i].size(0), out[i].size(2)
            na = len(self.anchors[i])
            out[i] = out[i].view(bs, na, self.nc + 185, gs, gs).permute(0, 1, 3, 4, 2).contiguous()
            if not training:
                grid_x = torch.arange(gs, device=device).repeat(gs, 1).view([1, 1, gs, gs, 1])
                grid_y = torch.arange(gs, device=device).repeat(gs, 1).t().view([1, 1, gs, gs, 1])
                grid_xy = torch.cat((grid_x, grid_y), -1)
                anchors = torch.tensor(self.anchors[i], device=device)
                anchor_wh = anchors[:, :2].view([1, na, 1, 1, 2])
                y = out[i].sigmoid()
                pxy = (y[..., 0:2] * 2 - 0.5 + grid_xy) * self.stride[i]
                pwh = (y[..., 2:4] * 2) ** 2 * anchor_wh * self.stride[i]
                pconf = y[..., 4:5]
                pcls = y[..., 5:5 + self.nc]
                pa = y[..., 5 + self.nc:]
                _, ptheta = torch.max(pa, 4, keepdim=True)
                ptheta = (ptheta - 90) / 180 * np.pi
                y = torch.cat((pxy, pwh, ptheta, pconf, pcls), -1)
                y = y.view(bs, -1, self.nc + 6)
                infer_out.append(y)
        return out if training else (out, torch.cat(infer_out, 1))


class YoloKFIoULayer(nn.Module):

    def __init__(self, num_classes, anchors, stride):
        super(YoloKFIoULayer, self).__init__()
        self.nc = num_classes
        self.anchors = anchors
        self.stride = stride

    def forward(self, out, training):
        device = out[0].device
        infer_out = []
        for i in range(3):
            bs, gs = out[i].size(0), out[i].size(2)
            na = len(self.anchors[i])
            out[i] = out[i].view(bs, na, self.nc + 6, gs, gs).permute(0, 1, 3, 4, 2).contiguous()
            if not training:
                grid_x = torch.arange(gs, device=device).repeat(gs, 1).view([1, 1, gs, gs, 1])
                grid_y = torch.arange(gs, device=device).repeat(gs, 1).t().view([1, 1, gs, gs, 1])
                grid_xy = torch.cat((grid_x, grid_y), -1)
                anchors = torch.tensor(self.anchors[i], device=device)
                anchor_wh = anchors[:, :2].view([1, na, 1, 1, 2])
                anchor_a = anchors[:, 2].view([1, na, 1, 1, 1])
                y = out[i].sigmoid()
                pxy = (y[..., 0:2] * 2 - 0.5 + grid_xy) * self.stride[i]
                pwh = (y[..., 2:4] * 2) ** 2 * anchor_wh * self.stride[i]
                pa = (y[..., 4:5] - 0.5) * 0.5236 + anchor_a
                pconf = y[..., 5:6]
                pcls = y[..., 6:]
                y = torch.cat((pxy, pwh, pa, pconf, pcls), -1)
                y = y.view(bs, -1, self.nc + 6)
                infer_out.append(y)
        return out if training else (out, torch.cat(infer_out, 1))


class Yolo(nn.Module):

    def __init__(self, n_classes, model_config, mode, ver):
        super().__init__()
        anchors = model_config['anchors']
        angles = [(a * np.pi / 180) for a in model_config['angles']]
        strides = [8, 16, 32]
        if mode == 'csl':
            output_ch = (4 + 180 + 1 + n_classes) * 3
            an = self._make_anchors(strides, anchors)
            YoloLayer = YoloCSLLayer(n_classes, an, strides)
        elif mode == 'kfiou':
            output_ch = (5 + 1 + n_classes) * 3 * 6
            an = self._make_rotated_anchors(strides, anchors, angles)
            YoloLayer = YoloKFIoULayer(n_classes, an, strides)
        else:
            raise NotImplementedError('Loss mode : {} not found.'.format(mode))
        self.anchors = an
        self.nc = n_classes
        yolo = {'yolov4': [Backbonev4, Neckv4], 'yolov5': [Backbonev5, Neckv5], 'yolov7': [Backbonev7, Neckv7]}
        self.backbone = yolo[ver][0]()
        self.neck = yolo[ver][1](output_ch)
        self.yolo = YoloLayer

    def forward(self, i, training):
        d3, d4, d5 = self.backbone(i)
        x2, x10, x18 = self.neck(d5, d4, d3)
        out = self.yolo([x2, x10, x18], training)
        return out

    @staticmethod
    def _make_anchors(strides, anchors):
        an = []
        for stride, anchor in zip(strides, anchors):
            tmp = []
            for i in range(0, len(anchor), 2):
                tmp.append([anchor[i] / stride, anchor[i + 1] / stride])
            an.append(tmp)
        return an

    @staticmethod
    def _make_rotated_anchors(strides, anchors, angles):
        an = []
        for stride, anchor in zip(strides, anchors):
            tmp = []
            for i in range(0, len(anchor), 2):
                for angle in angles:
                    tmp.append([anchor[i] / stride, anchor[i + 1] / stride, angle])
            an.append(tmp)
        return an


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Backbonev4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Backbonev5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Backbonev7,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (C3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C5,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ELAN1,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ELAN2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ImplicitA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImplicitM,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxConv,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPCSPC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

