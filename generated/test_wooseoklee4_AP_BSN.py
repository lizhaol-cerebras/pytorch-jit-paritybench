
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


import random


import numpy as np


from scipy.io import savemat


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import math


import time


from torch import nn


from torch import optim


import torch.autograd as autograd


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader


from torch.autograd import Variable


from math import exp


loss_class_dict = {}


class Loss(nn.Module):

    def __init__(self, loss_string, tmp_info=[]):
        super().__init__()
        loss_string = loss_string.replace(' ', '')
        self.loss_list = []
        for single_loss in loss_string.split('+'):
            weight, name = single_loss.split('*')
            ratio = True if 'r' in weight else False
            weight = float(weight.replace('r', ''))
            if name in loss_class_dict:
                self.loss_list.append({'name': name, 'weight': float(weight), 'func': loss_class_dict[name](), 'ratio': ratio})
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))
        self.tmp_info_list = []
        for name in tmp_info:
            if name in loss_class_dict:
                self.tmp_info_list.append({'name': name, 'func': loss_class_dict[name]()})
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))

    def forward(self, input_data, model_output, data, module, loss_name=None, change_name=None, ratio=1.0):
        """
        forward all loss and return as dict format.
        Args
            input_data   : input of the network (also in the data)
            model_output : output of the network
            data         : entire batch of data
            module       : dictionary of modules (for another network forward)
            loss_name    : (optional) choose specific loss with name
            change_name  : (optional) replace name of chosen loss
            ratio        : (optional) percentage of learning procedure for increase weight during training
        Return
            losses       : dictionary of loss
        """
        loss_arg = input_data, model_output, data, module
        if loss_name is not None:
            for single_loss in self.loss_list:
                if loss_name == single_loss['name']:
                    loss = single_loss['weight'] * single_loss['func'](*loss_arg)
                    if single_loss['ratio']:
                        loss *= ratio
                    if change_name is not None:
                        return {change_name: loss}
                    return {single_loss['name']: loss}
            raise RuntimeError('there is no such loss in training losses: {}'.format(loss_name))
        losses = {}
        for single_loss in self.loss_list:
            losses[single_loss['name']] = single_loss['weight'] * single_loss['func'](*loss_arg)
            if single_loss['ratio']:
                losses[single_loss['name']] *= ratio
        tmp_info = {}
        for single_tmp_info in self.tmp_info_list:
            with torch.no_grad():
                tmp_info[single_tmp_info['name']] = single_tmp_info['func'](*loss_arg)
        return losses, tmp_info


class CentralMaskedConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class DCl(nn.Module):

    def __init__(self, stride, in_ch):
        super().__init__()
        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class DC_branchl(nn.Module):

    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [DCl(stride, in_ch) for _ in range(num_module)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DBSNl(nn.Module):
    """
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    """

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        """
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        """
        super().__init__()
        assert base_ch % 2 == 0, 'base channel should be divided with 2'
        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        x = torch.cat([br1, br2], dim=1)
        return self.tail(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


def pixel_shuffle_down_sampling(x: 'torch.Tensor', f: 'int', pad: 'int'=0, pad_value: 'float'=0.0):
    """
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    """
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c, w + 2 * f * pad, h + 2 * f * pad)
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c, w + 2 * f * pad, h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: 'torch.Tensor', f: 'int', pad: 'int'=0):
    """
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    """
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


class APBSN(nn.Module):
    """
    Asymmetric PD Blind-Spot Network (AP-BSN)
    """

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        """
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        """
        super().__init__()
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)

    def forward(self, img, pd=None):
        """
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        """
        if pd is None:
            pd = self.pd_a
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]
        return img_pd_bsn

    def denoise(self, x):
        """
        Denoising process for inference.
        """
        b, c, h, w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)
        if not self.R3:
            """ Directly return the result (w/o R3) """
            return img_pd_bsn[:, :, :h, :w]
        else:
            denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p
                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
            return torch.mean(denoised, dim=-1)
        """
        elif self.R3 == 'PD-refinement':
            s = 2
            denoised = torch.empty(*(x.shape), s**2, device=x.device)
            for i in range(s):
                for j in range(s):
                    tmp_input = torch.clone(x_mean).detach()
                    tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
                    p = self.pd_pad
                    tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                    if self.pd_pad == 0:
                        denoised[..., i*s+j] = self.bsn(tmp_input)
                    else:
                        denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return_denoised = torch.mean(denoised, dim=-1)
        else:
            raise RuntimeError('post-processing type not supported')
        """


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CentralMaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DBSNl,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DCl,
     lambda: ([], {'stride': 1, 'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

