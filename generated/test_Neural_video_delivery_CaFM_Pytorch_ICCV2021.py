
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


from torch.utils.data import dataloader


from torch.utils.data import ConcatDataset


import numpy as np


import torch


import torch.utils.data as data


import random


import torch.multiprocessing as multiprocessing


from torch.utils.data import DataLoader


from torch.utils.data import SequentialSampler


from torch.utils.data import RandomSampler


from torch.utils.data import BatchSampler


from torch.utils.data import _utils


from torch.utils.data._utils import collate


from torch.utils.data._utils import signal_handling


from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


from torch.utils.data._utils import ExceptionWrapper


from torch.utils.data._utils import IS_WINDOWS


from torch.utils.data._utils.worker import ManagerWatchdog


import matplotlib


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from types import SimpleNamespace


import torch.optim as optim


import torchvision.models as models


import torch.backends.cudnn as cudnn


import torch.nn.parallel as P


import torch.utils.model_zoo


import math


from torch import nn


import torch.nn.init as init


import torch.nn.utils as utils


import time


import torch.optim.lr_scheduler as lrs


class Adversarial(nn.Module):

    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.dis = discriminator.Discriminator(args)
        if gan_type == 'WGAN_GP':
            optim_dict = {'optimizer': 'ADAM', 'betas': (0, 0.9), 'epsilon': 1e-08, 'lr': 1e-05, 'weight_decay': args.weight_decay, 'decay': args.decay, 'gamma': args.gamma}
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args
        self.optimizer = utility.make_optimizer(optim_args, self.dis)

    def forward(self, fake, real):
        self.loss = 0
        fake_detach = fake.detach()
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(), inputs=hat, retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            elif self.gan_type == 'RGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()
            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)
        self.loss /= self.gan_k
        d_fake_bp = self.dis(fake)
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss


class Discriminator(nn.Module):
    """
        output is not normalized
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(nn.Conv2d(_in_channels, _out_channels, 3, padding=1, stride=stride, bias=False), nn.BatchNorm2d(_out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
        patch_size = args.patch_size // 2 ** ((depth + 1) // 2)
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(1024, 1)]
        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output


class VGG(nn.Module):

    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
        loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss


class Model(nn.Module):

    def __init__(self, args, ckp):
        super(Model, self).__init__()
        None
        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = args.model == 'VDSR'
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        if args.precision == 'half':
            self.model.half()
        self.load(ckp.get_path('model'), pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)
        None
    """
    def forward(self, x, idx_scale, seg_flag):
    """

    def forward(self, x, idx_scale, num):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)
        if self.training:
            if self.n_GPUs > 1:
                """
                TODO:data_paraller 怎么传递多个参数   
                """
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                """
                return self.model(x, seg_flag)
                """
                return self.model(x, num)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            if self.self_ensemble:
                return self.forward_x8(num, x, forward_function=forward_function)
            else:
                return forward_function(x, num)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def save_every(self, apath, epoch, is_best=False):
        save_dirs = []
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if resume == -1:
            load_from = torch.load(os.path.join(apath, 'model_latest.pt'), **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                None
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(self.model.url, model_dir=dir_model, **kwargs)
            elif pre_train:
                None
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(os.path.join(apath, 'model_{}.pt'.format(resume)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        h, w = args[0].size()[-2:]
        top = slice(0, h // 2 + shave)
        bottom = slice(h - h // 2 - shave, h)
        left = slice(0, w // 2 + shave)
        right = slice(w - w // 2 - shave, w)
        x_chops = [torch.cat([a[..., top, left], a[..., top, right], a[..., bottom, left], a[..., bottom, right]]) for a in args]
        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:i + n_GPUs] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list):
                    y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list):
                    y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.append(_y)
        h *= scale
        w *= scale
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        bottom_r = slice(h // 2 - h, None)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)
        right_r = slice(w // 2 - w, None)
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
        if len(y) == 1:
            y = y[0]
        return y

    def forward_x8(self, num, *args, forward_function=None):

        def _transform(v, op):
            if self.precision != 'single':
                v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            ret = torch.Tensor(tfnp)
            if self.precision == 'half':
                ret = ret.half()
            return ret
        list_x = []
        for a in args:
            x = [a]
            for tf in ('v', 'h', 't'):
                x.extend([_transform(_x, tf) for _x in x])
            list_x.append(x)
        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x, num)
            if not isinstance(y, list):
                y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.append(_y)
        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if i % 4 % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')
        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1:
            y = y[0]
        return y


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


def set_padding_size(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ContentAwareFM(nn.Module):

    def __init__(self, in_channel, kernel_size):
        super(ContentAwareFM, self).__init__()
        padding = set_padding_size(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, groups=in_channel // 2)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return self.transformer(x) * self.gamma + x


class BasicBlock_(nn.Module):

    def __init__(self, args, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        self.use_cafm = args.use_cafm
        super(BasicBlock_, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size, bias=bias)
        if args.cafm:
            if self.use_cafm:
                self.cafms = nn.ModuleList([ContentAwareFM(out_channels, 1) for _ in range(args.segnum)])
        self.act = act

    def forward(self, input, num):
        x = self.conv1(input)
        if self.use_cafm:
            x = self.cafms[num](x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, args, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.args = args
        if self.args.cafm:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
            if self.args.use_cafm:
                if self.args.cafm:
                    self.cafms1 = nn.ModuleList([ContentAwareFM(n_feats, 1) for _ in range(args.segnum)])
                    self.cafms2 = nn.ModuleList([ContentAwareFM(n_feats, 1) for _ in range(args.segnum)])
            self.act = act
        else:
            m = []
            for i in range(2):
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
                if bn:
                    m.append(ContentAwareFM(n_feats, 7))
                if i == 0:
                    m.append(act)
            self.body = nn.Sequential(*m)

    def forward(self, input, num):
        x = self.conv1(input)
        if self.args.cafm:
            if self.args.use_cafm:
                x = self.cafms1[num](x)
            x = self.conv2(self.act(x))
            if self.args.use_cafm:
                x = self.cafms2[num](x)
        else:
            x = self.conv2(self.act(x))
        res = x.mul(self.res_scale)
        res += input
        return res


class ResBlock_rcan(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, args, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_rcan, self).__init__()
        self.res_scale = res_scale
        self.args = args
        if self.args.cafm:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
            if self.args.use_cafm:
                if self.args.cafm:
                    self.cafms1 = nn.ModuleList([ContentAwareFM(n_feats, 1) for _ in range(args.segnum)])
                    self.cafms2 = nn.ModuleList([ContentAwareFM(n_feats, 1) for _ in range(args.segnum)])
            self.act = act
        else:
            m = []
            for i in range(2):
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
                if bn:
                    m.append(ContentAwareFM(n_feats, 7))
                if i == 0:
                    m.append(act)
            self.body = nn.Sequential(*m)

    def forward(self, input, num):
        x = self.conv1(input)
        if self.args.cafm:
            if self.args.use_cafm:
                x = self.cafms1[num](x)
            x = self.conv2(self.act(x))
            if self.args.use_cafm:
                x = self.cafms2[num](x)
        else:
            x = self.conv2(self.act(x))
        res = x.mul(self.res_scale)
        return res


class ResBlock_org(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, args, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_org, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, kernel_size, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, kernel_size, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, kernel_size, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {(2): (6, 2, 2), (4): (8, 4, 2), (8): (12, 8, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d
    return conv_f(in_channels, out_channels, kernel_size, stride=stride, padding=padding)


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1), nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels, scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)
        out = a_0.add(a_1)
        return out


class DDBPN(nn.Module):

    def __init__(self, args):
        super(DDBPN, self).__init__()
        scale = args.scale[0]
        n0 = 128
        nr = 32
        self.depth = 6
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0), nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True, i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3, padding=1)]
        self.reconstruction = nn.Sequential(*reconstruction)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.initial(x)
        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        out = self.add_mean(out)
        return out


url = {'r16f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr_baseline-a00cab12.pt', 'r80f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr-4a78bedf.pt'}


class ESPCN(nn.Module):

    def __init__(self, args):
        super(ESPCN, self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])
        self.n_colors = args.n_colors
        self.cafm = args.cafm
        self.use_cafm = args.use_cafm
        self.segnum = args.segnum
        self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * self.scale ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self._initialize_weights()
        if self.cafm:
            if self.use_cafm:
                self.cafms1 = nn.ModuleList([ContentAwareFM(64, 1) for _ in range(self.segnum)])
                self.cafms2 = nn.ModuleList([ContentAwareFM(64, 1) for _ in range(self.segnum)])
                self.cafms3 = nn.ModuleList([ContentAwareFM(32, 1) for _ in range(self.segnum)])

    def forward(self, x, num):
        if self.cafm:
            out = self.act_func(self.conv1(x))
            if self.use_cafm:
                out = self.cafms1[num](out)
            out = self.act_func(self.conv2(out))
            if self.use_cafm:
                out = self.cafms2[num](out)
            out = self.act_func(self.conv3(out))
            if self.use_cafm:
                out = self.cafms3[num](out)
            out = self.pixel_shuffle(self.conv4(out))
            return out
        else:
            out = self.act_func(self.conv1(x))
            out = self.act_func(self.conv2(out))
            out = self.act_func(self.conv3(out))
            out = self.pixel_shuffle(self.conv4(out))
            return out

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)


class DepthwiseConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = kernal_size // 2
        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, padding=padding, stride=1, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class GroupConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1, bias=False):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)
        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, bias=bias))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out


class MDConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_chunks, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)
        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(in_channels, out_channels, kernal_size=kernel_size, bias=bias))

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class AdaptiveFM(nn.Module):

    def __init__(self, in_channel, kernel_size):
        super(AdaptiveFM, self).__init__()
        padding = get_valid_padding(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, groups=in_channel)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return self.transformer(x) * self.gamma + x


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, args, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, segnum=1):
        super(RCAB, self).__init__()
        self.modules_body = common.ResBlock_rcan(conv, n_feat, kernel_size, args, bn=False, act=act, res_scale=args.res_scale)
        self.calayer = CALayer(n_feat, reduction)

    def forward(self, x, num):
        res = self.modules_body(x, num)
        res = self.calayer(res)
        res += x
        return res


class ResidualGroup(nn.Module):

    def __init__(self, args, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, segnum=1):
        super(ResidualGroup, self).__init__()
        self.n_resblocks = n_resblocks
        modules_body = [RCAB(args, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, segnum=segnum) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        if args.adafm:
            self.body = nn.ModuleList(modules_body)
        else:
            self.body = nn.Sequential(*modules_body)

    def forward(self, x, num):
        res = x
        for i in range(self.n_resblocks):
            res = self.body[i](res, num)
        res = self.body[-1](res)
        res += x
        return res


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


class SRCNN(nn.Module):

    def __init__(self, args):
        super(SRCNN, self).__init__()
        n_colors = args.n_colors
        self.scale = int(args.scale[0])
        self.cafm = args.cafm
        self.use_cafm = args.use_cafm
        self.segnum = args.segnum
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, n_colors, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        if self.cafm:
            if self.use_cafm:
                self.cafms1 = nn.ModuleList([ContentAwareFM(64, 1) for _ in range(self.segnum)])
                self.cafms2 = nn.ModuleList([ContentAwareFM(32, 1) for _ in range(self.segnum)])

    def forward(self, x, num):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        if self.use_cafm:
            x = self.cafms1[num](x)
        x = self.relu(self.conv2(x))
        if self.use_cafm:
            x = self.cafms2[num](x)
        x = self.conv3(x)
        return x


class SRCNNCond(nn.Module):

    def __init__(self, args):
        super(SRCNNCond, self).__init__()
        n_colors = args.n_colors
        n = 4
        self.conv1 = common.CondConv2d(in_channels=n_colors, out_channels=64, kernel_size=9, stride=1, padding=9 // 2, num=n)
        self.conv2 = common.CondConv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=5 // 2, num=n)
        self.conv3 = common.CondConv2d(in_channels=32, out_channels=n_colors, kernel_size=5, stride=1, padding=5 // 2, num=n)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x, krl=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        self.conv1.initialize_weights(init.calculate_gain('relu'))
        self.conv2.initialize_weights(init.calculate_gain('relu'))
        self.conv3.initialize_weights(1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveFM,
     lambda: ([], {'in_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {})),
    (BasicBlock,
     lambda: ([], {'conv': torch.nn.ReLU, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock_,
     lambda: ([], {'args': SimpleNamespace(use_cafm=4, cafm=4, segnum=4), 'conv': torch.nn.ReLU, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
    (ContentAwareFM,
     lambda: ([], {'in_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {})),
    (DepthwiseConv2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernal_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupConv2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MDConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'n_chunks': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MeanShift,
     lambda: ([], {'rgb_range': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (RDB,
     lambda: ([], {'growRate0': 4, 'growRate': 4, 'nConvLayers': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RDB_Conv,
     lambda: ([], {'inChannels': 4, 'growRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'conv': torch.nn.ReLU, 'n_feats': 4, 'kernel_size': 4, 'args': SimpleNamespace(cafm=4, use_cafm=4, segnum=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
    (ResBlock_org,
     lambda: ([], {'conv': torch.nn.ReLU, 'n_feats': 4, 'kernel_size': 4, 'args': SimpleNamespace()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock_rcan,
     lambda: ([], {'conv': torch.nn.ReLU, 'n_feats': 4, 'kernel_size': 4, 'args': SimpleNamespace(cafm=4, use_cafm=4, segnum=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
]

