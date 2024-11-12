
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


import torch.nn as nn


import time


import numpy as np


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


import math


import random


import pandas as pd


import torchvision.transforms as transforms


import torchvision.models as models


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from math import cos


from math import pi


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch import nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, base_encoder, num_classes, norm_layer=None):
        super(ResNet, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer)
        assert not hasattr(self.backbone, 'fc'), 'fc should not in backbone'
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class ResNetCls(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, width_multiplier=1, normalize=False):
        super(ResNetCls, self).__init__(block, layers, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, width_multiplier=width_multiplier)
        self.fc = nn.Linear(self.out_channels, num_classes)
        self.normalize = normalize

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.normalize:
            x = nn.functional.normalize(x, dim=1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=2):
        super(MLP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Linear(in_channels, hid_channels, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Linear(hid_channels, out_channels, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='normal'):
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class Encoder(nn.Module):

    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2):
        super(Encoder, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer)
        in_dim = self.backbone.out_channels
        self.neck = MLP(in_dim, hid_dim, out_dim, norm_layer=norm_layer, num_mlp=num_mlp)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im):
        out = self.backbone(im)
        out = self.neck(out)
        return out


class BYOL(nn.Module):
    """
    BYOL re-implementation. Paper: https://arxiv.org/abs/2006.07733
    """

    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2):
        super(BYOL, self).__init__()
        self.base_m = m
        self.curr_m = m
        self.online_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.target_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.predictor = MLP(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        momentum = 1.0 - (1.0 - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data = param_tgt.data * momentum + param_ol.data * (1.0 - momentum)

    def loss_func(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, im_v1, im_v2=None):
        """
        Input:
            im_v1: a batch of view1 images
            im_v2: a batch of view2 images
        Output:
            loss
        """
        if im_v2 is None:
            feats = self.online_net.backbone(im_v1)
            return feats
        proj_online_v1 = self.online_net(im_v1)
        proj_online_v2 = self.online_net(im_v2)
        with torch.no_grad():
            proj_target_v1 = self.target_net(im_v1).clone().detach()
            proj_target_v2 = self.target_net(im_v2).clone().detach()
        loss = self.loss_func(self.predictor(proj_online_v1), proj_target_v2) + self.loss_func(self.predictor(proj_online_v2), proj_target_v1)
        return loss


class BYOLEMAN(BYOL):

    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2):
        super(BYOL, self).__init__()
        self.base_m = m
        self.curr_m = m
        self.online_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.target_net = Encoder(base_encoder, hid_dim, dim, num_mlp=num_neck_mlp)
        self.predictor = MLP(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        momentum = 1.0 - (1.0 - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        state_dict_ol = self.online_net.state_dict()
        state_dict_tgt = self.target_net.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, 'state_dict names are different!'
            assert v_ol.shape == v_tgt.shape, 'state_dict shapes are different!'
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1.0 - momentum) * v_ol)


class FixMatch(nn.Module):

    def __init__(self, base_encoder, num_classes=1000, eman=False, momentum=0.999, norm=None):
        super(FixMatch, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.main = ResNet(base_encoder, num_classes, norm_layer=norm)
        if eman:
            None
            self.ema = ResNet(base_encoder, num_classes, norm_layer=norm)
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)
                param_ema.requires_grad = False
        else:
            self.ema = None

    def momentum_update_ema(self):
        state_dict_main = self.main.state_dict()
        state_dict_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, 'state_dict names are different!'
            assert v_main.shape == v_ema.shape, 'state_dict shapes are different!'
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * self.momentum + (1.0 - self.momentum) * v_main)

    def forward(self, im_x, im_u_w=None, im_u_s=None):
        if im_u_w is None and im_u_s is None:
            logits = self.main(im_x)
            return logits
        batch_size_x = im_x.shape[0]
        if not self.eman:
            inputs = torch.cat((im_x, im_u_w, im_u_s))
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)
        else:
            inputs = torch.cat((im_x, im_u_s))
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_s = logits[batch_size_x:]
            with torch.no_grad():
                logits_u_w = self.ema(im_u_w)
        return logits_x, logits_u_w, logits_u_s


def build_hidden_head(num_mlp, dim_mlp, dim):
    modules = []
    for _ in range(1, num_mlp):
        modules.append(nn.Linear(dim_mlp, dim_mlp))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(dim_mlp, dim))
    return nn.Sequential(*modules)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, num_mlp=2, norm_layer=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        assert num_mlp >= 1
        if norm_layer is nn.BatchNorm2d or norm_layer is None:
            self.do_shuffle_bn = True
        else:
            self.do_shuffle_bn = False
        self.encoder_q = base_encoder(norm_layer=norm_layer)
        self.encoder_k = base_encoder(norm_layer=norm_layer)
        dim_mlp = self.encoder_q.out_channels
        self.encoder_q.fc = build_hidden_head(num_mlp, dim_mlp, dim)
        self.encoder_k.fc = build_hidden_head(num_mlp, dim_mlp, dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = dist_utils.concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if im_k is None:
            feats = self.encoder_q(im_q)
            return feats
        q = self.encoder_q(im_q)
        q = self.encoder_q.fc(q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            if self.do_shuffle_bn:
                im_k, idx_unshuffle = dist_utils.batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1)
            if self.do_shuffle_bn:
                k = dist_utils.batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


class MoCoEMAN(MoCo):

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, num_mlp=2, norm_layer=None):
        super(MoCoEMAN, self).__init__(base_encoder, dim, K, m, T, num_mlp, norm_layer)
        self.do_shuffle_bn = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder's state_dict. In MoCo, it is parameters
        """
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for (k_q, v_q), (k_k, v_k) in zip(state_dict_q.items(), state_dict_k.items()):
            assert k_k == k_q, 'state_dict names are different!'
            if 'num_batches_tracked' in k_k:
                v_k.copy_(v_q)
            else:
                v_k.copy_(v_k * self.m + (1.0 - self.m) * v_q)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'in_channels': 4, 'hid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

