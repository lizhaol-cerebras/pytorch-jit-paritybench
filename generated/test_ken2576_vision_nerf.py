
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


from typing import Optional


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from torch.utils.data import DistributedSampler


from torch.utils.data import WeightedRandomSampler


import math


import torchvision.transforms as transforms


from scipy.spatial.transform import Rotation as R


import torch.nn.functional as F


import torchvision.transforms as T


import torch.nn as nn


from collections import OrderedDict


import types


import time


import random


import torch.utils.data.distributed


import torch.distributed as dist


from torch.utils.data import DataLoader


from torchvision.utils import make_grid


from matplotlib.backends.backend_agg import FigureCanvasAgg


from matplotlib.figure import Figure


import matplotlib as mpl


from matplotlib import cm


TINY_NUMBER = 1e-06


def img2mse(x, y, mask=None):
    """
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    """
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


class Criterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']
        loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        return loss, scalars_to_log


class GaussianActivation(nn.Module):

    def __init__(self, a=1.0):
        super(GaussianActivation, self).__init__()
        self.a = a

    def forward(self, x):
        return torch.exp(-0.5 * x ** 2 / self.a ** 2)


class ResnetBlock(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_gaussian=False):
        super().__init__()
        if use_gaussian:
            self.prelu_0 = GaussianActivation()
            self.prelu_1 = GaussianActivation()
        else:
            self.prelu_0 = torch.nn.ReLU(inplace=True)
            self.prelu_1 = torch.nn.ReLU(inplace=True)
        self.fc_0 = torch.nn.Linear(input_size, hidden_size)
        self.fc_1 = torch.nn.Linear(hidden_size, output_size)
        self.shortcut = torch.nn.Linear(input_size, output_size, bias=False) if input_size != output_size else None

    def forward(self, x):
        residual = self.fc_1(self.prelu_1(self.fc_0(self.prelu_0(x))))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return residual + shortcut


class PosEncodeResnet(torch.nn.Module):

    def __init__(self, args, pos_size, x_size, hidden_size, output_size, block_num, freq_factor=np.pi, use_gaussian=False):
        """
        Args:
            pos_size: size of positional encodings
            x_size: size of input vector
            hidden_size: hidden channels
            output_size: output channels
            freq_num: how many frequency bases
            block_num: how many resnet blocks
        """
        super().__init__()
        self.args = args
        self.freq_factor = freq_factor
        input_size = pos_size * (2 * self.args.freq_num + 1) + x_size
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.blocks = torch.nn.ModuleList([ResnetBlock(hidden_size, hidden_size, hidden_size, use_gaussian=use_gaussian) for i in range(block_num)])
        if use_gaussian:
            self.output_prelu = GaussianActivation()
        else:
            self.output_prelu = torch.nn.ReLU(inplace=True)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def posenc(self, x):
        freq_multiplier = (self.freq_factor * 2 ** torch.arange(self.args.freq_num, device=x.device)).view(1, 1, 1, -1)
        x_expand = x.unsqueeze(-1)
        sin_val = torch.sin(x_expand * freq_multiplier)
        cos_val = torch.cos(x_expand * freq_multiplier)
        return torch.cat([x_expand, sin_val, cos_val], -1).view(x.shape[:2] + (-1,))

    def forward(self, pos_x, in_x):
        """
        Args:
            pos_x: input to be encoded with positional encodings
            in_x: input NOT to be encoded with positional encodings
        """
        x = self.posenc(pos_x)
        x = torch.cat([x, in_x], axis=-1)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_layer(self.output_prelu(x))
        out = torch.cat([self.sigmoid(out[..., :-1]), self.softplus(out[..., -1:])], -1)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
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


class Slice(nn.Module):

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = posemb[:, :self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


activations = {}


def forward_flex(self, x):
    b, c, h, w = x.shape
    pos_embed = self._resize_pos_embed(self.pos_embed, h // self.patch_size[1], w // self.patch_size[0])
    B = x.shape[0]
    if hasattr(self.patch_embed, 'backbone'):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, 'dist_token', None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x


def get_activation(name):

    def hook(model, input, output):
        activations[name] = output
    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == 'project':
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper


def _make_vit_b16_backbone(model, features=[96, 192, 384, 768], size=[384, 384], hooks=[2, 5, 8, 11], vit_features=768, use_readout='ignore', start_index=1):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))
    pretrained.activations = activations
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    pretrained.act_postprocess1 = nn.Sequential(readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess2 = nn.Sequential(readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess3 = nn.Sequential(readout_oper[2], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0))
    pretrained.act_postprocess4 = nn.Sequential(readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained


def _make_pretrained_vitb16_128(pretrained, use_readout='ignore', hooks=None):
    model = timm.create_model('vit_base_patch16_224', img_size=128, pretrained=pretrained)
    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, hooks=None, use_readout='ignore'):
    if backbone == 'vitb16_128':
        pretrained = _make_pretrained_vitb16_128(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)
    else:
        None
        assert False
    return pretrained, scratch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    glob = pretrained.model.forward_flex(x)
    layer_1 = pretrained.activations['1']
    layer_2 = pretrained.activations['2']
    layer_3 = pretrained.activations['3']
    layer_4 = pretrained.activations['4']
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)
    unflatten = nn.Sequential(nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]])))
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    layer_1 = pretrained.act_postprocess1[3:len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3:len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3:len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3:len(pretrained.act_postprocess4)](layer_4)
    return layer_1, layer_2, layer_3, layer_4


class VIT(nn.Module):

    def __init__(self, features=256, backbone='vitb16_128', readout='project', channels_last=False, train_pos_embed=True, norm_layer=None, use_skip_conv=True):
        super(VIT, self).__init__()
        self.channels_last = channels_last
        hooks = {'vitb16_128': [2, 5, 8, 11]}
        self.pretrained, self.scratch = _make_encoder(backbone, features, True, groups=1, expand=False, hooks=hooks[backbone], use_readout=readout)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_skip_conv = use_skip_conv
        if use_skip_conv:
            self.scratch.output_conv = nn.Sequential(nn.ReLU(True), nn.Conv2d(4 * features, features, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1))
            downsample = nn.Sequential(conv1x1(64, features // 2, 1), norm_layer(features // 2, track_running_stats=False, affine=True))
            self.scratch.skip_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect'), norm_layer(64, track_running_stats=False, affine=True), nn.ReLU(inplace=True), BasicBlock(64, features // 2, 1, downsample, norm_layer), BasicBlock(features // 2, features // 2, 1, None, norm_layer), BasicBlock(features // 2, features // 2, 1, None, norm_layer))
        else:
            self.scratch.output_conv = nn.Sequential(nn.ReLU(True), nn.Conv2d(4 * features, features, kernel_size=3, stride=1, padding=1))
        self.pretrained.model.pos_embed.requires_grad = train_pos_embed

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        if self.use_skip_conv:
            skip_x = self.scratch.skip_conv(x)
            sz = skip_x.shape[-2:]
        else:
            sz = layer_1.shape[-2:]
        new_latents = []
        new_latents.append(F.interpolate(layer_1_rn, sz, mode='bilinear', align_corners=True))
        new_latents.append(F.interpolate(layer_2_rn, sz, mode='bilinear', align_corners=True))
        new_latents.append(F.interpolate(layer_3_rn, sz, mode='bilinear', align_corners=True))
        new_latents.append(F.interpolate(layer_4_rn, sz, mode='bilinear', align_corners=True))
        new_latents = torch.cat(new_latents, 1)
        out = self.scratch.output_conv(new_latents)
        if self.use_skip_conv:
            out = torch.cat([out, skip_x], 1)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AddReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaussianActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PosEncodeResnet,
     lambda: ([], {'args': SimpleNamespace(freq_num=4), 'pos_size': 4, 'x_size': 4, 'hidden_size': 4, 'output_size': 4, 'block_num': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (ProjectReadout,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlock,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Slice,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

