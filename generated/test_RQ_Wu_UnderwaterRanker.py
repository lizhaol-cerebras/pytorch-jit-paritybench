
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


import torch.nn as nn


import torch


import torch.nn.functional as F


from functools import partial


from torch import nn


from torch import einsum


from torch.utils import data


from torchvision import transforms


from torchvision.transforms import InterpolationMode


import torchvision.transforms.functional as transFunc


import torchvision.transforms.functional as F


import numpy as np


from random import Random


import matplotlib.pyplot as plt


from torchvision.models.vgg import vgg16


from torchvision.models.vgg import vgg19


from scipy.stats import spearmanr


from scipy.stats import pearsonr


from scipy.stats import kendalltau


import time


from torch.utils.tensorboard import SummaryWriter


import random


from re import L


import warnings


import math


import logging


class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.InstanceNorm2d(out_channel), nn.ELU())

    def forward(self, x):
        y = self.out(x)
        return y


class ChannelAttention(nn.Module):

    def __init__(self, channels, factor):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_map = nn.Sequential(nn.Conv2d(channels, channels // factor, 1, 1, 0), nn.LeakyReLU(), nn.Conv2d(channels // factor, channels, 1, 1, 0), nn.Softmax())

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        ch_map = self.channel_map(avg_pool)
        return x * ch_map


class Encoder(nn.Module):

    def __init__(self, basic_channel):
        super(Encoder, self).__init__()
        self.e_stage1 = nn.Sequential(nn.Conv2d(3, basic_channel, 3, 1, 1), BasicBlock(basic_channel, basic_channel))
        self.e_stage2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), BasicBlock(basic_channel, basic_channel * 2))
        self.e_stage3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), BasicBlock(basic_channel * 2, basic_channel * 4))
        self.e_stage4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), BasicBlock(basic_channel * 4, basic_channel * 8))

    def forward(self, x):
        x1 = self.e_stage1(x)
        x2 = self.e_stage2(x1)
        x3 = self.e_stage3(x2)
        x4 = self.e_stage4(x3)
        return x1, x2, x3, x4


class Decoder(nn.Module):

    def __init__(self, basic_channel, is_residual=True):
        super(Decoder, self).__init__()
        self.is_residual = is_residual
        self.d_stage4 = nn.Sequential(BasicBlock(basic_channel * 8, basic_channel * 4), nn.UpsamplingBilinear2d(scale_factor=2))
        self.d_stage3 = nn.Sequential(BasicBlock(basic_channel * 4, basic_channel * 2), nn.UpsamplingBilinear2d(scale_factor=2))
        self.d_stage2 = nn.Sequential(BasicBlock(basic_channel * 2, basic_channel), nn.UpsamplingBilinear2d(scale_factor=2))
        self.d_stage1 = nn.Sequential(BasicBlock(basic_channel, basic_channel // 4))
        self.output = nn.Sequential(nn.Conv2d(basic_channel // 4, 3, 1, 1, 0), nn.Tanh())

    def forward(self, x, x1, x2, x3, x4):
        y3 = self.d_stage4(x4)
        y2 = self.d_stage3(y3 + x3)
        y1 = self.d_stage2(y2 + x2)
        y = self.output(self.d_stage1(y1 + x1))
        if self.is_residual:
            return y + x
        else:
            return y


def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        im_max = torch.max(img)
        im_min = torch.min(img)
        img = (img - im_min) / (im_max - im_min + 1e-07)
    return img


class NU2Net(nn.Module):

    def __init__(self, basic_channel=64, is_residual=True, tail='norm'):
        super(NU2Net, self).__init__()
        self.tail = tail
        self.encoder = Encoder(basic_channel)
        self.decoder = Decoder(basic_channel, is_residual=is_residual)
        if self.tail == 'IN+clip' or self.tail == 'IN+sigmoid':
            self.IN = nn.InstanceNorm2d(3)

    def forward(self, raw_img, **kwargs):
        x1, x2, x3, x4 = self.encoder(raw_img)
        y = self.decoder(raw_img, x1, x2, x3, x4)
        if self.tail == 'norm':
            y = normalize_img(y)
        elif self.tail == 'clip':
            y = torch.clamp(y, min=0.0, max=1.0)
        elif self.tail == 'sigmoid':
            y = torch.sigmoid(y)
        elif self.tail == 'IN+clip':
            y = torch.clamp(self.IN(y), min=0.0, max=1.0)
        elif self.tail == 'IN+sigmoid':
            y = torch.sigmoid(self.IN(y))
        elif self.tail == 'none':
            y = y
        return y


class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()
        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch, kernel_size=(cur_window, cur_window), padding=(padding_size, padding_size), dilation=(dilation, dilation), groups=cur_head_split * Ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [(x * Ch) for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W or N == 2 + H * W
        diff = N - H * W
        q_img = q[:, :, diff:, :]
        v_img = v[:, :, diff:, :]
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)
        EV_hat_img = q_img * conv_v_img
        zero = torch.zeros((B, h, diff, Ch), dtype=q.dtype, layout=q.layout, device=q.device)
        EV_hat = torch.cat((zero, EV_hat_img), dim=2)
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding. 
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W or N == 2 + H * W
        diff = N - H * W
        other_token, img_tokens = x[:, :diff], x[:, diff:]
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((other_token, x), dim=1)
        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()
        self.cpe = shared_cpe
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)
        return x


class ParallelBlock(nn.Module):
    """ Parallel block class. """

    def __init__(self, dims, num_heads, mlp_ratios=[], qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpes=None, shared_crpes=None, connect_type='neighbor'):
        super().__init__()
        self.connect_type = connect_type
        if self.connect_type == 'dynamic':
            self.alpha1 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha2 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha3 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha4 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha5 = nn.Parameter(torch.zeros(1) + 0.05)
            self.alpha6 = nn.Parameter(torch.zeros(1) + 0.05)
        self.cpes = shared_cpes
        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[1])
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[2])
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpes[3])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])
        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def upsample(self, x, output_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W or 2 + H * W
        diff = N - H * W
        other_token = x[:, :diff, :]
        img_tokens = x[:, diff:, :]
        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(img_tokens, size=output_size, mode='bilinear')
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)
        out = torch.cat((other_token, img_tokens), dim=1)
        return out

    def forward(self, x1, x2, x3, x4, sizes):
        _, (H2, W2), (H3, W3), (H4, W4) = sizes
        x2 = self.cpes[1](x2, size=(H2, W2))
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=(H2, W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3, W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4, W4))
        upsample3_2 = self.upsample(cur3, output_size=(H2, W2), size=(H3, W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3, W3), size=(H4, W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2, W2), size=(H4, W4))
        downsample2_3 = self.downsample(cur2, output_size=(H3, W3), size=(H2, W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4, W4), size=(H3, W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4, W4), size=(H2, W2))
        if self.connect_type == 'neighbor':
            cur2 = cur2 + upsample3_2
            cur3 = cur3 + upsample4_3 + downsample2_3
            cur4 = cur4 + downsample3_4
        elif self.connect_type == 'dense':
            cur2 = cur2 + upsample3_2 + upsample4_2
            cur3 = cur3 + upsample4_3 + downsample2_3
            cur4 = cur4 + downsample3_4 + downsample2_4
        elif self.connect_type == 'direct':
            cur2 = cur2
            cur3 = cur3
            cur4 = cur4
        elif self.connect_type == 'dynamic':
            cur2 = cur2 + self.alpha1 * upsample3_2 + self.alpha2 * upsample4_2
            cur3 = cur3 + self.alpha3 * upsample4_3 + self.alpha4 * downsample2_3
            cur4 = cur4 + self.alpha5 * downsample3_4 + self.alpha6 * downsample2_4
        del upsample3_2, upsample4_3, upsample4_2, downsample2_3, downsample2_4, downsample3_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        del cur2, cur3, cur4
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)
        return x1, x2, x3, x4


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        out = self.norm(x)
        return out, (out_H, out_W)


class URanker(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1, embed_dims=[0, 0, 0, 0], serial_depths=[0, 0, 0, 0], parallel_depth=0, num_heads=0, mlp_ratios=[0, 0, 0, 0], qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), return_interm_layers=False, out_features=None, crpe_window={(3): 2, (5): 3, (7): 3}, add_historgram=False, his_channel=192, connect_type='neighbor', **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes
        self.add_historgram = add_historgram
        self.connect_type = connect_type
        if self.add_historgram:
            self.historgram_embed1 = nn.Linear(his_channel, embed_dims[0])
            self.historgram_embed2 = nn.Linear(his_channel, embed_dims[1])
            self.historgram_embed3 = nn.Linear(his_channel, embed_dims[2])
            self.historgram_embed4 = nn.Linear(his_channel, embed_dims[3])
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        dpr = drop_path_rate
        self.serial_blocks1 = nn.ModuleList([SerialBlock(dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe1, shared_crpe=self.crpe1) for _ in range(serial_depths[0])])
        self.serial_blocks2 = nn.ModuleList([SerialBlock(dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe2, shared_crpe=self.crpe2) for _ in range(serial_depths[1])])
        self.serial_blocks3 = nn.ModuleList([SerialBlock(dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe3, shared_crpe=self.crpe3) for _ in range(serial_depths[2])])
        self.serial_blocks4 = nn.ModuleList([SerialBlock(dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe4, shared_crpe=self.crpe4) for _ in range(serial_depths[3])])
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([ParallelBlock(dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4], shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4], connect_type=self.connect_type) for _ in range(parallel_depth)])
        if not self.return_interm_layers:
            self.norm1 = norm_layer(embed_dims[0])
            self.norm2 = norm_layer(embed_dims[1])
            self.norm3 = norm_layer(embed_dims[2])
            self.norm4 = norm_layer(embed_dims[3])
            if self.parallel_depth > 0:
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.head2 = nn.Linear(embed_dims[3], num_classes)
                self.head3 = nn.Linear(embed_dims[3], num_classes)
                self.head4 = nn.Linear(embed_dims[3], num_classes)
            else:
                self.head2 = nn.Linear(embed_dims[3], num_classes)
                self.head3 = nn.Linear(embed_dims[3], num_classes)
                self.head4 = nn.Linear(embed_dims[3], num_classes)
        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        trunc_normal_(self.cls_token3, std=0.02)
        trunc_normal_(self.cls_token4, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def insert_his(self, x, his_token):
        x = torch.cat((his_token, x), dim=1)
        return x

    def remove_token(self, x):
        """ Remove CLS token. """
        if self.add_historgram:
            return x[:, 2:, :]
        else:
            return x[:, 1:, :]

    def forward_features(self, x0, x_his):
        B = x0.shape[0]
        x1, (H1, W1) = self.patch_embed1(x0)
        if self.add_historgram:
            x1 = self.insert_his(x1, self.historgram_embed1(x_his))
        x1 = self.insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_token(x1)
        x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2, (H2, W2) = self.patch_embed2(x1_nocls)
        if self.add_historgram:
            x2 = self.insert_his(x2, self.historgram_embed2(x_his))
        x2 = self.insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_token(x2)
        x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3, (H3, W3) = self.patch_embed3(x2_nocls)
        if self.add_historgram:
            x3 = self.insert_his(x3, self.historgram_embed3(x_his))
        x3 = self.insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_token(x3)
        x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x4, (H4, W4) = self.patch_embed4(x3_nocls)
        if self.add_historgram:
            x4 = self.insert_his(x4, self.historgram_embed4(x_his))
        x4 = self.insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_token(x4)
        x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        if self.parallel_depth == 0:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                x4_cls = x4[:, 0]
                return x4_cls
        for blk in self.parallel_blocks:
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])
        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = self.remove_token(x1)
                x1_nocls = x1_nocls.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = self.remove_token(x2)
                x2_nocls = x2_nocls.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = self.remove_token(x3)
                x3_nocls = x3_nocls.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = self.remove_token(x4)
                x4_nocls = x4_nocls.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            return x2_cls, x3_cls, x4_cls

    def forward(self, x, x_his):
        if self.return_interm_layers:
            return self.forward_features(x, x_his)
        else:
            x2, x3, x4 = self.forward_features(x, x_his)
            pred2 = self.head2(x2)
            pred3 = self.head3(x3)
            pred4 = self.head4(x4)
            x = (pred2 + pred3 + pred4) / 3.0
            result = {'final_result': x}
            return result


class perception_loss(nn.Module):

    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        h1 = self.to_relu_1_2(x1)
        h1 = self.to_relu_2_2(h1)
        h1 = self.to_relu_3_3(h1)
        h1 = self.to_relu_4_3(h1)
        h2 = self.to_relu_1_2(x2)
        h2 = self.to_relu_2_2(h2)
        h2 = self.to_relu_3_3(h2)
        h2 = self.to_relu_4_3(h2)
        return torch.mean(torch.abs(h1 - h2))


class perception_loss_norm_vgg19(nn.Module):

    def __init__(self):
        super(perception_loss_norm_vgg19, self).__init__()
        features = vgg19(pretrained=True).features
        self.to_relu_5_4 = features[:-1]
        self.requires_grad_(False)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        h1 = self.to_relu_5_4(x1)
        h2 = self.to_relu_5_4(x2)
        return torch.mean(torch.abs(h1 - h2))


class perception_loss_norm(perception_loss):

    def __init__(self):
        super().__init__()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        return super().forward(x1, x2)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelAttention,
     lambda: ([], {'channels': 4, 'factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {'basic_channel': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NU2Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (perception_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (perception_loss_norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512]), torch.rand([4, 3, 8, 8])], {})),
    (perception_loss_norm_vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
]

