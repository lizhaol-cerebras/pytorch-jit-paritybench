
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


import torch.nn as nn


from torch.utils.data import DataLoader


import time


import random


import torch.nn.functional as F


import torch.optim as optim


from scipy.optimize import least_squares


import copy


from torch.utils.data import Dataset


from collections import defaultdict


import math


import warnings


from collections import OrderedDict


from functools import partial


from itertools import repeat


from torch.nn import functional as F


import matplotlib


import matplotlib.pyplot as plt


import pandas as pd


from time import time


from torch.optim.lr_scheduler import StepLR


class MLP(nn.Module):

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


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_count_s = None
        self.attn_count_t = None

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        if self.mode == 'series':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'parallel':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x_t = self.forward_temporal(q, k, v, seqlen=seqlen)
            x_s = self.forward_spatial(q, k, v)
            alpha = torch.cat([x_s, x_t], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(B, 1, C, 2)
            alpha = alpha.softmax(dim=-1)
            x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
        elif self.mode == 'coupling':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_coupling(q, k, v, seqlen=seqlen)
        elif self.mode == 'vanilla':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def reshape_T(self, x, seqlen=1, inverse=False):
        if not inverse:
            N, C = x.shape[-2:]
            x = x.reshape(-1, seqlen, self.num_heads, N, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, seqlen * N, C)
        else:
            TN, C = x.shape[-2:]
            x = x.reshape(-1, self.num_heads, seqlen, TN // seqlen, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, TN // seqlen, C)
        return x

    def forward_coupling(self, q, k, v, seqlen=8):
        BT, _, N, C = q.shape
        q = self.reshape_T(q, seqlen)
        k = self.reshape_T(k, seqlen)
        v = self.reshape_T(v, seqlen)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.reshape_T(x, seqlen, inverse=True)
        x = x.transpose(1, 2).reshape(BT, N, C * self.num_heads)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        attn = qt @ kt.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x

    def count_attn(self, attn):
        attn = attn.detach().cpu().numpy()
        attn = attn.mean(axis=1)
        attn_t = attn[:, :, 1].mean(axis=1)
        attn_s = attn[:, :, 0].mean(axis=1)
        if self.attn_count_s is None:
            self.attn_count_s = attn_s
            self.attn_count_t = attn_t
        else:
            self.attn_count_s = np.concatenate([self.attn_count_s, attn_s], axis=0)
            self.attn_count_t = np.concatenate([self.attn_count_t, attn_t], axis=0)


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, mlp_out_ratio=1.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode='spatial')
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode='temporal')
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, seqlen=1):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == 'stage_para':
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                alpha = torch.cat([x_s, x_t], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DSTformer(nn.Module):

    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, st_mode='stage_st') for i in range(depth)])
        self.blocks_ts = nn.ModuleList([Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, st_mode='stage_ts') for i in range(depth)])
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(dim_feat, dim_rep)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, dim_out, global_pool=''):
        self.dim_out = dim_out
        self.head = nn.Linear(self.dim_feat, dim_out) if dim_out > 0 else nn.Identity()

    def forward(self, x, return_rep=False):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :]
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)
        alphas = []
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)
        if return_rep:
            return x
        x = self.head(x)
        return x

    def get_representation(self, x):
        return self.forward(x, return_rep=True)


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def get_angles(x):
    """
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    """
    limbs_id = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    angle_id = [[0, 3], [0, 6], [3, 6], [0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 10], [7, 13], [8, 13], [10, 13], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15]]
    eps = 1e-07
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    angles = limbs[:, :, angle_id, :]
    angle_cos = F.cosine_similarity(angles[:, :, :, 0, :], angles[:, :, :, 1, :], dim=-1)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))


def loss_angle(x, gt):
    """
        Input: (N, T, 17, 3), (N, T, 17, 3)
    """
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)


def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0]
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:, 1:] - x_a[:, :-1]
    gt_av = gt_a[:, 1:] - gt_a[:, :-1]
    return nn.L1Loss()(x_av, gt_av)


def get_limb_lens(x):
    """
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    """
    limbs_id = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens


def loss_limb_gt(x, gt):
    """
        Input: (N, T, 17, 3), (N, T, 17, 3)
    """
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)


def loss_limb_var(x):
    """
        Input: (N, T, 17, 3)
    """
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0]
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var


def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0]
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


class MeshLoss(nn.Module):

    def __init__(self, loss_type='MSE', device='cuda'):
        super(MeshLoss, self).__init__()
        self.device = device
        self.loss_type = loss_type
        if loss_type == 'MSE':
            self.criterion_keypoints = nn.MSELoss(reduction='none')
            self.criterion_regr = nn.MSELoss()
        elif loss_type == 'L1':
            self.criterion_keypoints = nn.L1Loss(reduction='none')
            self.criterion_regr = nn.L1Loss()

    def forward(self, smpl_output, data_gt):
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        data_3d_theta = reduce(data_gt['theta'])
        preds = smpl_output[-1]
        pred_theta = preds['theta']
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        preds_local = preds['kp_3d'] - preds['kp_3d'][:, :, 0:1, :]
        gt_local = data_gt['kp_3d'] - data_gt['kp_3d'][:, :, 0:1, :]
        real_shape, pred_shape = data_3d_theta[:, 72:], pred_theta[:, 72:]
        real_pose, pred_pose = data_3d_theta[:, :72], pred_theta[:, :72]
        loss_dict = {}
        loss_dict['loss_3d_pos'] = loss_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_scale'] = n_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_velocity'] = loss_velocity(preds_local, gt_local)
        loss_dict['loss_lv'] = loss_limb_var(preds_local)
        loss_dict['loss_lg'] = loss_limb_gt(preds_local, gt_local)
        loss_dict['loss_a'] = loss_angle(preds_local, gt_local)
        loss_dict['loss_av'] = loss_angle_velocity(preds_local, gt_local)
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_norm = torch.norm(pred_theta, dim=-1).mean()
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose
            loss_dict['loss_norm'] = loss_norm
        return loss_dict

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.0)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.0)
        return loss_regr_pose, loss_regr_betas


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class ActionHeadClassification(nn.Module):

    def __init__(self, dropout_ratio=0.0, dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        """
            Input: (N, M, T, J, C)
        """
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class ActionHeadEmbed(nn.Module):

    def __init__(self, dropout_ratio=0.0, dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)

    def forward(self, feat):
        """
            Input: (N, M, T, J, C)
        """
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat


class ActionNet(nn.Module):

    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0.0, version='class', hidden_dim=2048, num_joints=17):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        if version == 'class':
            self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, num_joints=num_joints)
        elif version == 'embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        else:
            raise Exception('Version Error.')

    def forward(self, x):
        """
            Input: (N, M x T x 17 x 3) 
        """
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, M, T, self.feat_J, -1])
        out = self.head(feat)
        return out


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-06)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-06)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    return rot_mats


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: 'torch.Tensor' = quaternion[..., 1]
    q2: 'torch.Tensor' = quaternion[..., 2]
    q3: 'torch.Tensor' = quaternion[..., 3]
    sin_squared_theta: 'torch.Tensor' = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: 'torch.Tensor' = torch.sqrt(sin_squared_theta)
    cos_theta: 'torch.Tensor' = quaternion[..., 0]
    two_theta: 'torch.Tensor' = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: 'torch.Tensor' = two_theta / sin_theta
    k_neg: 'torch.Tensor' = 2.0 * torch.ones_like(sin_theta)
    k: 'torch.Tensor' = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: 'torch.Tensor' = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


class SMPLRegressor(nn.Module):

    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.0):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints * dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints * dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        self.head_shape = nn.Linear(hidden_dim, 10)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)
        self.smpl = SMPL(args.data_root, batch_size=64, create_transl=False)
        mean_params = np.load(self.smpl.smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)
        feat_pose = feat.reshape(NT, -1)
        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)
        feat_shape = feat.permute(0, 2, 1)
        feat_shape = self.pool2(feat_shape).reshape(N, -1)
        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)
        pred_pose = self.init_pose.expand(NT, -1)
        pred_shape = self.init_shape.expand(N, -1)
        pred_pose = self.head_pose(feat_pose) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, N, -1).permute(1, 0, 2).reshape(NT, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices * 1000.0
        assert self.J_regressor is not None
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{'theta': torch.cat([pose, pred_shape], dim=1), 'verts': pred_vertices, 'kp_3d': pred_joints}]
        return output


class MeshRegressor(nn.Module):

    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
        super(MeshRegressor, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        """
            Input: (N x T x 17 x 3) 
        """
        N, T, J, C = x.shape
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, T, self.feat_J, -1])
        smpl_output = self.head(feat)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        return smpl_output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SupConLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

