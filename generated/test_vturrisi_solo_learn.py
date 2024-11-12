
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


from typing import Tuple


import torch.nn as nn


from torch.utils.data import DataLoader


import inspect


import logging


import copy


from functools import partial


import math


from functools import reduce


import torch.nn.functional as F


from typing import Callable


from typing import Optional


from typing import Union


import torchvision


from torch import nn


from torch.utils.data import Dataset


from torchvision import transforms


from torchvision.datasets import STL10


from torchvision.datasets import ImageFolder


from typing import List


import random


from typing import Sequence


from typing import Type


from torch.utils.data.dataset import Dataset


import torch.distributed as dist


import numpy as np


from typing import Any


from typing import Dict


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


import string


import time


import pandas as pd


from matplotlib import pyplot as plt


from scipy.sparse import csr_matrix


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import warnings


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from torch.cuda.amp import custom_fwd


from torch.nn.functional import conv2d


from torchvision.models.resnet import resnet18


from torchvision.datasets import CIFAR10


from torchvision.datasets.cifar import CIFAR10


from torchvision.models import resnet18


from torchvision.datasets import FakeData


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        """Computes standard normal cumulative distribution function"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        logging.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
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
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-05):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=GroupNorm, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)
    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_faat: whetehr output features of the 4 stages, for dense prediction
    --init_cfg，--pretrained:
        for mmdetection and mmsegmentation to load pretrianfed weights
    """

    def __init__(self, layers, embed_dims=None, mlp_ratios=None, downsamples=None, pool_size=3, norm_layer=GroupNorm, act_layer=nn.GELU, num_classes=1000, in_patch_size=7, in_stride=4, in_pad=2, down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.0, drop_path_rate=0.0, use_layer_scale=True, layer_scale_init_value=1e-05, fork_feat=False, init_cfg=None, pretrained=None, **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.num_features = embed_dims[-1]
        self.patch_embed = PatchEmbed(patch_size=in_patch_size, stride=in_stride, padding=in_pad, in_chans=3, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratios[i], act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(PatchEmbed(patch_size=down_patch_size, stride=down_stride, padding=down_pad, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out


class WideResnetBasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideResnetNetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, first_stride=1, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super().__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.num_features = channels[-1]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = WideResnetBasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = WideResnetNetworkBlock(n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block2 = WideResnetNetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = WideResnetNetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        x = out.view(-1, self.num_features)
        return x


class DINOLoss(nn.Module):

    def __init__(self, num_prototypes: 'int', warmup_teacher_temp: 'float', teacher_temp: 'float', warmup_teacher_temp_epochs: 'float', num_epochs: 'int', student_temp: 'float'=0.1, num_large_crops: 'int'=2, center_momentum: 'float'=0.9):
        """Auxiliary module to compute DINO's loss.

        Args:
            num_prototypes (int): number of prototypes.
            warmup_teacher_temp (float): base temperature for the temperature schedule
                of the teacher.
            teacher_temp (float): final temperature for the teacher.
            warmup_teacher_temp_epochs (float): number of epochs for the cosine annealing schedule.
            num_epochs (int): total number of epochs.
            student_temp (float, optional): temperature for the student. Defaults to 0.1.
            num_large_crops (int, optional): number of crops/views. Defaults to 2.
            center_momentum (float, optional): momentum for the EMA update of the center of
                mass of the teacher. Defaults to 0.9.
        """
        super().__init__()
        self.epoch = 0
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_large_crops = num_large_crops
        self.register_buffer('center', torch.zeros(1, num_prototypes))
        self.teacher_temp_schedule = np.concatenate((np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs), np.ones(num_epochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output: 'torch.Tensor', teacher_output: 'torch.Tensor') ->torch.Tensor:
        """Computes DINO's loss given a batch of logits of the student and a batch of logits of the
        teacher.

        Args:
            student_output (torch.Tensor): NxP Tensor containing student logits for all views.
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits for all views.

        Returns:
            torch.Tensor: DINO loss.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_large_crops)
        temp = self.teacher_temp_schedule[self.epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: 'torch.Tensor'):
        """Updates the center for DINO's loss using exponential moving average.

        Args:
            teacher_output (torch.Tensor): NxP Tensor containing teacher logits of all views.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOHead(nn.Module):
    mlp: 'Any'
    last_layer: 'Any'

    def __init__(self, in_dim: 'int', num_prototypes: 'int', use_bn: 'bool'=True, norm_last_layer: 'bool'=True, num_layers: 'int'=3, hidden_dim: 'int'=2048, bottleneck_dim: 'int'=256):
        """DINO head that takes as input the features of the backbone, projects them in a lower
        dimensional space and multiplies with the prototypes.

        Args:
            in_dim (int): number of dimensions of the input (aka backbone features).
            num_prototypes (int): number of prototypes.
            use_bn (bool, optional): whether to use batch norm in projector. Defaults to True.
            norm_last_layer (bool, optional): whether to l2-norm the last layer. Defaults to True.
            num_layers (int, optional): number of layers in projector. Defaults to 3.
            hidden_dim (int, optional): number of dimension in hidden layers. Defaults to 2048.
            bottleneck_dim (int, optional): number of dimensions in bottleneck. Defaults to 256.
        """
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: 'List[Any]' = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, num_prototypes, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: 'nn.Module'):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Computes the forward pass of the backbone, the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


def generate_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Adapted from https://github.com/facebookresearch/mae.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def generate_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = generate_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Adapted from https://github.com/facebookresearch/mae.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = generate_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MAEDecoder(nn.Module):

    def __init__(self, in_dim, embed_dim, depth, num_heads, num_patches, patch_size, mlp_ratio=4.0) ->None:
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * 3, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = generate_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x


def _1d_filter(tensor: 'torch.Tensor') ->torch.Tensor:
    return tensor.isfinite()


def _2d_filter(tensor: 'torch.Tensor') ->torch.Tensor:
    return tensor.isfinite().all(dim=1)


def _multi_input_filter(tensors: 'List[torch.Tensor]') ->Tuple[torch.Tensor]:
    if len(tensors[0].size()) == 1:
        filter_func = _1d_filter
    elif len(tensors[0].size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError('Only 1d and 2d tensors are supported.')
    selected = filter_func(tensors[0])
    for tensor in tensors[1:]:
        selected = torch.logical_and(selected, filter_func(tensor))
    tensors = [tensor[selected] for tensor in tensors]
    return tensors, selected


def _single_input_filter(tensor: 'torch.Tensor') ->Tuple[torch.Tensor]:
    if len(tensor.size()) == 1:
        filter_func = _1d_filter
    elif len(tensor.size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError('Only 1d and 2d tensors are supported.')
    selected = filter_func(tensor)
    tensor = tensor[selected]
    return tensor, selected


def filter_inf_n_nan(tensors: 'List[torch.Tensor]', return_indexes: 'bool'=False):
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (List): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """
    if isinstance(tensors, torch.Tensor):
        tensors, selected = _single_input_filter(tensors)
    else:
        tensors, selected = _multi_input_filter(tensors)
    if return_indexes:
        return tensors, selected
    return tensors


class FilterInfNNan(nn.Module):

    def __init__(self, module):
        """Layer that filters out inf and nans from any tensor.
        This is usefull when there are instability issues,
        which cause a small number of values to go bad.

        Args:
            tensor (List): tensor to remove nans and infs from.

        Returns:
            torch.Tensor: filtered view of the tensor without nans or infs.
        """
        super().__init__()
        self.module = module

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'module':
                raise AttributeError()
            return getattr(self.module, name)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)
        self.register_buffer('cached_penc', None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError('The input tensor has to be 3d!')
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x
        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):

    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)
        self.register_buffer('cached_penc', None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError('The input tensor has to be 4d!')
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y
        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):

    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)
        self.register_buffer('cached_penc', None)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError('The input tensor has to be 5d!')
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        sin_inp_z = torch.einsum('i,j->ij', pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z
        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):

    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):

    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert tensor.size() == penc.size(), 'The original tensor size {} and the positional encoding tensor size {} must match!'.format(tensor.size(), penc.size())
        return tensor + penc


class SinkhornKnopp(torch.nn.Module):

    def __init__(self, num_iters: 'int'=3, epsilon: 'float'=0.05, world_size: 'int'=1):
        """Approximates optimal transport using the Sinkhorn-Knopp algorithm.

        A simple iterative method to approach the double stochastic matrix is to alternately rescale
        rows and columns of the matrix to sum to 1.

        Args:
            num_iters (int, optional):  number of times to perform row and column normalization.
                Defaults to 3.
            epsilon (float, optional): weight for the entropy regularization term. Defaults to 0.05.
            world_size (int, optional): number of nodes for distributed training. Defaults to 1.
        """
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.world_size = world_size

    @torch.no_grad()
    def forward(self, Q: 'torch.Tensor') ->torch.Tensor:
        """Produces assignments using Sinkhorn-Knopp algorithm.

        Applies the entropy regularization, normalizes the Q matrix and then normalizes rows and
        columns in an alternating fashion for num_iter times. Before returning it normalizes again
        the columns in order for the output to be an assignment of samples to prototypes.

        Args:
            Q (torch.Tensor): cosine similarities between the features of the
                samples and the prototypes.

        Returns:
            torch.Tensor: assignment of samples to prototypes according to optimal transport.
        """
        Q = torch.exp(Q / self.epsilon).t()
        B = Q.shape[1] * self.world_size
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        for _ in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()


class Whitening2d(nn.Module):

    def __init__(self, output_dim: 'int', eps: 'float'=0.0):
        """Layer that computes hard whitening for W-MSE using the Cholesky decomposition.

        Args:
            output_dim (int): number of dimension of projected features.
            eps (float, optional): eps for numerical stability in Cholesky decomposition. Defaults
                to 0.0.
        """
        super().__init__()
        self.output_dim = output_dim
        self.eps = eps

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Performs whitening using the Cholesky decomposition.

        Args:
            x (torch.Tensor): a batch or slice of projected features.

        Returns:
            torch.Tensor: a batch or slice of whitened features.
        """
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.output_dim, -1).mean(-1).view(1, -1, 1, 1)
        xn = x - m
        T = xn.permute(1, 0, 2, 3).contiguous().view(self.output_dim, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)
        eye = torch.eye(self.output_dim).type(f_cov.type())
        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrinked), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.output_dim, 1, 1)
        decorrelated = conv2d(xn, inv_sqrt)
        return decorrelated.squeeze(2).squeeze(2)


class iterative_normalization_py(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args) ->torch.Tensor:
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(beta=eps, input=P[0], alpha=1.0 / m, batch1=xc, batch2=xc.transpose(1, 2))
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(beta=1.5, input=P[k], alpha=-0.5, batch1=torch.matrix_power(P[k], 3), batch2=Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())
            running_mean.copy_(momentum * mean + (1.0 - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1.0 - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_tensors
        if len(saved) == 0:
            return None, None, None, None, None, None, None, None
        xc = saved[0]
        rTr = saved[1]
        sn = saved[2].transpose(-2, -1)
        P = saved[3:]
        g, d, m = xc.size()
        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(beta=1.5, alpha=-0.5, batch1=g_tmp, batch2=P2)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P2, batch2=g_tmp)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1])
        g_sn += g_P
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):

    def __init__(self, num_features: 'int', num_groups: 'int'=64, num_channels: 'Optional[int]'=None, T: 'int'=5, dim: 'int'=2, eps: 'float'=1e-05, momentum: 'float'=0.1, affine: 'bool'=True):
        super().__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, f'num features={num_features}, num groups={num_groups}'
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*shape))
            self.bias = nn.Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T, self.eps, self.momentum, self.training)
        if self.affine:
            return X_hat * self.weight + self.bias
        return X_hat

    def extra_repr(self):
        return f'{self.num_features}, num_channels={self.num_channels}, T={self.T}, eps={self.eps}, momentum={{momentum}}, affine={{affine}}'


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DINOHead,
     lambda: ([], {'in_dim': 4, 'num_prototypes': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DINOLoss,
     lambda: ([], {'num_prototypes': 4, 'warmup_teacher_temp': 4, 'teacher_temp': 4, 'warmup_teacher_temp_epochs': 4, 'num_epochs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FilterInfNNan,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4])], {})),
    (GroupNorm,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IterNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNormChannel,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PoolFormerBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionalEncoding2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncodingPermute1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionalEncodingPermute2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinkhornKnopp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (Summer,
     lambda: ([], {'penc': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Whitening2d,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (WideResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (WideResnetBasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WideResnetNetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': torch.nn.ReLU, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

