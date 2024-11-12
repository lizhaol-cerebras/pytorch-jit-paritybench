
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


from functools import partial


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler as _DistributedSampler


import copy


import numpy as np


import pandas as pd


from torch import Tensor


from math import pi as PI


import torch.distributed as dist


from collections import defaultdict


import torch.utils.data as torch_data


import torch.nn.functional as F


import torchvision


from collections import namedtuple


import torch.nn as nn


from torch import nn


from torch.utils.checkpoint import checkpoint


from math import ceil


from collections import OrderedDict


from torch import hub


from torch.autograd import Variable


import math


import warnings


from copy import deepcopy


import torch.utils.checkpoint as cp


from torch.nn.init import kaiming_normal_


from scipy.optimize import linear_sum_assignment


import time


from typing import Optional


from typing import List


from torch.nn.init import xavier_uniform_


from torch.nn.init import zeros_


import collections.abc


from itertools import repeat


from typing import Sequence


from typing import ValuesView


from torch.autograd import Function


from typing import Tuple


import scipy


from scipy.spatial import Delaunay


import logging


import random


import torch.multiprocessing as mp


from typing import Set


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import re


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


from torch._utils import _unflatten_dense_tensors


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import clip_grad_norm_


import matplotlib


class BaseBEVBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [nn.ZeroPad2d(1), nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()]
            for k in range(layer_nums[idx]):
                cur_layers.extend([nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False):
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class BaseBEVBackboneV1(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)
        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [nn.ZeroPad2d(1), nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()]
            for k in range(layer_nums[idx]):
                cur_layers.extend([nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']
        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']
        ups = [self.deblocks[0](x_conv4)]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))
        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
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
        out = out + identity
        out = self.relu(out)
        return out


class BaseBEVResBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)]
            for k in range(layer_nums[idx]):
                cur_layers.extend([BasicBlock(num_filters[idx], num_filters[idx])])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class ConvFuser(nn.Module):

    def __init__(self, model_cfg) ->None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        cat_bev = torch.cat([img_bev, lidar_bev], dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights, out_channels=self.num_bev_features, **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict['voxel_features']
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)
        bev_features = self.block(bev_features)
        batch_dict['spatial_features'] = bev_features
        return batch_dict


class HeightCompression(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class PointPillarScatter(nn.Module):

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(self.num_bev_features, self.nz * self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(self.num_bev_features_before_compression, self.nz * self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(nn.Conv1d(input_channel, num_pos_feats, kernel_size=1), nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True), nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


def get_continous_inds(setnum_per_win):
    """
    Args:
        setnum_per_win (Tensor[int]): Number of sets assigned to each window with shape (win_num).
    Returns:
        set_win_inds (Tensor[int]): Window indexs of each set with shape (set_num).
        set_inds_in_win (Tensor[int]): Set indexs inner window with shape (set_num).

    Examples:
        setnum_per_win = torch.tensor([1, 2, 1, 3])
        set_inds_in_win = get_continous_inds(setnum_per_win)
        # we can get: set_inds_in_win = tensor([0, 0, 1, 0, 0, 1, 2])
    """
    set_num = setnum_per_win.sum().item()
    setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1]
    set_win_inds = torch.full((set_num,), 0, device=setnum_per_win.device)
    set_win_inds[setnum_per_win_cumsum] = 1
    set_win_inds = torch.cumsum(set_win_inds, dim=0)
    roll_set_win_inds_left = torch.roll(set_win_inds, -1)
    diff = set_win_inds - roll_set_win_inds_left
    end_pos_mask = diff != 0
    template = torch.ones_like(set_win_inds)
    template[end_pos_mask] = (setnum_per_win - 1) * -1
    set_inds_in_win = torch.cumsum(template, dim=0)
    set_inds_in_win[end_pos_mask] = setnum_per_win
    set_inds_in_win = set_inds_in_win - 1
    return set_win_inds, set_inds_in_win


class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):
        out_inds = torch.zeros_like(group_inds) - 1
        ingroup_indices.forward(group_inds, out_inds)
        ctx.mark_non_differentiable(out_inds)
        return out_inds

    @staticmethod
    def backward(ctx, g):
        return None


ingroup_inds = IngroupIndicesFunction.apply


get_inner_win_inds_cuda = ingroup_inds


def get_pooling_index(coors, sparse_shape, window_shape):
    win_shape_x, win_shape_y, win_shape_z = window_shape
    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    max_num_win_x = int(np.ceil(sparse_shape_x / win_shape_x))
    max_num_win_y = int(np.ceil(sparse_shape_y / win_shape_y))
    max_num_win_z = int(np.ceil(sparse_shape_z / win_shape_z))
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z
    coors_x = coors[:, 3]
    coors_y = coors[:, 2]
    coors_z = coors[:, 1]
    win_coors_x = coors_x // win_shape_x
    win_coors_y = coors_y // win_shape_y
    win_coors_z = coors_z // win_shape_z
    batch_win_inds = coors[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + win_coors_y * max_num_win_z + win_coors_z
    coors_in_win_x = coors_x % win_shape_x
    coors_in_win_y = coors_y % win_shape_y
    coors_in_win_z = coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    index_in_win = coors_in_win_x * win_shape_y * win_shape_z + coors_in_win_y * win_shape_z + coors_in_win_z
    batch_win_coords = torch.stack([coors[:, 0], win_coors_z, win_coors_y, win_coors_x], dim=-1)
    return batch_win_inds, coors_in_win, index_in_win, batch_win_coords


@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift, shift_list=None, return_win_coors=False):
    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape
    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'
    max_num_win_x = int(np.ceil(sparse_shape_x / win_shape_x) + 1)
    max_num_win_y = int(np.ceil(sparse_shape_y / win_shape_y) + 1)
    max_num_win_z = int(np.ceil(sparse_shape_z / win_shape_z) + 1)
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z
    if do_shift:
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    elif shift_list is not None:
        shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
    else:
        shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z
    if sparse_shape_z == win_shape_z:
        shift_z = 0
    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z
    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z
    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()
    batch_win_inds = coors[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + win_coors_y * max_num_win_z + win_coors_z
    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    if return_win_coors:
        batch_win_coords = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)
        return batch_win_inds, coors_in_win, batch_win_coords
    return batch_win_inds, coors_in_win


class DSVTInputLayer(nn.Module):
    """ 
    This class converts the output of vfe to dsvt input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        normalize_pos (bool): Whether to normalize coordinates in position embedding.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = self.model_cfg.sparse_shape
        self.window_shape = self.model_cfg.window_shape
        self.downsample_stride = self.model_cfg.downsample_stride
        self.d_model = self.model_cfg.d_model
        self.set_info = self.model_cfg.set_info
        self.stage_num = len(self.d_model)
        self.hybrid_factor = self.model_cfg.hybrid_factor
        self.window_shape = [[self.window_shape[s_id], [(self.window_shape[s_id][coord_id] * self.hybrid_factor[coord_id]) for coord_id in range(3)]] for s_id in range(self.stage_num)]
        self.shift_list = self.model_cfg.shifts_list
        self.normalize_pos = self.model_cfg.normalize_pos
        self.num_shifts = [2] * len(self.window_shape)
        self.sparse_shape_list = [self.sparse_shape]
        for ds_stride in self.downsample_stride:
            last_sparse_shape = self.sparse_shape_list[-1]
            self.sparse_shape_list.append((ceil(last_sparse_shape[0] / ds_stride[0]), ceil(last_sparse_shape[1] / ds_stride[1]), ceil(last_sparse_shape[2] / ds_stride[2])))
        self.posembed_layers = nn.ModuleList()
        for i in range(len(self.set_info)):
            input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2
            stage_posembed_layers = nn.ModuleList()
            for j in range(self.set_info[i][1]):
                block_posembed_layers = nn.ModuleList()
                for s in range(self.num_shifts[i]):
                    block_posembed_layers.append(PositionEmbeddingLearned(input_dim, self.d_model[i]))
                stage_posembed_layers.append(block_posembed_layers)
            self.posembed_layers.append(stage_posembed_layers)

    def forward(self, batch_dict):
        """
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region index used in pooling operation between stage_{i-1} and stage_{i}
                    with shape (N_{i-1}). 
                - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner region with shape (N_{i-1}). Combined with pooling_mapping_index_stage{i}, 
                    we can map each voxel in satge_{i-1} to pooling_preholder_feats_stage{i}, which are input of downsample operation.
                - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder features initial with value 0. 
                    Shape of (N_{i}, downsample_stride[i-1].prob(), d_moel[i-1]), where prob() returns the product of all elements.
                - ...
        """
        voxel_feats = batch_dict['voxel_features']
        voxel_coors = batch_dict['voxel_coords'].long()
        voxel_info = {}
        voxel_info['voxel_feats_stage0'] = voxel_feats.clone()
        voxel_info['voxel_coors_stage0'] = voxel_coors.clone()
        for stage_id in range(self.stage_num):
            voxel_info = self.window_partition(voxel_info, stage_id)
            voxel_info = self.get_set(voxel_info, stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    voxel_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = self.get_pos_embed(voxel_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
            if stage_id < self.stage_num - 1:
                voxel_info = self.subm_pooling(voxel_info, stage_id)
        return voxel_info

    @torch.no_grad()
    def subm_pooling(self, voxel_info, stage_id):
        cur_stage_downsample = self.downsample_stride[stage_id]
        batch_win_inds, _, index_in_win, batch_win_coors = get_pooling_index(voxel_info[f'voxel_coors_stage{stage_id}'], self.sparse_shape_list[stage_id], cur_stage_downsample)
        unique_batch_win_inds, contiguous_batch_win_inds = torch.unique(batch_win_inds, return_inverse=True)
        voxel_info[f'pooling_mapping_index_stage{stage_id + 1}'] = contiguous_batch_win_inds
        placeholder_prepool_feats = voxel_info[f'voxel_feats_stage0'].new_zeros((len(unique_batch_win_inds), torch.prod(torch.IntTensor(cur_stage_downsample)).item(), self.d_model[stage_id]))
        voxel_info[f'pooling_index_in_pool_stage{stage_id + 1}'] = index_in_win
        voxel_info[f'pooling_preholder_feats_stage{stage_id + 1}'] = placeholder_prepool_feats
        unique, inverse = unique_batch_win_inds.clone(), contiguous_batch_win_inds.clone()
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        pool_coors = batch_win_coors[perm]
        voxel_info[f'voxel_coors_stage{stage_id + 1}'] = pool_coors
        return voxel_info

    def get_set(self, voxel_info, stage_id):
        """
        This is one of the core operation of DSVT. 
        Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
        To make it clear and easy to follow, we do not use loop to process two shifts.
        Args:
            voxel_info (dict): 
                The dict contains the following keys
                - batch_win_inds_s{i} (Tensor[float]): Windows indexs of each voxel with shape (N), computed by 'window_partition'.
                - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner window of each voxel with shape (N, 3), computed by 'window_partition'.
                    Each row is (z, y, x). 
                - ...
        
        Returns:
            See from 'forward' function.
        """
        batch_win_inds_shift0 = voxel_info[f'batch_win_inds_stage{stage_id}_shift0']
        coors_in_win_shift0 = voxel_info[f'coors_in_win_stage{stage_id}_shift0']
        set_voxel_inds_shift0 = self.get_set_single_shift(batch_win_inds_shift0, stage_id, shift_id=0, coors_in_win=coors_in_win_shift0)
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0
        prefix_set_voxel_inds_s0 = torch.roll(set_voxel_inds_shift0.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s0[:, :, 0] = -1
        set_voxel_mask_s0 = set_voxel_inds_shift0 == prefix_set_voxel_inds_s0
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0
        batch_win_inds_shift1 = voxel_info[f'batch_win_inds_stage{stage_id}_shift1']
        coors_in_win_shift1 = voxel_info[f'coors_in_win_stage{stage_id}_shift1']
        set_voxel_inds_shift1 = self.get_set_single_shift(batch_win_inds_shift1, stage_id, shift_id=1, coors_in_win=coors_in_win_shift1)
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1
        prefix_set_voxel_inds_s1 = torch.roll(set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s1[:, :, 0] = -1
        set_voxel_mask_s1 = set_voxel_inds_shift1 == prefix_set_voxel_inds_s1
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1
        return voxel_info

    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        device = batch_win_inds.device
        voxel_num_set = self.set_info[stage_id][0]
        max_voxel = self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2]
        contiguous_win_inds = torch.unique(batch_win_inds, return_inverse=True)[1]
        voxelnum_per_win = torch.bincount(contiguous_win_inds)
        win_num = voxelnum_per_win.shape[0]
        setnum_per_win_float = voxelnum_per_win / voxel_num_set
        setnum_per_win = torch.ceil(setnum_per_win_float).long()
        set_win_inds, set_inds_in_win = get_continous_inds(setnum_per_win)
        offset_idx = set_inds_in_win[:, None].repeat(1, voxel_num_set) * voxel_num_set
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * voxelnum_per_win[set_win_inds][:, None]
        base_select_idx = base_select_idx.double() / (setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        base_select_idx = torch.floor(base_select_idx)
        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
        _, order1 = torch.sort(global_voxel_inds)
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)
        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * contiguous_win_inds
        voxel_inds_padding_sorty = -1 * torch.ones(win_num * max_voxel, dtype=torch.long, device=device)
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device)
        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]
        global_voxel_inds_sortx = contiguous_win_inds * max_voxel + coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sortx)
        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1])
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * torch.ones(win_num * max_voxel, dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device)
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]
        all_set_voxel_inds = torch.stack((set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)
        return all_set_voxel_inds

    @torch.no_grad()
    def window_partition(self, voxel_info, stage_id):
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'voxel_coors_stage{stage_id}'], self.sparse_shape_list[stage_id], self.window_shape[stage_id][i], i == 1, self.shift_list[stage_id][i])
            voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win
        return voxel_info

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        """
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        """
        window_shape = self.window_shape[stage_id][shift_id]
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3
        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z / 2, coors_in_win[:, 1] - win_y / 2, coors_in_win[:, 2] - win_x / 2
        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415
            y = y / win_y * 2 * 3.1415
            z = z / win_z * 2 * 3.1415
        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)
        return pos_embed


class Stage_ReductionAtt_Block(nn.Module):

    def __init__(self, input_channel, pool_volume):
        super().__init__()
        self.pool_volume = pool_volume
        self.query_func = torch.nn.MaxPool1d(pool_volume)
        self.norm = nn.LayerNorm(input_channel)
        self.self_attn = nn.MultiheadAttention(input_channel, 8, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(pool_volume, input_channel))
        nn.init.normal_(self.pos_embedding, std=0.01)

    def forward(self, x, key_padding_mask):
        src = self.query_func(x).permute(0, 2, 1)
        key = value = x.permute(0, 2, 1)
        key = key + self.pos_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query = src.clone()
        output = self.self_attn(query, key, value, key_padding_mask)[0]
        src = self.norm(output + src).squeeze(1)
        return src


class Stage_Reduction_Block(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=False)
        self.norm = nn.LayerNorm(output_channel)

    def forward(self, x):
        src = x
        src = self.norm(self.linear1(x))
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation='relu', batch_first=True, mlp_dropout=0):
        super().__init__()
        self.nhead = nhead
        if batch_first:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None):
        """
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
        Returns:
            src (Tensor[float]): Voxel features.
        """
        set_features = src[voxel_inds]
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features
        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]
        flatten_inds = voxel_inds.reshape(-1)
        unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
        src2 = src2.reshape(-1, self.d_model)[perm]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DSVT_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True, mlp_dropout=0):
        super().__init__()
        self.win_attn = SetAttention(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, set_voxel_inds, set_voxel_masks, pos=None):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds)
        src = src + identity
        src = self.norm(src)
        return src


class DSVTBlock(nn.Module):
    """ Consist of two encoder layer, shift and shift back.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True):
        super().__init__()
        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(self, src, set_voxel_inds_list, set_voxel_masks_list, pos_embed_list, block_id):
        num_shifts = 2
        output = src
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)
        return output


def _get_block_module(name):
    """Return an block module given a string"""
    if name == 'DSVTBlock':
        return DSVTBlock
    raise RuntimeError(f'This Block not exist.')


class DSVT(nn.Module):
    """Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_layer = DSVTInputLayer(self.model_cfg.INPUT_LAYER)
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        self.use_torch_ckpt = self.model_cfg.get('USE_CHECKPOINT', False)
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_module = _get_block_module(block_name_this_stage)
            block_list = []
            norm_list = []
            for i in range(num_blocks_this_stage):
                block_list.append(block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage, dropout, activation, batch_first=True))
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))
            if stage_id < stage_num - 1:
                downsample_window = self.model_cfg.INPUT_LAYER.downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id + 1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError
        self.num_shifts = [2] * stage_num
        self.output_shape = self.model_cfg.output_shape
        self.stage_num = stage_num
        self.set_info = set_info
        self.num_point_features = self.model_cfg.conv_out_channel
        self._reset_parameters()

    def forward(self, batch_dict):
        """
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        """
        voxel_info = self.input_layer(batch_dict)
        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s + 1}'] for s in range(self.stage_num - 1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s + 1}'] for s in range(self.stage_num - 1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s + 1}'] for s in range(self.stage_num - 1)]
        output = voxel_feat
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                if self.use_torch_ckpt == False:
                    output = block(output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], block_id=block_id)
                else:
                    output = checkpoint(block, output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], block_id)
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            if stage_id < self.stage_num - 1:
                prepool_features = pooling_preholder_feats[stage_id].type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id], pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)
                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        if self.pretrained:
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        return_layers = {_layer: _layer for _layer in feat_extract_layer}
        self.model.backbone.return_layers.update(return_layers)

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        model = constructor(pretrained=False, pretrained_backbone=False, num_classes=self.num_classes, aux_loss=self.aux_loss)
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)
            pretrained_dict = torch.load(self.pretrained_path)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        if 'aux_classifier.0.weight' in pretrained_dict and 'aux_classifier.0.weight' not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items() if 'aux_classifier' not in key}
        model_num_classes = model_dict['classifier.4.weight'].shape[0]
        pretrained_num_classes = pretrained_dict['classifier.4.weight'].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop('classifier.4.weight')
            pretrained_dict.pop('classifier.4.bias')
        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        if self.pretrained:
            images = (images - self.norm_mean[None, :, None, None].type_as(images)) / self.norm_std[None, :, None, None].type_as(images)
        x = images
        result = OrderedDict()
        features = self.model.backbone(x)
        for _layer in self.feat_extract_layer:
            result[_layer] = features[_layer]
        return result
        if 'features' in features.keys():
            feat_shape = features['features'].shape[-2:]
        else:
            feat_shape = features['layer1'].shape[-2:]
        x = features['out']
        result['logits'] = x
        if self.model.aux_classifier is not None:
            x = features['aux']
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result


class SemDeepLabV3(SegTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes SemDeepLabV3 model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50/ResNet101]
        """
        if backbone_name == 'ResNet50':
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == 'ResNet101':
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError
        super().__init__(constructor=constructor, **kwargs)


class PyramidFeat2D(nn.Module):

    def __init__(self, optimize, model_cfg):
        """
        Initialize 2D feature network via pretrained model
        Args:
            model_cfg: EasyDict, Dense classification network config
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.is_optimize = optimize
        self.ifn = SemDeepLabV3(num_classes=model_cfg.num_class, backbone_name=model_cfg.backbone, **model_cfg.args)
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.channel_reduce['in_channels']):
            _channel_out = model_cfg.channel_reduce['out_channels'][_idx]
            self.out_channels[model_cfg.args['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {'in_channels': _channel, 'out_channels': _channel_out, 'kernel_size': model_cfg.channel_reduce['kernel_size'][_idx], 'stride': model_cfg.channel_reduce['stride'][_idx], 'bias': model_cfg.channel_reduce['bias'][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

    def get_output_feature_dim(self):
        return self.out_channels

    def forward(self, images):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        batch_dict = {}
        ifn_result = self.ifn(images)
        for _idx, _layer in enumerate(self.model_cfg.args['feat_extract_layer']):
            image_features = ifn_result[_layer]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            batch_dict[_layer + '_feat2d'] = image_features
        if self.training:
            if 'logits' in ifn_result:
                ifn_result['logits'].detach_()
            if not self.is_optimize:
                image_features.detach_()
        return batch_dict

    def get_loss(self):
        """
        Gets loss
        Args:
        Returns:
            loss: (1), Network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        return None, None


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)
        mask = torch.Tensor(*size).fill_(0)
        index = index.view(*view)
        ones = 1.0
        if isinstance(index, Variable):
            ones = Variable(torch.Tensor(index.size()).fill_(1))
            mask = Variable(mask, volatile=index.volatile)
        return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma
        return loss.mean()


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t(torch.t(Ia) * wa) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)
    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]
    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = sector_idx == k
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(min(cur_num_points, math.ceil(ratio * num_sampled_points)))
    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        None
    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()
    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt).long()
    sampled_points = xyz[sampled_pt_idxs]
    return sampled_points


class VoxelSetAbstraction(nn.Module):

    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None, num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']
            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=input_channels, config=SA_cfg[src_name])
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += cur_num_c_out
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points'])
            c_in += cur_num_c_out
        self.vsa_point_feature_fusion = nn.Sequential(nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False), nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES), nn.ReLU())
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = keypoints[:, 0] == k
            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)
        point_bev_features = torch.cat(point_bev_features_list, dim=0)
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """
        sampled_points, _ = sample_points_with_roi(rois=roi_boxes, points=points, sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI, num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000))
        sampled_points = sector_fps(points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS, num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS)
        return sampled_points

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(batch_dict['voxel_coords'][:, 1:4], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = batch_indices == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS).long()
                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints = self.sectorized_proposal_centric_sampling(roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0])
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            else:
                raise NotImplementedError
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
        return keypoints

    @staticmethod
    def aggregate_keypoint_features_from_one_source(batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt, filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = xyz_bs_idxs == bs_idx
                _, valid_mask = sample_points_with_roi(rois=rois[bs_idx], points=xyz[bs_mask], sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part)
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()
            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()
        pooled_points, pooled_features = aggregate_func(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=xyz_features.contiguous())
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(keypoints, batch_dict['spatial_features'], batch_dict['batch_size'], bev_stride=batch_dict['spatial_features_stride'])
            point_features_list.append(point_bev_features)
        batch_size = batch_dict['batch_size']
        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            pooled_features = self.aggregate_keypoint_features_from_one_source(batch_size=batch_size, aggregate_func=self.SA_rawpoints, xyz=raw_points[:, 1:4], xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None, xyz_bs_idxs=raw_points[:, 0], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False), radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None), rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
            xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name], voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            pooled_features = self.aggregate_keypoint_features_from_one_source(batch_size=batch_size, aggregate_func=self.SA_layers[k], xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False), radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None), rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)
        point_features = torch.cat(point_features_list, dim=-1)
        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = keypoints
        return batch_dict


class PointNet2MSG(nn.Module):

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules.PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        point_features = l_features[0].permute(0, 2, 1).contiguous()
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules_stack.StackSAModuleMSG(radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules_stack.StackPointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points:(k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)
            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)
        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1], known=l_xyz[i], known_batch_cnt=l_batch_cnt[i], unknown_feats=l_features[i - 1], known_feats=l_features[i])
        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    m = spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())
    return m


class VoxelBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128
        self.backbone_channels = {'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


def replace_feature(out, new_features):
    if 'replace_feature' in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class VoxelResBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128
        self.backbone_channels = {'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 128}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False), norm_fn(out_channels), nn.ReLU())
    return m


class PillarBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        block = post_act_block
        dense_block = post_act_block_dense
        self.conv1 = spconv.SparseSequential(block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'), block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        norm_fn = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.conv5 = nn.Sequential(dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1), dense_block(256, 256, 3, norm_fn=norm_fn, padding=1), dense_block(256, 256, 3, norm_fn=norm_fn, padding=1))
        self.num_point_features = 256
        self.backbone_channels = {'x_conv1': 32, 'x_conv2': 64, 'x_conv3': 128, 'x_conv4': 256, 'x_conv5': 256}

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=pillar_features, indices=pillar_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        batch_dict.update({'multi_scale_2d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4, 'x_conv5': x_conv5}})
        batch_dict.update({'multi_scale_2d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8, 'x_conv5': 16}})
        return batch_dict


class PillarRes18BackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        block = post_act_block
        dense_block = post_act_block_dense
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'))
        norm_fn = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.conv5 = nn.Sequential(dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1), BasicBlock(256, 256, norm_fn=norm_fn), BasicBlock(256, 256, norm_fn=norm_fn))
        self.num_point_features = 256
        self.backbone_channels = {'x_conv1': 32, 'x_conv2': 64, 'x_conv3': 128, 'x_conv4': 256, 'x_conv5': 256}

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=pillar_features, indices=pillar_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        batch_dict.update({'multi_scale_2d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4, 'x_conv5': x_conv5}})
        batch_dict.update({'multi_scale_2d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8, 'x_conv5': 16}})
        return batch_dict


class ConfigDict:

    def __init__(self, name):
        self.name = name

    def __getitem__(self, item):
        return getattr(self, item)


def sort_by_indices(features, indices, features_add=None):
    """
        To sort the sparse features with its indices in a convenient manner.
        Args:
            features: [N, C], sparse features
            indices: [N, 4], indices of sparse features
            features_add: [N, C], additional features to sort
    """
    idx = indices[:, 1:]
    idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() + idx.select(1, 1) * idx[:, 2].max() + idx.select(1, 2)
    _, ind = idx_sum.sort()
    features = features[ind]
    indices = indices[ind]
    if not features_add is None:
        features_add = features_add[ind]
    return features, indices, features_add


def check_repeat(features, indices, features_add=None, sort_first=True, flip_first=True):
    """
        Check that whether there are replicate indices in the sparse features, 
        remove the replicate features if any.
    """
    if sort_first:
        features, indices, features_add = sort_by_indices(features, indices, features_add)
    if flip_first:
        features, indices = features.flip([0]), indices.flip([0])
    if not features_add is None:
        features_add = features_add.flip([0])
    idx = indices[:, 1:].int()
    idx_sum = torch.add(torch.add(idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max(), idx.select(1, 1) * idx[:, 2].max()), idx.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)
    if _unique.shape[0] < indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
        features_new.index_add_(0, inverse.long(), features)
        features = features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        indices = indices[perm_].int()
        if not features_add is None:
            features_add_new = torch.zeros((_unique.shape[0],), device=features_add.device)
            features_add_new.index_add_(0, inverse.long(), features_add)
            features_add = features_add_new / counts
    return features, indices, features_add


def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape
    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)
    return box_idxs_of_pts


def split_voxels(x, b, imps_3d, voxels_3d, kernel_offsets, mask_multi=True, topk=True, threshold=0.5):
    """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            x: [N, C], input sparse features
            b: int, batch size id
            imps_3d: [N, kernelsize**3], the prediced importance values
            voxels_3d: [N, 3], the 3d positions of voxel centers 
            kernel_offsets: [kernelsize**3, 3], the offset coords in an kernel
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
    """
    index = x.indices[:, 0]
    batch_index = index == b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    mask_voxel = imps_3d[batch_index, -1].sigmoid()
    mask_kernel = imps_3d[batch_index, :-1].sigmoid()
    if mask_multi:
        features_ori *= mask_voxel.unsqueeze(-1)
    if topk:
        _, indices = mask_voxel.sort(descending=True)
        indices_fore = indices[:int(mask_voxel.shape[0] * threshold)]
        indices_back = indices[int(mask_voxel.shape[0] * threshold):]
    else:
        indices_fore = mask_voxel > threshold
        indices_back = mask_voxel <= threshold
    features_fore = features_ori[indices_fore]
    coords_fore = indices_ori[indices_fore]
    mask_kernel_fore = mask_kernel[indices_fore]
    mask_kernel_bool = mask_kernel_fore >= threshold
    voxel_kerels_imp = kernel_offsets.unsqueeze(0).repeat(mask_kernel_bool.shape[0], 1, 1)
    mask_kernel_fore = mask_kernel[indices_fore][mask_kernel_bool]
    indices_fore_kernels = coords_fore[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
    indices_with_imp = indices_fore_kernels + voxel_kerels_imp
    selected_indices = indices_with_imp[mask_kernel_bool]
    spatial_indices = (selected_indices[:, 0] > 0) * (selected_indices[:, 1] > 0) * (selected_indices[:, 2] > 0) * (selected_indices[:, 0] < x.spatial_shape[0]) * (selected_indices[:, 1] < x.spatial_shape[1]) * (selected_indices[:, 2] < x.spatial_shape[2])
    selected_indices = selected_indices[spatial_indices]
    mask_kernel_fore = mask_kernel_fore[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_fore.device) * b, selected_indices], dim=1)
    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_fore.device)
    features_fore_cat = torch.cat([features_fore, selected_features], dim=0)
    coords_fore = torch.cat([coords_fore, selected_indices], dim=0)
    mask_kernel_fore = torch.cat([torch.ones(features_fore.shape[0], device=features_fore.device), mask_kernel_fore], dim=0)
    features_fore, coords_fore, mask_kernel_fore = check_repeat(features_fore_cat, coords_fore, features_add=mask_kernel_fore)
    features_back = features_ori[indices_back]
    coords_back = indices_ori[indices_back]
    return features_fore, coords_fore, features_back, coords_back, mask_kernel_fore


class objDict:

    @staticmethod
    def to_object(obj: 'object', **data):
        obj.__dict__.update(data)


class VoxelBackBone8xFocal(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU(True))
        block = post_act_block
        use_img = model_cfg.get('USE_IMG', False)
        topk = model_cfg.get('TOPK', True)
        threshold = model_cfg.get('THRESHOLD', 0.5)
        kernel_size = model_cfg.get('KERNEL_SIZE', 3)
        mask_multi = model_cfg.get('MASK_MULTI', False)
        skip_mask_kernel = model_cfg.get('SKIP_MASK_KERNEL', False)
        skip_mask_kernel_image = model_cfg.get('SKIP_MASK_KERNEL_IMG', False)
        enlarge_voxel_channels = model_cfg.get('ENLARGE_VOXEL_CHANNELS', -1)
        img_pretrain = model_cfg.get('IMG_PRETRAIN', '../checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth')
        if use_img:
            model_cfg_seg = dict(name='SemDeepLabV3', backbone='ResNet50', num_class=21, args={'feat_extract_layer': ['layer1'], 'pretrained_path': img_pretrain}, channel_reduce={'in_channels': [256], 'out_channels': [16], 'kernel_size': [1], 'stride': [1], 'bias': [False]})
            cfg_dict = ConfigDict('SemDeepLabV3')
            objDict.to_object(cfg_dict, **model_cfg_seg)
            self.semseg = PyramidFeat2D(optimize=True, model_cfg=cfg_dict)
            self.conv_focal_multimodal = FocalSparseConv(16, 16, image_channel=model_cfg_seg['channel_reduce']['out_channels'][0], topk=topk, threshold=threshold, use_img=True, skip_mask_kernel=skip_mask_kernel_image, voxel_stride=1, norm_fn=norm_fn, indice_key='spconv_focal_multimodal')
        special_spconv_fn = partial(FocalSparseConv, mask_multi=mask_multi, enlarge_voxel_channels=enlarge_voxel_channels, topk=topk, threshold=threshold, kernel_size=kernel_size, padding=kernel_size // 2, skip_mask_kernel=skip_mask_kernel)
        self.use_img = use_img
        self.conv1 = SparseSequentialBatchdict(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'), special_spconv_fn(16, 16, voxel_stride=1, norm_fn=norm_fn, indice_key='focal1'))
        self.conv2 = SparseSequentialBatchdict(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), special_spconv_fn(32, 32, voxel_stride=2, norm_fn=norm_fn, indice_key='focal2'))
        self.conv3 = SparseSequentialBatchdict(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), special_spconv_fn(64, 64, voxel_stride=4, norm_fn=norm_fn, indice_key='focal3'))
        self.conv4 = SparseSequentialBatchdict(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU(True))
        self.num_point_features = 128
        self.backbone_channels = {'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64}
        self.forward_ret_dict = {}

    def get_loss(self, tb_dict=None):
        loss = self.forward_ret_dict['loss_box_of_pts']
        if tb_dict is None:
            tb_dict = {}
        tb_dict['loss_box_of_pts'] = loss.item()
        return loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        loss_img = 0
        x = self.conv_input(input_sp_tensor)
        x_conv1, batch_dict, loss1 = self.conv1(x, batch_dict)
        if self.use_img:
            x_image = self.semseg(batch_dict['images'])['layer1_feat2d']
            x_conv1, batch_dict, loss_img = self.conv_focal_multimodal(x_conv1, batch_dict, x_image)
        x_conv2, batch_dict, loss2 = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict, loss3 = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict, loss4 = self.conv4(x_conv3, batch_dict)
        self.forward_ret_dict['loss_box_of_pts'] = loss1 + loss2 + loss3 + loss4 + loss_img
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


class VoxelResBackBone8xVoxelNeXt(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        out_channel = model_cfg.get('OUT_CHANNEL', 128)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'), norm_fn(channels[0]), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2), indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2), indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2), indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'))
        self.conv5 = spconv.SparseSequential(block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv5', conv_type='spconv'), SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'), SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'))
        self.conv6 = spconv.SparseSequential(block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv6', conv_type='spconv'), SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'), SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'))
        self.conv_out = spconv.SparseSequential(spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'), norm_fn(out_channel), nn.ReLU())
        self.shared_conv = spconv.SparseSequential(spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True), nn.BatchNorm1d(out_channel), nn.ReLU(True))
        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {'x_conv1': channels[0], 'x_conv2': channels[1], 'x_conv3': channels[2], 'x_conv4': channels[3]}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)
        x_out = spconv.SparseConvTensor(features=features_unique, indices=indices_unique, spatial_shape=spatial_shape, batch_size=x_conv.batch_size)
        return x_out

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        out = self.shared_conv(out)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


class VoxelResBackBone8xVoxelNeXt2D(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        block = post_act_block
        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2), indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2), indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2), indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'))
        self.conv5 = spconv.SparseSequential(block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv5', conv_type='spconv'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'))
        self.conv6 = spconv.SparseSequential(block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2), indice_key='spconv6', conv_type='spconv'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'), SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'))
        self.conv_out = spconv.SparseSequential(spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'), norm_fn(256), nn.ReLU())
        self.shared_conv = spconv.SparseSequential(spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True), nn.BatchNorm1d(256), nn.ReLU(True))
        self.num_point_features = 256
        self.backbone_channels = {'x_conv1': 32, 'x_conv2': 64, 'x_conv3': 128, 'x_conv4': 256, 'x_conv5': 256}
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)
        x_out = spconv.SparseConvTensor(features=features_unique, indices=indices_unique, spatial_shape=x_conv.spatial_shape, batch_size=x_conv.batch_size)
        return x_out

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=pillar_features, indices=pillar_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        out = self.shared_conv(out)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_2d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4, 'x_conv5': x_conv5}})
        batch_dict.update({'multi_scale_2d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8, 'x_conv5': 16}})
        return batch_dict


class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)
            self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        else:
            self.conv_out = None
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        if self.conv_out is not None:
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        batch_dict['point_features'] = x_up1.features
        point_coords = common_utils.get_voxel_centers(x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
        batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict


class PFNLayerV2(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class Sampler(nn.Module):

    def __init__(self, mode='bilinear', padding_mode='zeros'):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        if torch.__version__ >= '1.3':
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.grid_sample = F.grid_sample

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: (B, C, D, H, W), Input frustum features
            grid: (B, X, Y, Z, 3), Sampling grids for input features
        Returns
            output_features: (B, C, X, Y, Z) Output voxel features
        """
        output = self.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size, pc_range=pc_range, disc_cfg=disc_cfg)
        self.sampler = Sampler(**model_cfg.SAMPLER)

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features: (B, C, D, H_image, W_image), Image frustum features
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        grid = self.grid_generator(lidar_to_cam=batch_dict['trans_lidar_to_cam'], cam_to_img=batch_dict['trans_cam_to_img'], image_shape=batch_dict['image_shape'])
        voxel_features = self.sampler(input_features=batch_dict['frustum_features'], grid=grid)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        batch_dict['voxel_features'] = voxel_features
        return batch_dict


class DDNTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        if self.pretrained:
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {feat_extract_layer: 'features', **self.model.backbone.return_layers}

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        model = constructor(pretrained=False, pretrained_backbone=False, num_classes=self.num_classes, aux_loss=self.aux_loss)
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict, pretrained_dict=pretrained_dict)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        if 'aux_classifier.0.weight' in pretrained_dict and 'aux_classifier.0.weight' not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items() if 'aux_classifier' not in key}
        model_num_classes = model_dict['classifier.4.weight'].shape[0]
        pretrained_num_classes = pretrained_dict['classifier.4.weight'].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop('classifier.4.weight')
            pretrained_dict.pop('classifier.4.bias')
        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        x = self.preprocess(images)
        result = OrderedDict()
        features = self.model.backbone(x)
        result['features'] = features['features']
        feat_shape = features['features'].shape[-2:]
        x = features['out']
        x = self.model.classifier(x)
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result['logits'] = x
        if self.model.aux_classifier is not None:
            x = features['aux']
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            mask = x == 0
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)
            x[mask] = 0
        return x


class Balancer(nn.Module):

    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss: (B, H, W), Pixel-wise loss
            gt_boxes2d: (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        fg_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d, shape=loss.shape, downsample_factor=self.downsample_factor, device=loss.device)
        bg_mask = ~fg_mask
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels
        loss = fg_loss + bg_loss
        tb_dict = {'balancer_loss': loss.item(), 'fg_loss': fg_loss.item(), 'bg_loss': bg_loss.item()}
        return loss, tb_dict


class DDNLoss(nn.Module):

    def __init__(self, weight, alpha, gamma, disc_cfg, fg_weight, bg_weight, downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor, fg_weight=fg_weight, bg_weight=bg_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='none')
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}
        depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)
        loss = self.loss_func(depth_logits, depth_target)
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        loss *= self.weight
        tb_dict.update({'ddn_loss': loss.item()})
        return loss, tb_dict


class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](num_classes=self.disc_cfg['num_bins'] + 1, backbone_name=model_cfg.DDN.BACKBONE_NAME, **model_cfg.DDN.ARGS)
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](disc_cfg=self.disc_cfg, downsample_factor=downsample_factor, **model_cfg.LOSS.ARGS)
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        images = batch_dict['images']
        ddn_result = self.ddn(images)
        image_features = ddn_result['features']
        depth_logits = ddn_result['logits']
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)
        frustum_features = self.create_frustum_features(image_features=image_features, depth_logits=depth_logits)
        batch_dict['frustum_features'] = frustum_features
        if self.training:
            self.forward_ret_dict['depth_maps'] = batch_dict['depth_maps']
            self.forward_ret_dict['gt_boxes2d'] = batch_dict['gt_boxes2d']
            self.forward_ret_dict['depth_logits'] = depth_logits
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part]) for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class VFETemplate(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError


class GeneralizedLSSFPN(nn.Module):
    """
        This module implements FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py with minimal modifications.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels = self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL
        self.in_channels = in_channels
        if end_level == -1:
            self.backbone_end_level = num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = BasicBlock2D(in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels), out_channels, kernel_size=1, bias=False)
            fpn_conv = BasicBlock2D(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        inputs = batch_dict['image_features']
        assert len(inputs) == len(self.in_channels)
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        outs = [laterals[i] for i in range(used_backbone_levels)]
        batch_dict['image_fpn'] = tuple(outs)
        return batch_dict


def _no_grad_trunc_normal_(tensor: 'Tensor', mean: 'float', std: 'float', a: 'float', b: 'float') ->Tensor:

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: 'Tensor', mean: 'float'=0.0, std: 'float'=1.0, a: 'float'=-2.0, b: 'float'=2.0) ->Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self._is_init = False
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


def drop_path(x: 'torch.Tensor', drop_prob: 'float'=0.0, training: 'bool'=False) ->torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: 'float'=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
    """

    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0, dropout_layer=dict(type='DropPath', drop_prob=0.0)):
        super().__init__()
        self._is_init = False
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=to_2tuple(window_size), qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)
        self.drop = DropPath(dropout_layer['drop_prob'])

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, dropout_layer=None, add_identity=True, init_cfg=None, **kwargs):
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = nn.GELU()
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(nn.Linear(in_channels, feedforward_channels), self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(dropout_layer['drop_prob'])
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class SwinBlock(nn.Module):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size=7, shift=False, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False):
        super(SwinBlock, self).__init__()
        self._is_init = False
        self.with_cp = with_cp
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2 if shift else 0, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=2, ffn_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), act_cfg=act_cfg, add_identity=True)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class SwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, depth, window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, downsample=None, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False):
        super().__init__()
        self._is_init = False
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, window_size=window_size, shift=False if i % 2 == 0 else True, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i], act_cfg=act_cfg, norm_cfg=norm_cfg, with_cp=with_cp)
            self.blocks.append(block)
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self, in_channels=3, embed_dims=768, conv_type='Conv2d', kernel_size=16, stride=16, padding='corner', dilation=1, bias=True, norm_cfg=None, input_size=None, init_cfg=None):
        super(PatchEmbed, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        if norm_cfg is not None:
            self.norm = nn.LayerNorm(embed_dims)
        else:
            self.norm = None
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = input_h, input_w
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = h_out, w_out
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """
        if self.adap_padding:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(nn.Module):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=None, padding='corner', dilation=1, bias=False, norm_cfg=dict(type='LN'), init_cfg=None):
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        if norm_cfg is not None:
            self.norm = nn.LayerNorm(sample_dim)
        else:
            self.norm = None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect input_size is `Sequence` but get {input_size}'
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]
        x = self.sampler(x)
        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1
        output_size = out_h, out_w
        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


def constant_init(module: 'nn.Module', val: 'float', bias: 'float'=0) ->None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def swin_converter(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k
        new_ckpt['backbone.' + new_k] = new_v
    return new_ckpt


def trunc_normal_init(module: 'nn.Module', mean: 'float'=0, std: 'float'=1, a: 'float'=-2, b: 'float'=2, bias: 'float'=0) ->None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SwinTransformer(nn.Module):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/swin.py 
    with minimal modifications.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        pretrain_img_size = self.model_cfg.get('PRETRAIN_IMG_SIZE', 224)
        init_cfg = self.model_cfg.get('INIT_CFG', None)
        depths = self.model_cfg.DEPTHS
        in_channels = self.model_cfg.get('IN_CHANNELS', 3)
        strides = self.model_cfg.get('STRIDES', (4, 2, 2, 2))
        patch_size = self.model_cfg.get('PATCH_SIZE', 4)
        embed_dims = self.model_cfg.EMBED_DIMS
        num_heads = self.model_cfg.NUM_HEADS
        window_size = self.model_cfg.WINDOW_SIZE
        mlp_ratio = self.model_cfg.MLP_RATIO
        qkv_bias = self.model_cfg.get('QKV_BIAS', True)
        qk_scale = self.model_cfg.get('QK_SCALE', None)
        drop_rate = self.model_cfg.DROP_RATE
        attn_drop_rate = self.model_cfg.ATTN_DROP_RATE
        drop_path_rate = self.model_cfg.DROP_PATH_RATE
        patch_norm = self.model_cfg.get('PATCH_NORM', True)
        out_indices = self.model_cfg.get('OUT_INDICES', [0, 1, 2, 3])
        with_cp = self.model_cfg.get('WITH_CP', False)
        use_abs_pos_embed = self.model_cfg.get('USE_ABS_POS_EMBED', False)
        act_cfg = dict(type='GELU')
        norm_cfg = dict(type='LN')
        self.convert_weights = self.model_cfg.get('CONVERT_WEIGHTS', False)
        self.frozen_stages = self.model_cfg.get('FROZEN_STAGES', -1)
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, f'The size of image should have length 1 or 2, but got {len(pretrain_img_size)}'
        super(SwinTransformer, self).__init__()
        self.init_cfg = init_cfg
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims, conv_type='Conv2d', kernel_size=patch_size, stride=strides[0], norm_cfg=norm_cfg if patch_norm else None)
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = nn.ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(in_channels=in_channels, out_channels=2 * in_channels, stride=strides[i + 1], norm_cfg=norm_cfg if patch_norm else None)
            else:
                downsample = None
            stage = SwinBlockSequence(embed_dims=in_channels, num_heads=num_heads[i], feedforward_channels=mlp_ratio * in_channels, depth=depths[i], window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=downsample, act_cfg=act_cfg, norm_cfg=norm_cfg, with_cp=with_cp)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()
        for i in range(1, self.frozen_stages + 1):
            if i - 1 in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            None
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__} '
            ckpt = torch.load(self.init_cfg.checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                _state_dict = swin_converter(_state_dict)
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    None
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
            relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    None
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()
            self.load_state_dict(state_dict, False)

    def forward(self, batch_dict):
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        batch_dict['image_features'] = outs
        return batch_dict


class SingleHead(BaseBEVBackbone):

    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, rpn_head_cfg=None, head_label_indices=None, separate_reg_config=None):
        super().__init__(rpn_head_cfg, input_channels)
        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer('head_label_indices', head_label_indices)
        if self.separate_reg_config is not None:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            num_middle_conv = self.separate_reg_config.NUM_MIDDLE_CONV
            num_middle_filter = self.separate_reg_config.NUM_MIDDLE_FILTER
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_conv):
                conv_cls_list.extend([nn.Conv2d(c_in, num_middle_filter, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_middle_filter), nn.ReLU()])
                c_in = num_middle_filter
            conv_cls_list.append(nn.Conv2d(c_in, self.num_anchors_per_location * self.num_class, kernel_size=3, stride=1, padding=1))
            self.conv_cls = nn.Sequential(*conv_cls_list)
            for reg_config in self.separate_reg_config.REG_LIST:
                reg_name, reg_channel = reg_config.split(':')
                reg_channel = int(reg_channel)
                cur_conv_list = []
                c_in = input_channels
                for k in range(num_middle_conv):
                    cur_conv_list.extend([nn.Conv2d(c_in, num_middle_filter, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_middle_filter), nn.ReLU()])
                    c_in = num_middle_filter
                cur_conv_list.append(nn.Conv2d(c_in, self.num_anchors_per_location * int(reg_channel), kernel_size=3, stride=1, padding=1, bias=True))
                code_size_cnt += reg_channel
                self.conv_box[f'conv_{reg_name}'] = nn.Sequential(*cur_conv_list)
                self.conv_box_names.append(f'conv_{reg_name}')
            for m in self.conv_box.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            assert code_size_cnt == code_size, f'Code size does not match: {code_size_cnt}:{code_size}'
        else:
            self.conv_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1)
            self.conv_box = nn.Conv2d(input_channels, self.num_anchors_per_location * self.code_size, kernel_size=1)
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, kernel_size=1)
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)
        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location, self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location, self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(-1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4, 2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None
        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds
        return ret_dict


class ATSSTargetAssigner(object):
    """
    Reference: https://arxiv.org/abs/1912.02424
    """

    def __init__(self, topk, box_coder, match_height=False):
        self.topk = topk
        self.box_coder = box_coder
        self.match_height = match_height

    def assign_targets(self, anchors_list, gt_boxes_with_classes, use_multihead=False):
        """
        Args:
            anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        if not isinstance(anchors_list, list):
            anchors_list = [anchors_list]
            single_set_of_anchor = True
        else:
            single_set_of_anchor = len(anchors_list) == 1
        cls_labels_list, reg_targets_list, reg_weights_list = [], [], []
        for anchors in anchors_list:
            batch_size = gt_boxes_with_classes.shape[0]
            gt_classes = gt_boxes_with_classes[:, :, -1]
            gt_boxes = gt_boxes_with_classes[:, :, :-1]
            if use_multihead:
                anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
            else:
                anchors = anchors.view(-1, anchors.shape[-1])
            cls_labels, reg_targets, reg_weights = [], [], []
            for k in range(batch_size):
                cur_gt = gt_boxes[k]
                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1
                cur_gt = cur_gt[:cnt + 1]
                cur_gt_classes = gt_classes[k][:cnt + 1]
                cur_cls_labels, cur_reg_targets, cur_reg_weights = self.assign_targets_single(anchors, cur_gt, cur_gt_classes)
                cls_labels.append(cur_cls_labels)
                reg_targets.append(cur_reg_targets)
                reg_weights.append(cur_reg_weights)
            cls_labels = torch.stack(cls_labels, dim=0)
            reg_targets = torch.stack(reg_targets, dim=0)
            reg_weights = torch.stack(reg_weights, dim=0)
            cls_labels_list.append(cls_labels)
            reg_targets_list.append(reg_targets)
            reg_weights_list.append(reg_weights)
        if single_set_of_anchor:
            ret_dict = {'box_cls_labels': cls_labels_list[0], 'box_reg_targets': reg_targets_list[0], 'reg_weights': reg_weights_list[0]}
        else:
            ret_dict = {'box_cls_labels': torch.cat(cls_labels_list, dim=1), 'box_reg_targets': torch.cat(reg_targets_list, dim=1), 'reg_weights': torch.cat(reg_weights_list, dim=1)}
        return ret_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes):
        """
        Args:
            anchors: (N, 7) [x, y, z, dx, dy, dz, heading]
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
            gt_classes: (M)
        Returns:

        """
        num_anchor = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        if self.match_height:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
        else:
            ious = iou3d_nms_utils.boxes_iou_bev(anchors[:, 0:7], gt_boxes[:, 0:7])
        distance = (anchors[:, None, 0:3] - gt_boxes[None, :, 0:3]).norm(dim=-1)
        _, topk_idxs = distance.topk(self.topk, dim=0, largest=False)
        candidate_ious = ious[topk_idxs, torch.arange(num_gt)]
        iou_mean_per_gt = candidate_ious.mean(dim=0)
        iou_std_per_gt = candidate_ious.std(dim=0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt + 1e-06
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]
        candidate_anchors = anchors[topk_idxs.view(-1)]
        gt_boxes_of_each_anchor = gt_boxes[:, :].repeat(self.topk, 1)
        xyz_local = candidate_anchors[:, 0:3] - gt_boxes_of_each_anchor[:, 0:3]
        xyz_local = common_utils.rotate_points_along_z(xyz_local[:, None, :], -gt_boxes_of_each_anchor[:, 6]).squeeze(dim=1)
        xy_local = xyz_local[:, 0:2]
        lw = gt_boxes_of_each_anchor[:, 3:5][:, [1, 0]]
        is_in_gt = ((xy_local <= lw / 2) & (xy_local >= -lw / 2)).all(dim=-1).view(-1, num_gt)
        is_pos = is_pos & is_in_gt
        for ng in range(num_gt):
            topk_idxs[:, ng] += ng * num_anchor
        INF = -2147483647
        ious_inf = torch.full_like(ious, INF).t().contiguous().view(-1)
        index = topk_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gt, -1).t()
        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
        max_iou_of_each_gt, argmax_iou_of_each_gt = ious.max(dim=0)
        anchors_to_gt_indexs[argmax_iou_of_each_gt] = torch.arange(0, num_gt, device=ious.device)
        anchors_to_gt_values[argmax_iou_of_each_gt] = max_iou_of_each_gt
        cls_labels = gt_classes[anchors_to_gt_indexs]
        cls_labels[anchors_to_gt_values == INF] = 0
        matched_gts = gt_boxes[anchors_to_gt_indexs]
        pos_mask = cls_labels > 0
        reg_targets = matched_gts.new_zeros((num_anchor, self.box_coder.code_size))
        reg_weights = matched_gts.new_zeros(num_anchor)
        if pos_mask.sum() > 0:
            reg_targets[pos_mask > 0] = self.box_coder.encode_torch(matched_gts[pos_mask > 0], anchors[pos_mask > 0])
            reg_weights[pos_mask] = 1.0
        return cls_labels, reg_targets, reg_weights


class AnchorGenerator(object):

    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]
        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0
            x_shifts = torch.arange(self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-05, step=x_stride, dtype=torch.float32)
            y_shifts = torch.arange(self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-05, step=y_stride, dtype=torch.float32)
            z_shifts = x_shifts.new_tensor(anchor_height)
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([x_shifts, y_shifts, z_shifts])
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            anchors[..., 2] += anchors[..., 5] / 2
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


class AxisAlignedTargetAssigner(object):

    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        bbox_targets = []
        cls_labels = []
        reg_weights = []
        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()
            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([(self.class_names[c - 1] == anchor_class_name) for c in cur_gt_classes], dtype=torch.bool)
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                single_target = self.assign_targets_single(anchors, cur_gt[mask], gt_classes=selected_classes, matched_threshold=self.matched_thresholds[anchor_class_name], unmatched_threshold=self.unmatched_thresholds[anchor_class_name])
                target_list.append(single_target)
            if self.use_multihead:
                target_dict = {'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list], 'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list], 'reg_weights': [t['reg_weights'].view(-1) for t in target_list]}
                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list], 'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size) for t in target_list], 'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]}
                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=-2).view(-1, self.box_coder.code_size)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
        bbox_targets = torch.stack(bbox_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        all_targets_dict = {'box_cls_labels': cls_labels, 'box_reg_targets': bbox_targets, 'reg_weights': reg_weights}
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)
        fg_inds = (labels > 0).nonzero()[:, 0]
        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]
            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
        elif len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)
        reg_weights = anchors.new_zeros((num_anchors,))
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0
        ret_dict = {'box_cls_labels': labels, 'box_reg_targets': bbox_targets, 'reg_weights': reg_weights}
        return ret_dict


class AnchorHeadTemplate(nn.Module):

    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6), **anchor_target_cfg.get('BOX_CODER_CONFIG', {}))
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range, anchor_ndim=self.box_coder.code_size)
        self.anchors = [x for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)
        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(anchor_range=point_cloud_range, anchor_generator_config=anchor_generator_cfg)
        feature_map_size = [(grid_size[:2] // config['feature_map_stride']) for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors
        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(topk=anchor_target_cfg.TOPK, box_coder=self.box_coder, use_multihead=self.use_multihead, match_height=anchor_target_cfg.MATCH_HEIGHT)
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(model_cfg=self.model_cfg, class_names=self.class_names, box_coder=self.box_coder, match_height=anchor_target_cfg.MATCH_HEIGHT)
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None else losses_cfg.REG_LOSS_TYPE
        self.add_module('reg_loss_func', getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']))
        self.add_module('dir_loss_func', loss_utils.WeightedCrossEntropyLoss())

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(self.anchors, gt_boxes)
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {'rpn_loss_cls': cls_loss.item()}
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype, device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_preds.shape[-1])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {'rpn_loss_loc': loc_loss.item()}
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(anchors, box_reg_targets, dir_offset=self.model_cfg.DIR_OFFSET, num_bins=self.model_cfg.NUM_DIR_BINS)
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = 2 * np.pi / self.model_cfg.NUM_DIR_BINS
            dir_rot = common_utils.limit_period(batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period)
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels
        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(-(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2)
        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class SeparateHead(nn.Module):

    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']
            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size // 2), bias=use_bias, indice_key=cur_name), nn.BatchNorm1d(input_channels), nn.ReLU()))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name + 'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, 'bias') and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features
        return ret_dict


class CenterHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array([self.class_names.index(x) for x in cur_class_names if x in class_names]))
            self.class_id_mapping_each_head.append(cur_class_id_mapping)
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'
        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-05), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(nn.Conv2d(input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1, bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)), norm_func(self.model_cfg.SHARED_CONV_CHANNEL), nn.ReLU())
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(SeparateHead(input_channels=self.model_cfg.SHARED_CONV_CHANNEL, sep_head_dict=cur_head_dict, init_bias=-2.19, use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False), norm_func=norm_func))
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1
            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'heatmap_masks': [], 'target_boxes_src': []}
        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                gt_boxes_single_head = []
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])
                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)
                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(), feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE, num_max_objs=target_assigner_cfg.NUM_MAX_OBJS, gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP, min_radius=target_assigner_cfg.MIN_RADIUS)
                heatmap_list.append(heatmap)
                target_boxes_list.append(ret_boxes)
                inds_list.append(inds)
                masks_list.append(mask)
                target_boxes_src_list.append(ret_boxes_src)
            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=0.0001, max=1 - 0.0001)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        tb_dict = {}
        loss = 0
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):
                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(pred_dict=pred_dict, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, feature_map_stride=self.feature_map_stride)
                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)
                    iou_loss = loss_utils.calculate_iou_loss_centerhead(iou_preds=pred_dict['iou'], batch_box_preds=batch_box_preds_for_iou.clone().detach(), mask=target_dicts['masks'][idx], ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx])
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(batch_box_preds=batch_box_preds_for_iou, mask=target_dicts['masks'][idx], ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx])
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.0).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.0).sum()
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).float()
        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None
            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin, center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, feature_map_stride=self.feature_map_stride, K=post_process_cfg.MAX_OBJ_PER_SAMPLE, circle_nms=post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms', score_thresh=post_process_cfg.SCORE_THRESH, post_center_limit_range=post_center_limit_range)
            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])
                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'], nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None)
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'], box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None))
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError
                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]
                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1
        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_dicts[0]['pred_boxes']
        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])
            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        if self.training:
            target_dict = self.assign_targets(data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:], feature_map_stride=data_dict.get('spatial_features_2d_strides', None))
            self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts)
            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts
        return data_dict


class PointHeadTemplate(nn.Module):

    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None))
        else:
            self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([nn.Linear(c_in, fc_cfg[k], bias=False), nn.BatchNorm1d(fc_cfg[k]), nn.ReLU()])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None, ret_box_labels=False, ret_part_labels=False, set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, 'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = bs_idx == k
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            box_fg_flag = box_idxs_of_pts >= 0
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = (box_centers - points_single).norm(dim=1) < central_radius
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag], gt_classes=gt_box_of_fg_points[:, -1].long())
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single
            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = transformed_points / gt_box_of_fg_points[:, 3:6] + offset
                point_part_labels[bs_mask] = point_part_labels_single
        targets_dict = {'point_cls_labels': point_cls_labels, 'point_box_labels': point_box_labels, 'point_part_labels': point_part_labels}
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)
        positives = point_cls_labels > 0
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_cls': point_loss_cls.item(), 'point_pos_num': pos_normalizer.item()})
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']
        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        point_loss_box_src = self.reg_loss_func(point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...])
        point_loss_box = point_loss_box_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)
        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class PointIntraPartOffsetHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(fc_cfg=self.model_cfg.CLS_FC, input_channels=input_channels, output_channels=num_class)
        self.part_reg_layers = self.make_fc_layers(fc_cfg=self.model_cfg.PART_FC, input_channels=input_channels, output_channels=3)
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(**target_cfg.BOX_CODER_CONFIG)
            self.box_layers = self.make_fc_layers(fc_cfg=self.model_cfg.REG_FC, input_channels=input_channels, output_channels=self.box_coder.code_size)
        else:
            self.box_layers = None

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes, set_ignore_flag=True, use_ball_constraint=False, ret_part_labels=True, ret_box_labels=self.box_layers is not None)
        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_part
        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)
        point_part_preds = self.part_reg_layers(point_features)
        ret_dict = {'point_cls_preds': point_cls_preds, 'point_part_preds': point_part_preds}
        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features)
            ret_dict['point_box_preds'] = point_box_preds
        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_part_offset = torch.sigmoid(point_part_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')
        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(points=batch_dict['point_coords'][:, 1:4], point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds'])
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False
        self.forward_ret_dict = ret_dict
        return batch_dict


class SeparateHead_Transfusion(nn.Module):

    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']
            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=use_bias), nn.BatchNorm1d(head_channels), nn.ReLU()))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, 'bias') and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


def height_overlaps(boxes1, boxes2):
    """
    Calculate height overlaps of two boxes.
    """
    boxes1_top_height = (boxes1[:, 2] + boxes1[:, 5]).view(-1, 1)
    boxes1_bottom_height = boxes1[:, 2].view(-1, 1)
    boxes2_top_height = (boxes2[:, 2] + boxes2[:, 5]).view(1, -1)
    boxes2_bottom_height = boxes2[:, 2].view(1, -1)
    heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
    lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
    overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
    return overlaps_h


def overlaps(boxes1, boxes2):
    """
    Calculate 3D overlaps of two boxes.
    """
    rows = len(boxes1)
    cols = len(boxes2)
    if rows * cols == 0:
        return boxes1.new(rows, cols)
    overlaps_h = height_overlaps(boxes1, boxes2)
    boxes1_bev = boxes1[:, :7]
    boxes2_bev = boxes2[:, :7]
    overlaps_bev = boxes1_bev.new_zeros((boxes1_bev.shape[0], boxes2_bev.shape[0]))
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous(), boxes2_bev.contiguous(), overlaps_bev)
    overlaps_3d = overlaps_bev * overlaps_h
    volume1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]).view(-1, 1)
    volume2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]).view(1, -1)
    iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-08)
    return iou3d


class HungarianAssigner3D:

    def __init__(self, cls_cost, reg_cost, iou_cost):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost

    def focal_loss_cost(self, cls_pred, gt_labels):
        weight = self.cls_cost.get('weight', 0.15)
        alpha = self.cls_cost.get('alpha', 0.25)
        gamma = self.cls_cost.get('gamma', 2.0)
        eps = self.cls_cost.get('eps', 1e-12)
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * weight

    def bevbox_cost(self, bboxes, gt_bboxes, point_cloud_range):
        weight = self.reg_cost.get('weight', 0.25)
        pc_start = bboxes.new(point_cloud_range[0:2])
        pc_range = bboxes.new(point_cloud_range[3:5]) - bboxes.new(point_cloud_range[0:2])
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * weight

    def iou3d_cost(self, bboxes, gt_bboxes):
        iou = overlaps(bboxes, gt_bboxes)
        weight = self.iou_cost.get('weight', 0.25)
        iou_cost = -iou
        return iou_cost * weight, iou

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, max_overlaps, assigned_labels
        cls_cost = self.focal_loss_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.bevbox_cost(bboxes, gt_bboxes, point_cloud_range)
        iou_cost, iou = self.iou3d_cost(bboxes, gt_bboxes)
        cost = cls_cost + reg_cost + iou_cost
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds)
        matched_col_inds = torch.from_numpy(matched_col_inds)
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        return assigned_gt_inds, max_overlaps


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == 'relu':
                return F.relu
            if activation == 'gelu':
                return F.gelu
            if activation == 'glu':
                return F.glu
            raise RuntimeError(f'activation should be relu/gelu, not {activation}.')
        self.activation = _get_activation_fn(activation)
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, key_padding_mask=None, attn_mask=None):
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed), key=self.with_pos_embed(key, key_pos_embed), value=self.with_pos_embed(key, key_pos_embed), key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        query = query.permute(1, 2, 0)
        return query


def clip_sigmoid(x, eps=0.0001):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class TransFusionHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True):
        super(TransFusionHead, self).__init__()
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class
        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')
        hidden_channel = self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE
        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma, alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        self.code_size = 10
        self.shared_conv = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channel, kernel_size=3, padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel, hidden_channel, kernel_size=3, padding=1, bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel, out_channels=num_class, kernel_size=3, padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation, self_posembed=PositionEmbeddingLearned(2, hidden_channel), cross_posembed=PositionEmbeddingLearned(2, hidden_channel))
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)
        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:-padding, padding:-padding] = local_max_inner
        if self.dataset_name == 'nuScenes':
            local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.dataset_name == 'Waymo':
            local_max[:, 1] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
        self.query_labels = top_proposals_class
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])
        query_feat = self.decoder(query_feat, lidar_feat_flatten, query_pos, bev_pos)
        res_layer = self.prediction_head(query_feat)
        res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
        res_layer['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)
        res_layer['dense_heatmap'] = dense_heatmap
        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']
        res = self.predict(feats)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]
            gt_labels_3d = gt_boxes[..., -1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx:batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)
        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        num_proposals = preds_dict['center'].shape[-1]
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None
        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]['pred_boxes']
        gt_bboxes_tensor = gt_bboxes_3d
        assigned_gt_inds, ious = self.bbox_assigner.assign(bboxes_tensor, gt_bboxes_tensor, gt_labels_3d, score, self.point_cloud_range)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]
        bbox_targets = torch.zeros([num_proposals, self.code_size])
        bbox_weights = torch.zeros([num_proposals, self.code_size])
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        if gt_labels_3d is not None:
            labels += self.num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = self.grid_size[:2] // self.feature_map_stride
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):
        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0
        loss_heatmap = self.loss_heatmap(clip_sigmoid(pred_dicts['dense_heatmap']), heatmap).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict['loss_heatmap'] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts['heatmap'].permute(0, 2, 1).reshape(-1, self.num_classes)
        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes + 1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(cls_score, one_hot_targets, label_weights).sum() / max(num_pos, 1)
        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)
        loss_bbox = self.loss_bbox(preds, bbox_targets)
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)
        loss_dict['loss_cls'] = loss_cls.item() * self.loss_cls_weight
        loss_dict['loss_bbox'] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight
        loss_dict[f'matched_ious'] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all
        return loss_all, loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size])
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).float()
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values
        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)
        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)
        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {'pred_boxes': boxes3d, 'pred_scores': scores, 'pred_labels': labels}
            predictions_dicts.append(predictions_dict)
        if filter is False:
            return predictions_dicts
        thresh_mask = final_scores > score_thresh
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]
            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {'pred_boxes': boxes3d, 'pred_scores': scores, 'pred_labels': labels}
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def get_bboxes(self, preds_dicts):
        batch_size = preds_dicts['heatmap'].shape[0]
        batch_score = preds_dicts['heatmap'].sigmoid()
        one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts['query_heatmap_score'] * one_hot
        batch_center = preds_dicts['center']
        batch_height = preds_dicts['height']
        batch_dim = preds_dicts['dim']
        batch_rot = preds_dicts['rot']
        batch_vel = None
        if 'vel' in preds_dicts:
            batch_vel = preds_dicts['vel']
        ret_dict = self.decode_bbox(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1
        return ret_dict


class VoxelNeXtHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range)
        self.voxel_size = torch.Tensor(voxel_size)
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])
        self.iou_branch = self.model_cfg.get('IOU_BRANCH', False)
        if self.iou_branch:
            self.rectifier = self.model_cfg.get('RECTIFIER')
            nms_configs = self.model_cfg.POST_PROCESSING.NMS_CONFIG
            self.nms_configs = [EasyDict(NMS_TYPE=nms_configs.NMS_TYPE, NMS_THRESH=nms_configs.NMS_THRESH[i], NMS_PRE_MAXSIZE=nms_configs.NMS_PRE_MAXSIZE[i], NMS_POST_MAXSIZE=nms_configs.NMS_POST_MAXSIZE[i]) for i in range(num_class)]
        self.double_flip = self.model_cfg.get('DOUBLE_FLIP', False)
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array([self.class_names.index(x) for x in cur_class_names if x in class_names]))
            self.class_id_mapping_each_head.append(cur_class_id_mapping)
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'
        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(SeparateHead(input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128), sep_head_dict=cur_head_dict, kernel_size=kernel_size_head, init_bias=-2.19, use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)))
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())
        if self.iou_branch:
            self.add_module('crit_iou', loss_utils.IouLossSparse())
            self.add_module('crit_iou_reg', loss_utils.IouRegLossSparse())

    def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'heatmap_masks': [], 'gt_boxes': []}
        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                gt_boxes_single_head = []
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])
                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)
                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head, num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx], spatial_shape=spatial_shape, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE, num_max_objs=target_assigner_cfg.NUM_MAX_OBJS, gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP, min_radius=target_assigner_cfg.MIN_RADIUS)
                heatmap_list.append(heatmap)
                target_boxes_list.append(ret_boxes)
                inds_list.append(inds)
                masks_list.append(mask)
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])
            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)
        return ret_dict

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0)) ** 2).sum(-1)
        return distances

    def assign_target_of_single_head(self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1
            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)
            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)
            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=0.0001, max=1 - 0.0001)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']
        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['voxel_indices'][:, 1:]
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            if self.iou_branch:
                batch_box_preds = self._get_predicted_boxes(pred_dict, spatial_indices)
                pred_boxes_for_iou = batch_box_preds.detach()
                iou_loss = self.crit_iou(pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx], pred_boxes_for_iou, target_dicts['gt_boxes'][idx], batch_indices)
                iou_reg_loss = self.crit_iou_reg(batch_box_preds, target_dicts['masks'][idx], target_dicts['inds'][idx], target_dicts['gt_boxes'][idx], batch_indices)
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] if 'iou_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS else self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                iou_reg_loss = iou_reg_loss * iou_weight
                loss += hm_loss + loc_loss + iou_loss + iou_reg_loss
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            else:
                loss += hm_loss + loc_loss
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = pred_dict['center']
        center_z = pred_dict['center_z']
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat(box_part_list, dim=-1)
        return pred_box

    def rotate_class_specific_nms_iou(self, boxes, scores, iou_preds, labels, rectifier, nms_configs):
        """
        :param boxes: (N, 5) [x, y, z, l, w, h, theta]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert isinstance(rectifier, list)
        box_preds_list, scores_list, labels_list = [], [], []
        for cls in range(self.num_class):
            mask = labels == cls
            boxes_cls = boxes[mask]
            scores_cls = torch.pow(scores[mask], 1 - rectifier[cls]) * torch.pow(iou_preds[mask].squeeze(-1), rectifier[cls])
            labels_cls = labels[mask]
            selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=scores_cls, box_preds=boxes_cls, nms_config=nms_configs[cls], score_thresh=None)
            box_preds_list.append(boxes_cls[selected])
            scores_list.append(scores_cls[selected])
            labels_list.append(labels_cls[selected])
        return torch.cat(box_preds_list, dim=0), torch.cat(scores_list, dim=0), torch.cat(labels_list, dim=0)

    def merge_double_flip(self, pred_dict, batch_size, voxel_indices, spatial_shape):
        pred_dict['hm'] = pred_dict['hm'].sigmoid()
        pred_dict['dim'] = pred_dict['dim'].exp()
        batch_indices = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]
        pred_dict_ = {k: [] for k in pred_dict.keys()}
        counts = []
        spatial_indices_ = []
        for bs_idx in range(batch_size):
            spatial_indices_batch = []
            pred_dict_batch = {k: [] for k in pred_dict.keys()}
            for i in range(4):
                bs_indices = batch_indices == bs_idx * 4 + i
                if i in [1, 3]:
                    spatial_indices[bs_indices, 0] = spatial_shape[0] - spatial_indices[bs_indices, 0]
                if i in [2, 3]:
                    spatial_indices[bs_indices, 1] = spatial_shape[1] - spatial_indices[bs_indices, 1]
                if i == 1:
                    pred_dict['center'][bs_indices, 1] = -pred_dict['center'][bs_indices, 1]
                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['vel'][bs_indices, 1] *= -1
                if i == 2:
                    pred_dict['center'][bs_indices, 0] = -pred_dict['center'][bs_indices, 0]
                    pred_dict['rot'][bs_indices, 0] *= -1
                    pred_dict['vel'][bs_indices, 0] *= -1
                if i == 3:
                    pred_dict['center'][bs_indices, 0] = -pred_dict['center'][bs_indices, 0]
                    pred_dict['center'][bs_indices, 1] = -pred_dict['center'][bs_indices, 1]
                    pred_dict['rot'][bs_indices, 1] *= -1
                    pred_dict['rot'][bs_indices, 0] *= -1
                    pred_dict['vel'][bs_indices] *= -1
                spatial_indices_batch.append(spatial_indices[bs_indices])
                for k in pred_dict.keys():
                    pred_dict_batch[k].append(pred_dict[k][bs_indices])
            spatial_indices_batch = torch.cat(spatial_indices_batch)
            spatial_indices_unique, _inv, count = torch.unique(spatial_indices_batch, dim=0, return_inverse=True, return_counts=True)
            spatial_indices_.append(spatial_indices_unique)
            counts.append(count)
            for k in pred_dict.keys():
                pred_dict_batch[k] = torch.cat(pred_dict_batch[k])
                features_unique = pred_dict_batch[k].new_zeros((spatial_indices_unique.shape[0], pred_dict_batch[k].shape[1]))
                features_unique.index_add_(0, _inv, pred_dict_batch[k])
                pred_dict_[k].append(features_unique)
        for k in pred_dict.keys():
            pred_dict_[k] = torch.cat(pred_dict_[k])
        counts = torch.cat(counts).unsqueeze(-1).float()
        voxel_indices_ = torch.cat([torch.cat([torch.full((indices.shape[0], 1), i, device=indices.device, dtype=indices.dtype), indices], dim=1) for i, indices in enumerate(spatial_indices_)])
        batch_hm = pred_dict_['hm']
        batch_center = pred_dict_['center']
        batch_center_z = pred_dict_['center_z']
        batch_dim = pred_dict_['dim']
        batch_rot_cos = pred_dict_['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict_['rot'][:, 1].unsqueeze(dim=1)
        batch_vel = pred_dict_['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
        batch_hm /= counts
        batch_center /= counts
        batch_center_z /= counts
        batch_dim /= counts
        batch_rot_cos /= counts
        batch_rot_sin /= counts
        if not batch_vel is None:
            batch_vel /= counts
        return batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, None, voxel_indices_

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).float()
        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': [], 'pred_ious': []} for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            if self.double_flip:
                batch_hm, batch_center, batch_center_z, batch_dim, batch_rot_cos, batch_rot_sin, batch_vel, batch_iou, voxel_indices_ = self.merge_double_flip(pred_dict, batch_size, voxel_indices.clone(), spatial_shape)
            else:
                batch_hm = pred_dict['hm'].sigmoid()
                batch_center = pred_dict['center']
                batch_center_z = pred_dict['center_z']
                batch_dim = pred_dict['dim'].exp()
                batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
                batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
                batch_iou = (pred_dict['iou'] + 1) * 0.5 if self.iou_branch else None
                batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
                voxel_indices_ = voxel_indices
            final_pred_dicts = centernet_utils.decode_bbox_from_voxels_nuscenes(batch_size=batch_size, indices=voxel_indices_, obj=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin, center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, feature_map_stride=self.feature_map_stride, K=post_process_cfg.MAX_OBJ_PER_SAMPLE, score_thresh=post_process_cfg.SCORE_THRESH, post_center_limit_range=post_center_limit_range)
            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if not self.iou_branch:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'], nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None)
                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]
                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
                ret_dict[k]['pred_ious'].append(final_dict['pred_ious'])
        for k in range(batch_size):
            pred_boxes = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            pred_scores = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            pred_labels = torch.cat(ret_dict[k]['pred_labels'], dim=0)
            if self.iou_branch:
                pred_ious = torch.cat(ret_dict[k]['pred_ious'], dim=0)
                pred_boxes, pred_scores, pred_labels = self.rotate_class_specific_nms_iou(pred_boxes, pred_scores, pred_ious, pred_labels, self.rectifier, self.nms_configs)
            ret_dict[k]['pred_boxes'] = pred_boxes
            ret_dict[k]['pred_scores'] = pred_scores
            ret_dict[k]['pred_labels'] = pred_labels + 1
        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_dicts[0]['pred_boxes']
        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])
            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())
        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        self.forward_ret_dict['batch_index'] = batch_index
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        if self.training:
            target_dict = self.assign_targets(data_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape)
            self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['voxel_indices'] = voxel_indices
        if not self.training or self.predict_boxes_when_training:
            if self.double_flip:
                data_dict['batch_size'] = data_dict['batch_size'] // 4
            pred_dicts = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts, voxel_indices, spatial_shape)
            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts
        return data_dict


def find_all_spconv_keys(model: 'nn.Module', prefix='') ->Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: 'Set[str]' = set()
    for name, child in model.named_children():
        new_prefix = f'{prefix}.{name}' if prefix != '' else name
        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f'{new_prefix}.weight'
            found_keys.add(new_prefix)
        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))
    return found_keys


class Detector3DTemplate(nn.Module):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.module_topology = ['vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'backbone_2d', 'dense_head', 'point_head', 'roi_head']

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {'module_list': [], 'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, 'num_point_features': self.dataset.point_feature_encoder.num_point_features, 'grid_size': self.dataset.grid_size, 'point_cloud_range': self.dataset.point_cloud_range, 'voxel_size': self.dataset.voxel_size, 'depth_downsample_factor': self.dataset.depth_downsample_factor}
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(model_info_dict=model_info_dict)
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](model_cfg=self.model_cfg.VFE, num_point_features=model_info_dict['num_rawpoint_features'], point_cloud_range=model_info_dict['point_cloud_range'], voxel_size=model_info_dict['voxel_size'], grid_size=model_info_dict['grid_size'], depth_downsample_factor=model_info_dict['depth_downsample_factor'])
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](model_cfg=self.model_cfg.BACKBONE_3D, input_channels=model_info_dict['num_point_features'], grid_size=model_info_dict['grid_size'], voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'])
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict
        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](model_cfg=self.model_cfg.MAP_TO_BEV, grid_size=model_info_dict['grid_size'])
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](model_cfg=self.model_cfg.BACKBONE_2D, input_channels=model_info_dict.get('num_bev_features', None))
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict
        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](model_cfg=self.model_cfg.PFE, voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'], num_bev_features=model_info_dict['num_bev_features'], num_rawpoint_features=model_info_dict['num_rawpoint_features'])
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](model_cfg=self.model_cfg.DENSE_HEAD, input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES, num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, class_names=self.class_names, grid_size=model_info_dict['grid_size'], point_cloud_range=model_info_dict['point_cloud_range'], predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False), voxel_size=model_info_dict.get('voxel_size', False))
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict
        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']
        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](model_cfg=self.model_cfg.POINT_HEAD, input_channels=num_point_features, num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1, predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False))
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](model_cfg=self.model_cfg.ROI_HEAD, input_channels=model_info_dict['num_point_features'], backbone_channels=model_info_dict.get('backbone_channels', None), point_cloud_range=model_info_dict['point_cloud_range'], voxel_size=model_info_dict['voxel_size'], num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1)
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=cls_preds, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % str(cur_thresh)] = 0
                recall_dict['rcnn_%s' % str(cur_thresh)] = 0
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()
        spconv_keys = find_all_spconv_keys(self)
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                val_native = val.transpose(-1, -2)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
        version = checkpoint.get('version', None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self._load_state_dict(checkpoint['model_state'], strict=True)
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        if 'version' in checkpoint:
            None
        logger.info('==> Done')
        return it, epoch


class MPPNet(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['proposals_list'] = batch_dict['roi_boxes']
        for cur_module in self.module_list[:]:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rcnn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                try:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                except:
                    record_dict = {'pred_boxes': torch.tensor([]), 'pred_scores': torch.tensor([]), 'pred_labels': torch.tensor([])}
                    pred_dicts.append(record_dict)
                    continue
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=cls_preds, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                if post_process_cfg.get('NOT_APPLY_NMS_FOR_VEL', False):
                    pedcyc_mask = final_labels != 1
                    final_scores_pedcyc = final_scores[pedcyc_mask]
                    final_labels_pedcyc = final_labels[pedcyc_mask]
                    final_boxes_pedcyc = final_boxes[pedcyc_mask]
                    car_mask = (label_preds == 1) & (cls_preds > post_process_cfg.SCORE_THRESH)
                    final_scores_car = cls_preds[car_mask]
                    final_labels_car = label_preds[car_mask]
                    final_boxes_car = box_preds[car_mask]
                    final_scores = torch.cat([final_scores_car, final_scores_pedcyc], 0)
                    final_labels = torch.cat([final_labels_car, final_labels_pedcyc], 0)
                    final_boxes = torch.cat([final_boxes_car, final_boxes_pedcyc], 0)
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes[:, :7], 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict


class MPPNetE2E(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_topology = ['vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d', 'dense_head', 'roi_head']
        self.num_frames = self.model_cfg.ROI_HEAD.Transformer.num_frames

    def reset_memorybank(self):
        self.memory_rois = None
        self.memory_labels = None
        self.memory_scores = None
        self.memory_feature = None

    def forward(self, batch_dict):
        if batch_dict['sample_idx'][0] == 0:
            self.reset_memorybank()
            batch_dict['memory_bank'] = {}
        else:
            batch_dict['memory_bank'] = {'feature_bank': self.memory_feature}
        if self.num_frames == 16:
            batch_dict['points_backup'] = batch_dict['points'].clone()
            time_mask = batch_dict['points'][:, -1] < 0.31
            batch_dict['points'] = batch_dict['points'][time_mask]
        for idx, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            if self.module_topology[idx] == 'dense_head':
                if self.memory_rois is None:
                    self.memory_rois = [batch_dict['rois']] * self.num_frames
                    self.memory_labels = [batch_dict['roi_labels'][:, :, None]] * self.num_frames
                    self.memory_scores = [batch_dict['roi_scores'][:, :, None]] * self.num_frames
                else:
                    self.memory_rois.pop()
                    self.memory_rois.insert(0, batch_dict['rois'])
                    self.memory_labels.pop()
                    self.memory_labels.insert(0, batch_dict['roi_labels'][:, :, None])
                    self.memory_scores.pop()
                    self.memory_scores.insert(0, batch_dict['roi_scores'][:, :, None])
                batch_dict['memory_bank'].update({'rois': self.memory_rois, 'roi_labels': self.memory_labels, 'roi_scores': self.memory_scores})
            if self.module_topology[idx] == 'roi_head':
                if self.memory_feature is None:
                    self.memory_feature = [batch_dict['geometory_feature_memory'][:, :64]] * self.num_frames
                else:
                    self.memory_feature.pop()
                    self.memory_feature.insert(0, batch_dict['geometory_feature_memory'][:, :64])
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                try:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                except:
                    record_dict = {'pred_boxes': torch.tensor([]), 'pred_scores': torch.tensor([]), 'pred_labels': torch.tensor([])}
                    pred_dicts.append(record_dict)
                    continue
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=cls_preds, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                if post_process_cfg.get('NOT_APPLY_NMS_FOR_VEL', False):
                    pedcyc_mask = final_labels != 1
                    final_scores_pedcyc = final_scores[pedcyc_mask]
                    final_labels_pedcyc = final_labels[pedcyc_mask]
                    final_boxes_pedcyc = final_boxes[pedcyc_mask]
                    car_mask = (label_preds == 1) & (cls_preds > post_process_cfg.SCORE_THRESH)
                    final_scores_car = cls_preds[car_mask]
                    final_labels_car = label_preds[car_mask]
                    final_boxes_car = box_preds[car_mask]
                    final_scores = torch.cat([final_scores_car, final_scores_pedcyc], 0)
                    final_labels = torch.cat([final_labels_car, final_labels_pedcyc], 0)
                    final_boxes = torch.cat([final_boxes_car, final_boxes_pedcyc], 0)
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes[:, :7], 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config)
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


class SECONDNetIoU(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    @staticmethod
    def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
        """
        Args:
            cls_scores: (N)
            iou_scores: (N)
            num_points_in_gt: (N, 7+c)
            cls_thresh: scalar
            iou_thresh: scalar
        """
        assert iou_thresh >= cls_thresh
        alpha = torch.zeros(cls_scores.shape, dtype=torch.float32)
        alpha[num_points_in_gt <= cls_thresh] = 0
        alpha[num_points_in_gt >= iou_thresh] = 1
        mask = (num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh)
        alpha[mask] = (num_points_in_gt[mask] - 10) / (iou_thresh - cls_thresh)
        scores = (1 - alpha) * cls_scores + alpha * iou_scores
        return scores

    def set_nms_score_by_class(self, iou_preds, cls_preds, label_preds, score_by_class):
        n_classes = torch.unique(label_preds).shape[0]
        nms_scores = torch.zeros(iou_preds.shape, dtype=torch.float32)
        for i in range(n_classes):
            mask = label_preds == i + 1
            class_name = self.class_names[i]
            score_type = score_by_class[class_name]
            if score_type == 'iou':
                nms_scores[mask] = iou_preds[mask]
            elif score_type == 'cls':
                nms_scores[mask] = cls_preds[mask]
            else:
                raise NotImplementedError
        return nms_scores

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]
            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]
            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1
                if post_process_cfg.NMS_CONFIG.get('SCORE_BY_CLASS', None) and post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'score_by_class':
                    nms_scores = self.set_nms_score_by_class(iou_preds, cls_preds, label_preds, post_process_cfg.NMS_CONFIG.SCORE_BY_CLASS)
                elif post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'weighted_iou_cls':
                    nms_scores = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou * iou_preds + post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls * cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'num_pts_iou_cls':
                    point_mask = batch_dict['points'][:, 0] == batch_mask
                    batch_points = batch_dict['points'][point_mask][:, 1:4]
                    num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(batch_points.cpu(), box_preds[:, 0:7].cpu()).sum(dim=1).float()
                    score_thresh_cfg = post_process_cfg.NMS_CONFIG.SCORE_THRESH
                    nms_scores = self.cal_scores_by_npoints(cls_preds, iou_preds, num_pts_in_gt, score_thresh_cfg.cls, score_thresh_cfg.iou)
                else:
                    raise NotImplementedError
                selected, selected_scores = class_agnostic_nms(box_scores=nms_scores, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels, 'pred_cls_scores': cls_preds[selected], 'pred_iou_scores': iou_preds[selected]}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict


class PointNetfeat(nn.Module):

    def __init__(self, input_dim, x=1, outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel == 256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(input_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(256 * x)
        self.bn4 = nn.BatchNorm1d(self.output_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x))
        x = torch.max(x_ori, 2, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        return x, x_ori


class PointNet(nn.Module):

    def __init__(self, input_dim, joint_feat=False, model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT
        times = 1
        self.feat = PointNetfeat(input_dim, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, channels)
        self.pre_bn = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.fc_s1 = nn.Linear(channels * times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_ce1 = nn.Linear(channels * times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(channels * times, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)

    def forward(self, x, feat=None):
        if self.joint_feat:
            if len(feat.shape) > 2:
                feat = torch.max(feat, 2, keepdim=True)[0]
                x = feat.view(-1, self.output_channel)
                x = F.relu(self.bn1(self.fc1(x)))
                feat = F.relu(self.bn2(self.fc2(x)))
            else:
                feat = feat
            feat_traj = None
        else:
            x, feat_traj = self.feat(self.pre_bn(x))
            x = F.relu(self.bn1(self.fc1(x)))
            feat = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)
        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)
        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)
        return torch.cat([centers, sizes, headings], -1), feat, feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SpatialMixerBlock(nn.Module):

    def __init__(self, hidden_dim, grid_size, channels, config=None, dropout=0.0):
        super().__init__()
        self.mixer_x = MLP(input_dim=grid_size, hidden_dim=hidden_dim, output_dim=grid_size, num_layers=3)
        self.mixer_y = MLP(input_dim=grid_size, hidden_dim=hidden_dim, output_dim=grid_size, num_layers=3)
        self.mixer_z = MLP(input_dim=grid_size, hidden_dim=hidden_dim, output_dim=grid_size, num_layers=3)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * channels, channels))
        self.config = config
        self.grid_size = grid_size

    def forward(self, src):
        src_3d = src.permute(1, 2, 0).contiguous().view(src.shape[1], src.shape[2], self.grid_size, self.grid_size, self.grid_size)
        src_3d = src_3d.permute(0, 1, 4, 3, 2).contiguous()
        mixed_x = self.mixer_x(src_3d)
        mixed_x = src_3d + mixed_x
        mixed_x = self.norm_x(mixed_x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        mixed_y = self.mixer_y(mixed_x.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3).contiguous()
        mixed_y = mixed_x + mixed_y
        mixed_y = self.norm_y(mixed_y.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        mixed_z = self.mixer_z(mixed_y.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2).contiguous()
        mixed_z = mixed_y + mixed_z
        mixed_z = self.norm_z(mixed_z.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        src_mixer = mixed_z.view(src.shape[1], src.shape[2], -1).permute(2, 0, 1)
        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        return src_mixer


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src, pos: 'Optional[Tensor]'=None):
        token_list = []
        output = src
        for layer in self.layers:
            output, tokens = layer(output, pos=pos)
            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)
        return output, token_list


class TransformerEncoderLayer(nn.Module):
    count = 0

    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, num_points=None, num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
        self.config = config
        self.num_point = num_points
        self.num_groups = num_groups
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if self.layer_count <= self.config.enc_layers - 1:
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(self.num_groups):
                self.cross_attn_layers.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))
            self.ffn = FFN(d_model, dim_feedforward)
            self.fusion_all_groups = MLP(input_dim=d_model * 4, hidden_dim=d_model, output_dim=d_model, num_layers=4)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.mlp_mixer_3d = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim, self.config.use_mlp_mixer.get('grid_size', 4), self.config.hidden_dim, self.config.use_mlp_mixer)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos: 'Optional[Tensor]'=None):
        src_intra_group_fusion = self.mlp_mixer_3d(src[1:])
        src = torch.cat([src[:1], src_intra_group_fusion], 0)
        token = src[:1]
        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion
        src_summary = self.self_attn(token, key, value=src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = torch.cat([token, src[1:]], 0)
        if self.layer_count <= self.config.enc_layers - 1:
            src_all_groups = src[1:].view((src.shape[0] - 1) * 4, -1, src.shape[-1])
            src_groups_list = src_all_groups.chunk(self.num_groups, 0)
            src_all_groups = torch.cat(src_groups_list, -1)
            src_all_groups_fusion = self.fusion_all_groups(src_all_groups)
            key = self.with_pos_embed(src_all_groups_fusion, pos[1:])
            query_list = [self.with_pos_embed(query, pos[1:]) for query in src_groups_list]
            inter_group_fusion_list = []
            for i in range(self.num_groups):
                inter_group_fusion = self.cross_attn_layers[i](query_list[i], key, value=src_all_groups_fusion)[0]
                inter_group_fusion = self.ffn(src_groups_list[i], inter_group_fusion)
                inter_group_fusion_list.append(inter_group_fusion)
            src_inter_group_fusion = torch.cat(inter_group_fusion_list, 1)
            src = torch.cat([src[:1], src_inter_group_fusion], 0)
        return src, torch.cat(src[:1].chunk(4, 1), 0)

    def forward_pre(self, src, pos: 'Optional[Tensor]'=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src, pos)


class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, num_lidar_points=None, num_proxy_points=None, share_head=True, num_groups=None, sequence_stride=None, num_frames=None):
        super().__init__()
        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups
        self.num_proxy_points = num_proxy_points
        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_lidar_points, num_groups=num_groups) for i in range(num_encoder_layers)]
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, self.config)
        self.token = nn.Parameter(torch.zeros(self.num_groups, 1, d_model))
        if self.num_frames > 4:
            self.group_length = self.num_frames // self.num_groups
            self.fusion_all_group = MLP(input_dim=self.config.hidden_dim * self.group_length, hidden_dim=self.config.hidden_dim, output_dim=self.config.hidden_dim, num_layers=4)
            self.fusion_norm = FFN(d_model, dim_feedforward)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos=None):
        BS, N, C = src.shape
        if not pos is None:
            pos = pos.permute(1, 0, 2)
        if self.num_frames == 16:
            token_list = [self.token[i:i + 1].repeat(BS, 1, 1) for i in range(self.num_groups)]
            if self.sequence_stride == 1:
                src_groups = src.view(src.shape[0], src.shape[1] // self.num_groups, -1).chunk(4, dim=1)
            elif self.sequence_stride == 4:
                src_groups = []
                for i in range(self.num_groups):
                    groups = []
                    for j in range(self.group_length):
                        points_index_start = (i + j * self.sequence_stride) * self.num_proxy_points
                        points_index_end = points_index_start + self.num_proxy_points
                        groups.append(src[:, points_index_start:points_index_end])
                    groups = torch.cat(groups, -1)
                    src_groups.append(groups)
            else:
                raise NotImplementedError
            src_merge = torch.cat(src_groups, 1)
            src = self.fusion_norm(src[:, :self.num_groups * self.num_proxy_points], self.fusion_all_group(src_merge))
            src = [torch.cat([token_list[i], src[:, i * self.num_proxy_points:(i + 1) * self.num_proxy_points]], dim=1) for i in range(self.num_groups)]
            src = torch.cat(src, dim=0)
        else:
            token_list = [self.token[i:i + 1].repeat(BS, 1, 1) for i in range(self.num_groups)]
            src = [torch.cat([token_list[i], src[:, i * self.num_proxy_points:(i + 1) * self.num_proxy_points]], dim=1) for i in range(self.num_groups)]
            src = torch.cat(src, dim=0)
        src = src.permute(1, 0, 2)
        memory, tokens = self.encoder(src, pos=pos)
        memory = torch.cat(memory[0:1].chunk(4, dim=1), 0)
        return memory, tokens


class ProposalTargetLayer(nn.Module):

    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(batch_dict=batch_dict)
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)
            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious, 'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels, 'reg_valid_mask': reg_valid_mask, 'rcnn_cls_labels': batch_cls_labels}
        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
        
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(rois=cur_roi, roi_labels=cur_roi_labels, gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long())
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
        fg_inds = (max_overlaps >= fg_thresh).nonzero().view(-1)
        easy_bg_inds = (max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) & (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO)
        elif fg_num_rois > 0 and bg_num_rois == 0:
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0]
        elif bg_num_rois > 0 and fg_num_rois == 0:
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO)
        else:
            None
            None
            raise NotImplementedError
        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]
            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError
        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:
            
        Returns:
        
        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = roi_labels == k
            gt_mask = gt_labels == k
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:, :7], cur_gt[:, :7])
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
        return max_overlaps, gt_assignment


class RoIHeadTemplate(nn.Module):

    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(**self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {}))
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module('reg_loss_func', loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']))

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False), nn.BatchNorm1d(fc_list[k]), nn.ReLU()])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config)
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        rois = targets_dict['rois']
        gt_of_rois = targets_dict['gt_of_rois']
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry
        gt_of_rois = common_utils.rotate_points_along_z(points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)).view(batch_size, -1, gt_of_rois.shape[-1])
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()
        tb_dict = {}
        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor)
            rcnn_loss_reg = self.reg_loss_func(rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0), reg_targets.unsqueeze(dim=0))
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors).view(-1, code_size)
                rcnn_boxes3d = common_utils.rotate_points_along_z(rcnn_boxes3d.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz
                loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']
                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError
        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError
        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)
        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)
        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)
        batch_box_preds = common_utils.rotate_points_along_z(batch_box_preds.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds


class SECONDHead(RoIHeadTemplate):

    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False), nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        self.iou_layers = self.make_fc_layers(input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC)
        self.init_weights(weight_init='xavier')
        if torch.__version__ >= '1.3':
            self.affine_grid = partial(F.affine_grid, align_corners=True)
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.affine_grid = F.affine_grid
            self.grid_sample = F.grid_sample

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)
        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO
        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])
            cosa = torch.cos(angle)
            sina = torch.sin(angle)
            theta = torch.stack(((x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * -sina, (x1 + x2 - width + 1) / (width - 1), (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)), dim=1).view(-1, 2, 3).float()
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = self.affine_grid(theta, torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size)))
            pooled_features = self.grid_sample(spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width), grid)
            pooled_features_list.append(pooled_features)
        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)
        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_features = self.roi_grid_pool(batch_dict)
        batch_size_rcnn = pooled_features.shape[0]
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        if not self.training:
            batch_dict['batch_cls_preds'] = rcnn_iou.view(batch_dict['batch_size'], -1, rcnn_iou.shape[-1])
            batch_dict['batch_box_preds'] = batch_dict['rois']
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_iou'] = rcnn_iou
            self.forward_ret_dict = targets_dict
        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(rcnn_iou_flat, rcnn_iou_labels.float(), reduction='none')
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError
        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)
        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict


class VoxelRCNNHead(RoIHeadTemplate):

    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(query_ranges=LAYER_cfg[src_name].QUERY_RANGES, nsamples=LAYER_cfg[src_name].NSAMPLE, radii=LAYER_cfg[src_name].POOL_RADIUS, mlps=mlps, pool_method=LAYER_cfg[src_name].POOL_METHOD)
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]), nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.CLS_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.CLS_FC[k]
            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.REG_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.REG_FC[k]
            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)
        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(rois, grid_size=self.pool_cfg.GRID_SIZE)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=cur_stride, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            pooled_features = pool_layer(xyz=cur_voxel_xyz.contiguous(), xyz_batch_cnt=cur_voxel_xyz_batch_cnt, new_xyz=roi_grid_xyz.contiguous().view(-1, 3), new_xyz_batch_cnt=roi_grid_batch_cnt, new_coords=cur_roi_grid_coords.contiguous().view(-1, 4), features=cur_sp_tensors.features.contiguous(), voxel2point_indices=v2p_ind_tensor)
            pooled_features = pooled_features.view(-1, self.pool_cfg.GRID_SIZE ** 3, pooled_features.shape[-1])
            pooled_features_list.append(pooled_features)
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)
        global_roi_grid_points = common_utils.rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) - local_roi_size.unsqueeze(dim=1) / 2
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_features = self.roi_grid_pool(batch_dict)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg)
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict
        return batch_dict


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([(row[0] + row[2] / 2.0) for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class DepthLSSTransform(nn.Module):
    """
        This module implements LSS, which lists images into 3D and then splats onto bev features.
        This code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.dtransform = nn.Sequential(nn.Conv2d(1, 8, 1), nn.BatchNorm2d(8), nn.ReLU(True), nn.Conv2d(8, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(True))
        self.depthnet = nn.Sequential(nn.Conv2d(in_channel + 64, in_channel, 3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(True), nn.Conv2d(in_channel, in_channel, 3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(True), nn.Conv2d(in_channel, self.D + self.C, 1))
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True), nn.Conv2d(out_channel, out_channel, 3, stride=downsample, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True), nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True))
        else:
            self.downsample = nn.Identity()

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        camera2lidar_rots = camera2lidar_rots
        camera2lidar_trans = camera2lidar_trans
        intrins = intrins
        post_rots = post_rots
        post_trans = post_trans
        B, N, _ = camera2lidar_trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(points.unsqueeze(-1)).squeeze(-1)
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats
        x = x
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = torch.cat(x.unbind(dim=2), 1)
        return final

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)
        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:self.D + self.C].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        x = batch_dict['image_fpn']
        x = x[0]
        BN, C, H, W = x.size()
        img = x.view(int(BN / 6), 6, C, H, W)
        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        points = batch_dict['points']
        batch_size = BN // 6
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size)
        for b in range(batch_size):
            batch_mask = points[:, 0] == b
            cur_coords = points[batch_mask][:, 1:4]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0))
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-05, 100000.0)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            cur_coords = cur_coords[..., [1, 0]]
            on_img = (cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) & (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0)
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, extra_rots=extra_rots, extra_trans=extra_trans)
        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        x = x.permute(0, 1, 3, 2)
        batch_dict['spatial_features_img'] = x
        return batch_dict


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: 'torch.Tensor', features: 'torch.Tensor'=None, new_xyz=None) ->(torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.farthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method)


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown: 'torch.Tensor', known: 'torch.Tensor', unknow_feats: 'torch.Tensor', known_feats: 'torch.Tensor') ->torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: 'float', nsample: 'int', xyz: 'torch.Tensor', xyz_batch_cnt: 'torch.Tensor', new_xyz: 'torch.Tensor', new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.IntTensor(M, nsample).zero_()
        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = idx[:, 0] == -1
        idx[empty_ball_mask] = 0
        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(empty_ball_mask)
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: 'torch.Tensor', features_batch_cnt: 'torch.Tensor', idx: 'torch.Tensor', idx_batch_cnt: 'torch.Tensor'):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()
        assert features.shape[0] == features_batch_cnt.sum(), 'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), 'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))
        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.FloatTensor(M, C, nsample)
        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)
        ctx.for_backwards = B, N, idx, features_batch_cnt, idx_batch_cnt
        return output

    @staticmethod
    def backward(ctx, grad_out: 'torch.Tensor'):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards
        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.FloatTensor(N, C).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx, idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):

    def __init__(self, radius: 'float', nsample: 'int', use_xyz: 'bool'=True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: 'torch.Tensor', xyz_batch_cnt: 'torch.Tensor', new_xyz: 'torch.Tensor', new_xyz_batch_cnt: 'torch.Tensor', features: 'torch.Tensor'=None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), 'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        grouped_xyz -= new_xyz.unsqueeze(-1)
        grouped_xyz[empty_ball_mask] = 0
        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features, idx


class GroupAll(nn.Module):

    def __init__(self, use_xyz: 'bool'=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: 'torch.Tensor', new_xyz: 'torch.Tensor', features: 'torch.Tensor'=None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):

    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-08)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]
        new_features = self.mlp(new_features)
        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
        return new_features


class VectorPoolLocalInterpolateModule(nn.Module):

    def __init__(self, mlp, num_voxels, max_neighbour_distance, nsample, neighbor_type, use_xyz=True, neighbour_distance_multiplier=1.0, xyz_encoding_type='concat'):
        """
        Args:
            mlp:
            num_voxels:
            max_neighbour_distance:
            neighbor_type: 1: ball, others: cube
            nsample: find all (-1), find limited number(>0)
            use_xyz:
            neighbour_distance_multiplier:
            xyz_encoding_type:
        """
        super().__init__()
        self.num_voxels = num_voxels
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz
        self.xyz_encoding_type = xyz_encoding_type
        if mlp is not None:
            if self.use_xyz:
                mlp[0] += 9 if self.xyz_encoding_type == 'concat' else 0
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None
        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz, support_features, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            support_features: (N1 + N2 ..., C) point-wise features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        with torch.no_grad():
            dist, idx, num_avg_length_of_neighbor_idxs = pointnet2_utils.three_nn_for_vector_pool_by_two_step(support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt, self.max_neighbour_distance, self.nsample, self.neighbor_type, self.num_avg_length_of_neighbor_idxs, self.num_total_grids, self.neighbor_distance_multiplier)
        self.num_avg_length_of_neighbor_idxs = max(self.num_avg_length_of_neighbor_idxs, num_avg_length_of_neighbor_idxs.item())
        dist_recip = 1.0 / (dist + 1e-08)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-08)
        empty_mask = idx.view(-1, 3)[:, 0] == -1
        idx.view(-1, 3)[empty_mask] = 0
        interpolated_feats = pointnet2_utils.three_interpolate(support_features, idx.view(-1, 3), weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(idx.shape[0], idx.shape[1], -1)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(-1, 3, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) - near_known_xyz).view(-1, idx.shape[1], 9)
            if self.xyz_encoding_type == 'concat':
                interpolated_feats = torch.cat((interpolated_feats, local_xyz), dim=-1)
            else:
                raise NotImplementedError
        new_features = interpolated_feats.view(-1, interpolated_feats.shape[-1])
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(1, 0)[None, :, :, None]
            new_features = self.mlp(new_features)
            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
        return new_features


class VectorPoolAggregationModule(nn.Module):

    def __init__(self, input_channels, num_local_voxel=(3, 3, 3), local_aggregation_type='local_interpolation', num_reduced_channels=30, num_channels_of_local_aggregation=32, post_mlps=(128,), max_neighbor_distance=None, neighbor_nsample=-1, neighbor_type=0, neighbor_distance_multiplier=2.0):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.total_voxels = self.num_local_voxel[0] * self.num_local_voxel[1] * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice']
        self.input_channels = input_channels
        self.num_reduced_channels = input_channels if num_reduced_channels is None else num_reduced_channels
        self.num_channels_of_local_aggregation = num_channels_of_local_aggregation
        self.max_neighbour_distance = max_neighbor_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type
        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = VectorPoolLocalInterpolateModule(mlp=None, num_voxels=self.num_local_voxel, max_neighbour_distance=self.max_neighbour_distance, nsample=self.neighbor_nsample, neighbor_type=self.neighbor_type, neighbour_distance_multiplier=neighbor_distance_multiplier)
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels
        num_c_out = self.total_voxels * self.num_channels_of_local_aggregation
        self.separate_local_aggregation_layer = nn.Sequential(nn.Conv1d(num_c_in, num_c_out, kernel_size=1, groups=self.total_voxels, bias=False), nn.BatchNorm1d(num_c_out), nn.ReLU())
        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False), nn.BatchNorm1d(cur_num_c), nn.ReLU()])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list)
        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) ->str:
        ret = f'radius={self.max_neighbour_distance}, local_voxels=({self.num_local_voxel}, local_aggregation_type={self.local_aggregation_type}, num_c_reduction={self.input_channels}->{self.num_reduced_channels}, num_c_local_aggregation={self.num_channels_of_local_aggregation}'
        return ret

    def vector_pool_with_voxel_query(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        use_xyz = 1
        pooling_type = 0 if self.local_aggregation_type == 'voxel_avg_pool' else 1
        new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid = pointnet2_utils.vector_pool_with_voxel_query_op(xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt, self.num_local_voxel[0], self.num_local_voxel[1], self.num_local_voxel[2], self.max_neighbour_distance, self.num_reduced_channels, use_xyz, self.num_mean_points_per_grid, self.neighbor_nsample, self.neighbor_type, pooling_type)
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid, num_mean_points_per_grid.item())
        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1, 3)
        new_features = new_features.view(num_new_pts, -1, self.num_reduced_channels)
        new_features = torch.cat((new_local_xyz, new_features), dim=-1).view(num_new_pts, -1)
        return new_features, point_cnt_of_grid

    @staticmethod
    def get_dense_voxels_by_center(point_centers, max_neighbour_distance, num_voxels):
        """
        Args:
            point_centers: (N, 3)
            max_neighbour_distance: float
            num_voxels: [num_x, num_y, num_z]

        Returns:
            voxel_centers: (N, total_voxels, 3)
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(-R + R / num_voxels[0], R - R / num_voxels[0] + 1e-05, 2 * R / num_voxels[0], device=device)
        y_grids = torch.arange(-R + R / num_voxels[1], R - R / num_voxels[1] + 1e-05, 2 * R / num_voxels[1], device=device)
        z_grids = torch.arange(-R + R / num_voxels[2], R - R / num_voxels[2] + 1e-05, 2 * R / num_voxels[2], device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(x_grids, y_grids, z_grids)
        xyz_offset = torch.cat((x_offset.contiguous().view(-1, 1), y_offset.contiguous().view(-1, 1), z_offset.contiguous().view(-1, 1)), dim=-1)
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            xyz: (N, 3)
            xyz_batch_cnt: (batch_size)
            features: (N, C)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size)
        Returns:
            new_features: (M, total_voxels * C)
        """
        voxel_centers = self.get_dense_voxels_by_center(point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel)
        voxel_features = self.local_interpolate_module.forward(support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt)
        voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, **kwargs):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        N, C = features.shape
        assert C % self.num_reduced_channels == 0, f'the input channels ({C}) should be an integral multiple of num_reduced_channels({self.num_reduced_channels})'
        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)
        if self.local_aggregation_type in ['voxel_avg_pool', 'voxel_random_choice']:
            vector_features, point_cnt_of_grid = self.vector_pool_with_voxel_query(xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
        elif self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
        else:
            raise NotImplementedError
        vector_features = vector_features.permute(1, 0)[None, :, :]
        new_features = self.separate_local_aggregation_layer(vector_features)
        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


class VectorPoolAggregationModuleMSG(nn.Module):

    def __init__(self, input_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_groups = self.model_cfg.NUM_GROUPS
        self.layers = []
        c_in = 0
        for k in range(self.num_groups):
            cur_config = self.model_cfg[f'GROUP_CFG_{k}']
            cur_vector_pool_module = VectorPoolAggregationModule(input_channels=input_channels, num_local_voxel=cur_config.NUM_LOCAL_VOXEL, post_mlps=cur_config.POST_MLPS, max_neighbor_distance=cur_config.MAX_NEIGHBOR_DISTANCE, neighbor_nsample=cur_config.NEIGHBOR_NSAMPLE, local_aggregation_type=self.model_cfg.LOCAL_AGGREGATION_TYPE, num_reduced_channels=self.model_cfg.get('NUM_REDUCED_CHANNELS', None), num_channels_of_local_aggregation=self.model_cfg.NUM_CHANNELS_OF_LOCAL_AGGREGATION, neighbor_distance_multiplier=2.0)
            self.__setattr__(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config.POST_MLPS[-1]
        c_in += 3
        shared_mlps = []
        for cur_num_c in self.model_cfg.MSG_POST_MLPS:
            shared_mlps.extend([nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False), nn.BatchNorm1d(cur_num_c), nn.ReLU()])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def forward(self, **kwargs):
        features_list = []
        for k in range(self.num_groups):
            cur_xyz, cur_features = self.__getattr__(f'layer_{k}')(**kwargs)
            features_list.append(cur_features)
        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return cur_xyz, new_features


class NeighborVoxelSAModuleMSG(nn.Module):

    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(query_ranges) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]
            cur_mlp_in = nn.Sequential(nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False), nn.BatchNorm1d(mlp_spec[1]))
            cur_mlp_pos = nn.Sequential(nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[1]))
            cur_mlp_out = nn.Sequential(nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False), nn.BatchNorm1d(mlp_spec[2]), nn.ReLU())
            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)
        self.relu = nn.ReLU()
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, new_coords, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_features_list = []
        for k in range(len(self.groupers)):
            features_in = features.permute(1, 0).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)
            features_in = features_in.permute(0, 2, 1).contiguous()
            features_in = features_in.view(-1, features_in.shape[-1])
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices)
            grouped_features[empty_ball_mask] = 0
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            position_features = self.mlps_pos[k](grouped_xyz)
            new_features = grouped_features + position_features
            new_features = self.relu(new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_features


class VoxelQuery(Function):

    @staticmethod
    def forward(ctx, max_range: 'int', radius: 'float', nsample: 'int', xyz: 'torch.Tensor', new_xyz: 'torch.Tensor', new_coords: 'torch.Tensor', point_indices: 'torch.Tensor'):
        """
        Args:
            ctx:
            max_range: int, max range of voxels to be grouped
            nsample: int, maximum number of features in the balls
            new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert new_coords.is_contiguous()
        assert point_indices.is_contiguous()
        M = new_coords.shape[0]
        B, Z, Y, X = point_indices.shape
        idx = torch.IntTensor(M, nsample).zero_()
        z_range, y_range, x_range = max_range
        pointnet2.voxel_query_wrapper(M, Z, Y, X, nsample, radius, z_range, y_range, x_range, new_xyz, xyz, new_coords, point_indices, idx)
        empty_ball_mask = idx[:, 0] == -1
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


voxel_query = VoxelQuery.apply


class VoxelQueryAndGrouping(nn.Module):

    def __init__(self, max_range: 'int', radius: 'float', nsample: 'int'):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample

    def forward(self, new_coords: 'torch.Tensor', xyz: 'torch.Tensor', xyz_batch_cnt: 'torch.Tensor', new_xyz: 'torch.Tensor', new_xyz_batch_cnt: 'torch.Tensor', features: 'torch.Tensor', voxel2point_indices: 'torch.Tensor'):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), 'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0]
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)
        idx1 = idx1.view(batch_size, -1, self.nsample)
        count = 0
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = idx1.view(-1, self.nsample)
        idx1[empty_ball_mask1] = 0
        idx = idx1
        empty_ball_mask = empty_ball_mask1
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        return grouped_features, grouped_xyz, empty_ball_mask


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            ctx:
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7)
            max_pts_each_voxel:
            pool_method: 'max' or 'avg'

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)
        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)
        ctx.roiaware_pool3d_for_backward = pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
        roipoint_pool3d_cuda.forward(points.contiguous(), pooled_boxes3d.contiguous(), point_features.contiguous(), pooled_features, pooled_empty_flag)
        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class RoIPointPool3d(nn.Module):

    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points)


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: 'float'=2.0, alpha: 'float'=0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: 'torch.Tensor', target: 'torch.Tensor'):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss
        if weights.shape.__len__() == 2 or weights.shape.__len__() == 1 and target.shape.__len__() == 2:
            weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()
        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: 'float'=1.0 / 9.0, code_weights: 'list'=None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-05:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)
        loss = self.smooth_l1_loss(diff, self.beta)
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask
    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)
    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    loss = loss / torch.clamp_min(num, min=1.0)
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


def neg_loss_sparse(pred, gt):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x n)
        gt: (batch x c x n)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(FocalLossSparse, self).__init__()
        self.neg_loss = neg_loss_sparse

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossSparse, self).__init__()

    def forward(self, output, mask, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)
        loss = _reg_loss(pred, target, mask)
        return loss


class IouLossSparse(nn.Module):
    """IouLoss loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(IouLossSparse, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt, batch_index):
        if mask.sum() == 0:
            return iou_pred.new_zeros(1)
        batch_size = mask.shape[0]
        mask = mask.bool()
        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            pred = iou_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            pred_box = box_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(pred_box, box_gt[bs_idx])
            target = 2 * target - 1
            loss += F.l1_loss(pred, target, reduction='sum')
        loss = loss / (mask.sum() + 0.0001)
        return loss


class IouRegLossSparse(nn.Module):
    """Distance IoU loss for output boxes
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
    """

    def __init__(self, type='DIoU'):
        super(IouRegLossSparse, self).__init__()

    def center_to_corner2d(self, center, dim):
        corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=torch.float32, device=dim.device)
        corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
        corners = corners + center.view(-1, 1, 2)
        return corners

    def bbox3d_iou_func(self, pred_boxes, gt_boxes):
        assert pred_boxes.shape[0] == gt_boxes.shape[0]
        qcorners = self.center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
        gcorners = self.center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])
        inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
        inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
        out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
        out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])
        volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
        volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]
        inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
        inter_h = torch.clamp(inter_h, min=0)
        inter = torch.clamp(inter_max_xy - inter_min_xy, min=0)
        volume_inter = inter[:, 0] * inter[:, 1] * inter_h
        volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter
        inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)
        outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
        outer_h = torch.clamp(outer_h, min=0)
        outer = torch.clamp(out_max_xy - out_min_xy, min=0)
        outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2
        dious = volume_inter / volume_union - inter_diag / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)
        return dious

    def forward(self, box_pred, mask, ind, box_gt, batch_index):
        if mask.sum() == 0:
            return box_pred.new_zeros(1)
        mask = mask.bool()
        batch_size = mask.shape[0]
        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            pred_box = box_pred[batch_inds][ind[bs_idx]]
            iou = self.bbox3d_iou_func(pred_box[mask[bs_idx]], box_gt[bs_idx])
            loss += (1.0 - iou).sum()
        loss = loss / (mask.sum() + 0.0001)
        return loss


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0
        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights
        return pos_loss + neg_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptivePadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock1D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BasicBlock2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalLossCenterNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FocalLossSparse,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GaussianFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (PointNetfeat,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionEmbeddingLearned,
     lambda: ([], {'input_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SigmoidFocalClassificationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Stage_Reduction_Block,
     lambda: ([], {'input_channel': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (WeightedCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
]

