
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


import warnings


from copy import deepcopy


from typing import Optional


from typing import Sequence


from typing import Union


import numpy as np


import torch.nn as nn


import logging


from typing import Dict


from typing import List


from typing import Tuple


from torch.nn.modules.conv import Conv2d


from torch.nn.modules.conv import Conv1d


from torch.optim.adamw import AdamW


from torch.optim.sgd import SGD


from torch.optim.adam import Adam


from torch.optim import AdamW


import copy


from typing import Callable


from typing import Set


from numpy import dtype


import random


import math


from abc import ABCMeta


from torch import Tensor


from torch import nn as nn


from torch import nn


from functools import partial


from numbers import Number


from torch.nn import functional as F


import torch.nn.functional as F


from typing import Any


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from abc import abstractmethod


from typing import OrderedDict


from torch import distributed as dist


from torch.autograd.function import Function


import itertools


from torch.nn.parameter import Parameter


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.nn.functional import smooth_l1_loss


from numpy import ndarray


from typing import Iterator


from enum import IntEnum


from enum import unique


from logging import warning


from collections.abc import Sized


import functools


from inspect import getfullargspec


from typing import Type


from torch import multiprocessing as mp


import time


import matplotlib.pyplot as plt


from matplotlib.collections import PatchCollection


from matplotlib.patches import PathPatch


from matplotlib.path import Path


from collections import OrderedDict


import torch.distributed as dist


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn.modules.batchnorm import _BatchNorm


from torch.autograd.function import once_differentiable


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torch import einsum


from math import ceil


import torch.utils.checkpoint as cp


from torch.nn.init import normal_


class _Voxelization(Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
            voxel_num = hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3, deterministic)
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class VoxelizationByGridShape(nn.Module):
    """Voxelization that allows inferring voxel size automatically based on
    grid shape.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        point_cloud_range (list):
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points (int): max number of points per voxel
        voxel_size (list): list [x, y, z] or [rho, phi, z]
            size of single voxel.
        grid_shape (list): [L, W, H], grid shape of voxelization.
        max_voxels (tuple or int): max number of voxels in
            (training, testing) time
        deterministic: bool. whether to invoke the non-deterministic
            version of hard-voxelization implementations. non-deterministic
            version is considerablly fast but is not deterministic. only
            affects hard voxelization. default True. for more information
            of this argument and the implementation insights, please refer
            to the following links:
            https://github.com/open-mmlab/mmdetection3d/issues/894
            https://github.com/open-mmlab/mmdetection3d/pull/904
            it is an experimental feature and we will appreciate it if
            you could share with us the failing cases.
    """

    def __init__(self, point_cloud_range: 'List', max_num_points: 'int', voxel_size: 'List'=[], grid_shape: 'List[int]'=[], max_voxels: 'Union[tuple, int]'=20000, deterministic: 'bool'=True):
        super().__init__()
        if voxel_size and grid_shape:
            raise ValueError('voxel_size is mutually exclusive grid_shape')
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        return voxelization(input, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', grid_shape=' + str(self.grid_shape)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s


class _DynamicScatter(Function):
    """Different from the mmcv implementation, here it is allowed to return
    point2voxel_map."""

    @staticmethod
    def forward(ctx: 'Any', feats: 'torch.Tensor', coors: 'torch.Tensor', reduce_type: 'str'='max', return_map: 'str'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats (torch.Tensor): [N, C]. Points features to be reduced
                into voxels.
            coors (torch.Tensor): [N, ndim]. Corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type (str, optional): Reduce op. support 'max', 'sum' and
                'mean'. Default: 'max'.
            return_map (str, optional): Whether to return point2voxel_map.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        results = ext_module.dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        if return_map:
            return voxel_feats, voxel_coors, point2voxel_map
        else:
            return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx: 'Any', grad_voxel_feats: 'torch.Tensor', grad_voxel_coors: 'Optional[torch.Tensor]'=None) ->tuple:
        feats, voxel_feats, point2voxel_map, voxel_points_count = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        ext_module.dynamic_point_to_voxel_backward(grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats, point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter_3d = _DynamicScatter.apply


class DynamicScatter3D(nn.Module):
    """Scatters points into voxels, used in the voxel encoder with dynamic
    voxelization.

    Note:
        The CPU and GPU implementation get the same output, but have numerical
        difference after summation and division (e.g., 5e-7).

    Args:
        voxel_size (list): list [x, y, z] size of three dimension.
        point_cloud_range (list): The coordinate range of points, [x_min,
            y_min, z_min, x_max, y_max, z_max].
        average_points (bool): whether to use avg pooling to scatter points
            into voxel.
    """

    def __init__(self, voxel_size: 'List', point_cloud_range: 'List', average_points: 'bool'):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points: 'torch.Tensor', coors: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter_3d(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points: 'torch.Tensor', coors: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points/features into voxels.

        Args:
            points (torch.Tensor): Points to be reduced into voxels.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(points[inds], coors[inds][:, 1:])
                coor_pad = F.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)
            return features, feature_coors

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', average_points=' + str(self.average_points)
        s += ')'
        return s


class GeneralSamplingModule(nn.Module):
    """Sampling Points.

    Sampling points with given index.
    """

    def forward(self, xyz: 'Tensor', features: 'Tensor', sample_inds: 'Tensor') ->Tuple[Tensor]:
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates of the features.
            features (Tensor): (B, C, N) features to sample.
            sample_inds (Tensor): (B, M) the given index,
                where M is the number of points.

        Returns:
            Tensor: (B, M, 3) coordinates of sampled features
            Tensor: (B, C, M) the sampled features.
            Tensor: (B, M) the given index.
        """
        xyz_t = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_points(xyz_t, sample_inds).transpose(1, 2).contiguous()
        new_features = gather_points(features, sample_inds).contiguous()
        return new_xyz, new_features, sample_inds


class BaseDGCNNGFModule(nn.Module):
    """Base module for point graph feature module used in DGCNN.

    Args:
        radii (List[float]): List of radius in each knn or ball query.
        sample_nums (List[int]): Number of samples in each knn or ball query.
        mlp_channels (List[List[int]]): Specify of the dgcnn before the global
            pooling for each graph feature module.
        knn_modes (List[str]): Type of KNN method, valid mode
            ['F-KNN', 'D-KNN']. Defaults to ['F-KNN'].
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        use_xyz (bool): Whether to use xyz as point features.
            Defaults to True.
        pool_mode (str): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool): If ball query, whether to normalize local XYZ
            with radius. Defaults to False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
    """

    def __init__(self, radii: 'List[float]', sample_nums: 'List[int]', mlp_channels: 'List[List[int]]', knn_modes: 'List[str]'=['F-KNN'], dilated_group: 'bool'=False, use_xyz: 'bool'=True, pool_mode: 'str'='max', normalize_xyz: 'bool'=False, grouper_return_grouped_xyz: 'bool'=False, grouper_return_grouped_idx: 'bool'=False) ->None:
        super(BaseDGCNNGFModule, self).__init__()
        assert len(sample_nums) == len(mlp_channels), 'Num_samples and mlp_channels should have the same length.'
        assert pool_mode in ['max', 'avg'], "Pool_mode should be one of ['max', 'avg']."
        assert isinstance(knn_modes, list) or isinstance(knn_modes, tuple), 'The type of knn_modes should be list or tuple.'
        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels
        self.pool_mode = pool_mode
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.knn_modes = knn_modes
        for i in range(len(sample_nums)):
            sample_num = sample_nums[i]
            if sample_num is not None:
                if self.knn_modes[i] == 'D-KNN':
                    grouper = QueryAndGroup(radii[i], sample_num, use_xyz=use_xyz, normalize_xyz=normalize_xyz, return_grouped_xyz=grouper_return_grouped_xyz, return_grouped_idx=True)
                else:
                    grouper = QueryAndGroup(radii[i], sample_num, use_xyz=use_xyz, normalize_xyz=normalize_xyz, return_grouped_xyz=grouper_return_grouped_xyz, return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _pool_features(self, features: 'Tensor') ->Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (Tensor): (B, C, N, K) Features of locally grouped
                points before pooling.

        Returns:
            Tensor: (B, C, N) Pooled features aggregating local information.
        """
        if self.pool_mode == 'max':
            new_features = F.max_pool2d(features, kernel_size=[1, features.size(3)])
        elif self.pool_mode == 'avg':
            new_features = F.avg_pool2d(features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError
        return new_features.squeeze(-1).contiguous()

    def forward(self, points: 'Tensor') ->Tensor:
        """forward.

        Args:
            points (Tensor): (B, N, C) Input points.

        Returns:
            Tensor: (B, N, C1) New points generated from each graph
            feature module.
        """
        new_points_list = [points]
        for i in range(len(self.groupers)):
            new_points = new_points_list[i]
            new_points_trans = new_points.transpose(1, 2).contiguous()
            if self.knn_modes[i] == 'D-KNN':
                idx = self.groupers[i](new_points[..., -3:].contiguous(), new_points[..., -3:].contiguous())[-1]
                grouped_results = grouping_operation(new_points_trans, idx)
                grouped_results -= new_points_trans.unsqueeze(-1)
            else:
                grouped_results = self.groupers[i](new_points, new_points)
            new_points = new_points_trans.unsqueeze(-1).repeat(1, 1, 1, grouped_results.shape[-1])
            new_points = torch.cat([grouped_results, new_points], dim=1)
            new_points = self.mlps[i](new_points)
            new_points = self._pool_features(new_points)
            new_points = new_points.transpose(1, 2).contiguous()
            new_points_list.append(new_points)
        return new_points


class DGCNNGFModule(BaseDGCNNGFModule):
    """Point graph feature module used in DGCNN.

    Args:
        mlp_channels (List[int]): Specify of the dgcnn before the global
            pooling for each graph feature module.
        num_sample (int, optional): Number of samples in each knn or ball
            query. Defaults to None.
        knn_mode (str): Type of KNN method, valid mode ['F-KNN', 'D-KNN'].
            Defaults to 'F-KNN'.
        radius (float, optional): Radius to group with. Defaults to None.
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN2d').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        use_xyz (bool): Whether to use xyz as point features. Defaults to True.
        pool_mode (str): Type of pooling method. Defaults to 'max'.
        normalize_xyz (bool): If ball query, whether to normalize local XYZ
            with radius. Defaults to False.
        bias (bool or str): If specified as `auto`, it will be decided by
            `norm_cfg`. `bias` will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.
    """

    def __init__(self, mlp_channels: 'List[int]', num_sample: 'Optional[int]'=None, knn_mode: 'str'='F-KNN', radius: 'Optional[float]'=None, dilated_group: 'bool'=False, norm_cfg: 'ConfigType'=dict(type='BN2d'), act_cfg: 'ConfigType'=dict(type='ReLU'), use_xyz: 'bool'=True, pool_mode: 'str'='max', normalize_xyz: 'bool'=False, bias: 'Union[bool, str]'='auto') ->None:
        super(DGCNNGFModule, self).__init__(mlp_channels=[mlp_channels], sample_nums=[num_sample], knn_modes=[knn_mode], radii=[radius], use_xyz=use_xyz, pool_mode=pool_mode, normalize_xyz=normalize_xyz, dilated_group=dilated_group)
        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(f'layer{i}', ConvModule(mlp_channel[i], mlp_channel[i + 1], kernel_size=(1, 1), stride=(1, 1), conv_cfg=dict(type='Conv2d'), norm_cfg=norm_cfg, act_cfg=act_cfg, bias=bias))
            self.mlps.append(mlp)


EPS = 1e-06


class ArrayConverter:
    """Utility class for data-type agnostic processing.

    Args:
        template_array (np.ndarray or torch.Tensor or list or tuple or int or
            float, optional): Template array. Defaults to None.
    """
    SUPPORTED_NON_ARRAY_TYPES = int, float, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64

    def __init__(self, template_array: 'Optional[TemplateArrayType]'=None) ->None:
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array: 'TemplateArrayType') ->None:
        """Set template array.

        Args:
            array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'
        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                None
                raise
        elif isinstance(array, (int, float)):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f'Template type {self.array_type} is not supported.')

    def convert(self, input_array: 'TemplateArrayType', target_type: 'Optional[Type]'=None, target_array: 'Optional[Union[np.ndarray, torch.Tensor]]'=None) ->Union[np.ndarray, torch.Tensor]:
        """Convert input array to target data type.

        Args:
            input_array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Input array.
            target_type (Type, optional): Type to which input array is
                converted. It should be `np.ndarray` or `torch.Tensor`.
                Defaults to None.
            target_array (np.ndarray or torch.Tensor, optional): Template array
                to which input array is converted. Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.

        Returns:
            np.ndarray or torch.Tensor: The converted array.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                None
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, 'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), 'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                converted_array = torch.tensor(input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), 'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array: 'Union[np.ndarray, torch.Tensor]') ->Union[np.ndarray, torch.Tensor, int, float]:
        """Recover input type to original array type.

        Args:
            input_array (np.ndarray or torch.Tensor): Input array.

        Returns:
            np.ndarray or torch.Tensor or int or float: Converted array.
        """
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), 'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array


def array_converter(to_torch: 'bool'=True, apply_to: 'Tuple[str, ...]'=tuple(), template_arg_name_: 'Optional[str]'=None, recover: 'bool'=True) ->Callable:
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy arrays for middle
    calculation, then convert output to original data-type if `recover=True`.

    Args:
        to_torch (bool): Whether to convert to PyTorch tensors for middle
            calculation. Defaults to True.
        apply_to (Tuple[str]): The arguments to which we apply data-type
            conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template
            (return arrays should have the same dtype and device as the
            template). Defaults to None. If None, we will use the first
            argument in `apply_to` as the template argument.
        recover (bool): Whether or not to recover the wrapped function outputs
            to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or when
            apply_to contains an arg which is not among all args, a ValueError
            will be raised. When the template argument or an argument to
            convert is a list or tuple, and cannot be converted to a NumPy
            array, a ValueError will be raised.
        TypeError: When the type of the template argument or an argument to
            convert does not belong to the above range, or the contents of such
            an list-or-tuple-type argument do not share the same data type, a
            TypeError will be raised.

    Returns:
        Callable: Wrapped function.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add(a, b)
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)
            func_name = func.__name__
            arg_spec = getfullargspec(func)
            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)
            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}
            all_arg_names = arg_names + kwonly_arg_names
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_
            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the argument list of function {func_name}')
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not an argument of {func_name}')
            new_args = []
            new_kwargs = {}
            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(converter.convert(input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)
                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value
            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=kwargs[arg_name], target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]
            new_args += nameless_args
            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data, tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data
            if recover:
                return recursive_recover(return_values)
            else:
                return return_values
        return new_func
    return array_converter_wrapper


@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(points: 'Union[np.ndarray, Tensor]', angles: 'Union[np.ndarray, Tensor, float]', axis: 'int'=0, return_mat: 'bool'=False, clockwise: 'bool'=False) ->Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], np.ndarray, Tensor]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]
    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)
    assert len(points.shape) == 3 and len(angles.shape) == 1 and points.shape[0] == angles.shape[0], f'Incorrect shape of points angles: {points.shape}, {angles.shape}'
    assert points.shape[-1] in [2, 3], f'Points size should be 2 or 3 instead of {points.shape[-1]}'
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([torch.stack([rot_cos, zeros, -rot_sin]), torch.stack([zeros, ones, zeros]), torch.stack([rot_sin, zeros, rot_cos])])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([torch.stack([rot_cos, rot_sin, zeros]), torch.stack([-rot_sin, rot_cos, zeros]), torch.stack([zeros, zeros, ones])])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([torch.stack([ones, zeros, zeros]), torch.stack([zeros, rot_cos, rot_sin]), torch.stack([zeros, -rot_sin, rot_cos])])
        else:
            raise ValueError(f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])])
    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)
    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)
    if batch_free:
        points_new = points_new.squeeze(0)
    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


class BasePoints:
    """Base class for Points.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor: 'Union[Tensor, np.ndarray, Sequence[Sequence[float]]]', points_dim: 'int'=3, attribute_dims: 'Optional[dict]'=None) ->None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, f'The points dimension must be 2 and the length of the last dimension must be {points_dim}, but got points with shape {tensor.shape}.'
        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self) ->Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: 'Union[Tensor, np.ndarray]') ->None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self) ->Union[Tensor, None]:
        """Tensor or None: Returns a vector with height of each point in shape
        (N, )."""
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor: 'Union[Tensor, np.ndarray]') ->None:
        """Set the height of each point.

        Args:
            tensor (Tensor or np.ndarray): Height of each point with shape
                (N, ).
        """
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self) ->Union[Tensor, None]:
        """Tensor or None: Returns a vector with color of each point in shape
        (N, 3)."""
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor: 'Union[Tensor, np.ndarray]') ->None:
        """Set the color of each point.

        Args:
            tensor (Tensor or np.ndarray): Color of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self) ->torch.Size:
        """torch.Size: Shape of points."""
        return self.tensor.shape

    def shuffle(self) ->Tensor:
        """Shuffle the points.

        Returns:
            Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation: 'Union[Tensor, np.ndarray, float]', axis: 'Optional[int]'=None) ->Tensor:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.

        Returns:
            Tensor: Rotation matrix.
        """
        if not isinstance(rotation, Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'
        if axis is None:
            axis = self.rotation_axis
        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(self.tensor[:, :3][None], rotation, axis=axis, return_mat=True)
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation
        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction: 'str'='horizontal') ->None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector: 'Union[Tensor, np.ndarray]') ->None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f'Unsupported translation vector of shape {trans_vector.shape}')
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range: 'Union[Tensor, np.ndarray, Sequence[float]]') ->Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (self.tensor[:, 0] > point_range[0]) & (self.tensor[:, 1] > point_range[1]) & (self.tensor[:, 2] > point_range[2]) & (self.tensor[:, 0] < point_range[3]) & (self.tensor[:, 1] < point_range[4]) & (self.tensor[:, 2] < point_range[5])
        return in_range_flags

    @property
    def bev(self) ->Tensor:
        """Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(self, point_range: 'Union[Tensor, np.ndarray, Sequence[float]]') ->Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point in order of (x_min, y_min, x_max, y_max).

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (self.bev[:, 0] > point_range[0]) & (self.bev[:, 1] > point_range[1]) & (self.bev[:, 0] < point_range[2]) & (self.bev[:, 1] < point_range[3])
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst: 'int', rt_mat: 'Optional[Union[Tensor, np.ndarray]]'=None) ->'BasePoints':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: 'float') ->None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item: 'Union[int, tuple, slice, np.ndarray, Tensor]') ->'BasePoints':
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), points_dim=self.points_dim, attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]
            keep_dims = list(set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')
        assert p.dim() == 2, f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) ->int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) ->str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list: "Sequence['BasePoints']") ->'BasePoints':
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)
        cat_points = cls(torch.cat([p.tensor for p in points_list], dim=0), points_dim=points_list[0].points_dim, attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def numpy(self) ->np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: 'Union[str, torch.device]', *args, **kwargs) ->'BasePoints':
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(self.tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def cpu(self) ->'BasePoints':
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(self.tensor.cpu(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) ->'BasePoints':
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(self.tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def clone(self) ->'BasePoints':
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def detach(self) ->'BasePoints':
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.detach(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self) ->torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) ->Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor

    def new_point(self, data: 'Union[Tensor, np.ndarray, Sequence[Sequence[float]]]') ->'BasePoints':
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, Tensor) else data
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)


@array_converter(apply_to=('val',))
def limit_period(val: 'Union[np.ndarray, Tensor]', offset: 'float'=0.5, period: 'float'=np.pi) ->Union[np.ndarray, Tensor]:
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


class BaseInstance3DBoxes:
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS: 'int' = 0

    def __init__(self, tensor: 'Union[Tensor, np.ndarray, Sequence[Sequence[float]]]', box_dim: 'int'=7, with_yaw: 'bool'=True, origin: 'Tuple[float, float, float]'=(0.5, 0.5, 0)) ->None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, f'The box dimension must be 2 and the length of the last dimension must be {box_dim}, but got boxes with shape {tensor.shape}.'
        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()
        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def shape(self) ->torch.Size:
        """torch.Size: Shape of boxes."""
        return self.tensor.shape

    @property
    def volume(self) ->Tensor:
        """Tensor: A vector with volume of each box in shape (N, )."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self) ->Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self) ->Tensor:
        """Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self) ->Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self) ->Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self) ->Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self) ->Tensor:
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is usually taken
            as the default center.

            The relative position of the centers in different kinds of boxes
            are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar. It is
            recommended to use ``bottom_center`` or ``gravity_center`` for
            clearer usage.

        Returns:
            Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self) ->Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self) ->Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self) ->Tensor:
        """Tensor: A tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self) ->Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self) ->Tensor:
        """Tensor: A tensor of 2D BEV box of each box without rotation."""
        bev_rotated_boxes = self.bev
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4])
        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def in_range_bev(self, box_range: 'Union[Tensor, np.ndarray, Sequence[float]]') ->Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box in order of (x_min, y_min, x_max, y_max).

        Note:
            The original implementation of SECOND checks whether boxes in a
            range by checking whether the points are in a convex polygon, we
            reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each box is inside the
            reference range.
        """
        in_range_flags = (self.bev[:, 0] > box_range[0]) & (self.bev[:, 1] > box_range[1]) & (self.bev[:, 0] < box_range[2]) & (self.bev[:, 1] < box_range[3])
        return in_range_flags

    @abstractmethod
    def rotate(self, angle: 'Union[Tensor, np.ndarray, float]', points: 'Optional[Union[Tensor, np.ndarray, BasePoints]]'=None) ->Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction: 'str'='horizontal', points: 'Optional[Union[Tensor, np.ndarray, BasePoints]]'=None) ->Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        pass

    def translate(self, trans_vector: 'Union[Tensor, np.ndarray]') ->None:
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size
                1x3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range: 'Union[Tensor, np.ndarray, Sequence[float]]') ->Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        gravity_center = self.gravity_center
        in_range_flags = (gravity_center[:, 0] > box_range[0]) & (gravity_center[:, 1] > box_range[1]) & (gravity_center[:, 2] > box_range[2]) & (gravity_center[:, 0] < box_range[3]) & (gravity_center[:, 1] < box_range[4]) & (gravity_center[:, 2] < box_range[5])
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst: 'int', rt_mat: 'Optional[Union[Tensor, np.ndarray]]'=None, correct_yaw: 'bool'=False) ->'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: 'float') ->None:
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset: 'float'=0.5, period: 'float'=np.pi) ->None:
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw. Defaults to 0.5.
            period (float): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: 'float'=0.0) ->Tensor:
        """Find boxes that are non-empty.

        A box is considered empty if either of its side is no larger than
        threshold.

        Args:
            threshold (float): The threshold of minimal sizes. Defaults to 0.0.

        Returns:
            Tensor: A binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item: 'Union[int, slice, np.ndarray, Tensor]') ->'BaseInstance3DBoxes':
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self) ->int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) ->str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list: "Sequence['BaseInstance3DBoxes']") ->'BaseInstance3DBoxes':
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (Sequence[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0), box_dim=boxes_list[0].box_dim, with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def numpy(self) ->np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: 'Union[str, torch.device]', *args, **kwargs) ->'BaseInstance3DBoxes':
        """Convert current boxes to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(self.tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def cpu(self) ->'BaseInstance3DBoxes':
        """Convert current boxes to cpu device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cpu device.
        """
        original_type = type(self)
        return original_type(self.tensor.cpu(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def cuda(self, *args, **kwargs) ->'BaseInstance3DBoxes':
        """Convert current boxes to cuda device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cuda device.
        """
        original_type = type(self)
        return original_type(self.tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def clone(self) ->'BaseInstance3DBoxes':
        """Clone the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def detach(self) ->'BaseInstance3DBoxes':
        """Detach the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(self.tensor.detach(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self) ->torch.device:
        """torch.device: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self) ->Iterator[Tensor]:
        """Yield a box as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A box of shape (box_dim, ).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1: "'BaseInstance3DBoxes'", boxes2: "'BaseInstance3DBoxes'") ->Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" should be in the same type, but got {type(boxes1)} and {type(boxes2)}.'
        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)
        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1: "'BaseInstance3DBoxes'", boxes2: "'BaseInstance3DBoxes'", mode: 'str'='iou') ->Tensor:
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            Tensor: Calculated 3D overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" should be in the same type, but got {type(boxes1)} and {type(boxes2)}.'
        assert mode in ['iou', 'iof']
        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)
        overlaps_h = cls.height_overlaps(boxes1, boxes2)
        boxes1_bev, boxes2_bev = boxes1.bev, boxes2.bev
        boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=0.0001)
        boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=0.0001)
        iou2d = box_iou_rotated(boxes1_bev, boxes2_bev)
        areas1 = (boxes1_bev[:, 2] * boxes1_bev[:, 3]).unsqueeze(1).expand(rows, cols)
        areas2 = (boxes2_bev[:, 2] * boxes2_bev[:, 3]).unsqueeze(0).expand(rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)
        overlaps_3d = overlaps_bev * overlaps_h
        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)
        if mode == 'iou':
            iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-08)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-08)
        return iou3d

    def new_box(self, data: 'Union[Tensor, np.ndarray, Sequence[Sequence[float]]]') ->'BaseInstance3DBoxes':
        """Create a new box object with data.

        The new box and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, the
            object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, Tensor) else data
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def points_in_boxes_part(self, points: 'Tensor', boxes_override: 'Optional[Tensor]'=None) ->Tensor:
        """Find the box in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Note:
            If a point is enclosed by multiple boxes, the index of the first
            box will be returned.

        Returns:
            Tensor: The index of the first box that each point is in with shape
            (M, ). Default value is -1 (if the point is not enclosed by any
            box).
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1
        boxes = boxes.unsqueeze(0)
        box_idx = points_in_boxes_part(points_clone, boxes)
        return box_idx.squeeze(0)

    def points_in_boxes_all(self, points: 'Tensor', boxes_override: 'Optional[Tensor]'=None) ->Tensor:
        """Find all boxes in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Returns:
            Tensor: A tensor indicating whether a point is in a box with shape
            (M, T). T is the number of boxes. Denote this tensor as A, it the
            m^th point is in the t^th box, then `A[m, t] == 1`, otherwise
            `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1
        boxes = boxes.unsqueeze(0)
        box_idxs_of_pts = points_in_boxes_all(points_clone, boxes)
        return box_idxs_of_pts.squeeze(0)

    def points_in_boxes(self, points: 'Tensor', boxes_override: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes is a deprecated method, please consider using points_in_boxes_part.')
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(self, points: 'Tensor', boxes_override: 'Optional[Tensor]'=None) ->Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes_batch is a deprecated method, please consider using points_in_boxes_all.')
        return self.points_in_boxes_all(points, boxes_override)

