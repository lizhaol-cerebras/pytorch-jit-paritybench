
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


import math


import matplotlib.pyplot as plt


import numpy as np


from matplotlib.patches import Ellipse


from matplotlib.legend_handler import HandlerLine2D


from torch import nn


import torch.utils.data as Data


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision.datasets.utils import download_and_extract_archive


from matplotlib.collections import PatchCollection


import time


import torch.optim as optim


from matplotlib.lines import Line2D


import matplotlib as mpl


from torch.utils.data import Dataset


from matplotlib.cm import ScalarMappable


from typing import Callable


from typing import Union


from typing import Tuple


from typing import Optional


import warnings


from torch.nn.functional import normalize


import collections


import numbers


from torch.utils._pytree import tree_map


from torch.utils._pytree import tree_flatten


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.modules.utils import _quadruple


from torch.nn.modules.utils import _ntuple


from typing import Any


import functools


from torch.autograd.functional import jacobian


from torch.linalg import pinv


from torch.linalg import cholesky


from torch.linalg import vecdot


import torch.nn.functional as F


from torch.distributions import MultivariateNormal


from torch import broadcast_shapes


from torch import Tensor


from torch.autograd import grad


from functools import partial


from torch.func import jacrev


from torch.func import jacfwd


from torch.func import functional_call


from torch import finfo


from torch.optim import Optimizer


from torch.linalg import cholesky_ex


from torch.linalg import lstsq


from typing import List


from torch.library import Library


import re


from collections import namedtuple


import copy


import random


import torch.linalg


import torch as torch


from torch import vmap


from torch.nn import Identity


from typing import Collection


from torchvision.transforms import Compose


class IMUIntegrator(nn.Module):

    def __init__(self):
        super().__init__()
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        if self.eval:
            rot = None
        else:
            rot = data['gt_rot'].contiguous()
        return self.imu(init_state=init_state, dt=data['dt'], gyro=data['gyro'], acc=data['acc'], rot=rot)


class IMUCorrector(nn.Module):

    def __init__(self, size_list=[6, 128, 128, 128, 6]):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        feature = torch.cat([data['acc'], data['gyro']], dim=-1)
        B, F = feature.shape[:2]
        output = self.net(feature.reshape(B * F, 6)).reshape(B, F, 6)
        corrected_acc = output[..., :3] * 0.1 + data['acc']
        corrected_gyro = output[..., 3:] * 0.1 + data['gyro']
        if self.eval:
            rot = None
        else:
            rot = data['gt_rot'].contiguous()
        return self.imu(init_state=init_state, dt=data['dt'], gyro=corrected_gyro, acc=corrected_acc, rot=rot)


class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        return error.Log().tensor()


def pixel2point(pixels, depth, intrinsics):
    """
    Convert batch of pixels with depth into points (in camera coordinate)

    Args:
        pixels: (``torch.Tensor``) The 2d coordinates of pixels in the camera
            pixel coordinate.
            Shape has to be (..., N, 2)

        depth: (``torch.Tensor``) The depths of pixels with respect to the
            sensor plane.
            Shape has to be (..., N)

        intrinsics: (``torch.Tensor``): The intrinsic parameters of cameras.
            The shape has to be (..., 3, 3).

    Returns:
        ``torch.Tensor`` The associated 3D-points with shape (..., N, 3)

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> pixels = torch.tensor([[0.5, 0.0],
        ...                        [1.0, 0.0],
        ...                        [0.0, 1.3],
        ...                        [1.0, 0.0],
        ...                        [0.5, 1.5],
        ...                        [5.0, 1.5]])
        >>> depths = torch.tensor([5.0, 3.0, 6.5, 2.0, 0.5, 0.7])
        >>> points = pp.pixel2point(pixels, depths, intrinsics)
        tensor([[-10.0000, -11.2500,   5.0000],
                [ -5.2500,  -6.7500,   3.0000],
                [-14.6250, -10.4000,   6.5000],
                [ -3.5000,  -4.5000,   2.0000],
                [ -1.0000,  -0.7500,   0.5000],
                [  0.1750,  -1.0500,   0.7000]])
    """
    assert pixels.size(-1) == 2, 'Pixels shape incorrect'
    assert depth.size(-1) == pixels.size(-2), 'Depth shape does not match pixels'
    assert intrinsics.size(-1) == intrinsics.size(-2) == 3, 'Intrinsics shape incorrect.'
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    assert not torch.any(fx == 0), 'fx Cannot contain zero'
    assert not torch.any(fy == 0), 'fy Cannot contain zero'
    pts3d_z = depth
    pts3d_x = (pixels[..., 0] - cx) * pts3d_z / fx
    pts3d_y = (pixels[..., 1] - cy) * pts3d_z / fy
    return torch.stack([pts3d_x, pts3d_y, pts3d_z], dim=-1)


def pm(input):
    """
    Returns plus or minus (:math:`\\pm`) states for tensor.

    Args:
        input (:obj:`Tensor`): the input tensor.

    Return:
        :obj:`Tensor`: the output tensor contains only :math:`-1` or :math:`+1`.

    Note:
        The :meth:`pm` function is different from :meth:`torch.sign`, which returns
        :math:`0` for zero inputs, it will return :math:`+1` if an input element is zero.

    Example:
        >>> pp.pm(torch.tensor([0.1, 0, -0.2]))
        tensor([ 1.,  1., -1.])
        >>> pp.pm(torch.tensor([0.1, 0, -0.2], dtype=torch.float64))
        tensor([ 1.,  1., -1.], dtype=torch.float64)
    """
    return torch.sign(torch.sign(input) * 2 + 1)


def homo2cart(coordinates: 'torch.Tensor'):
    """
    Converts batched Homogeneous coordinates to Cartesian coordinates
    by dividing the last row. Size of the last dimension will be reduced by 1.

    Args:
        coordinates (``torch.Tensor``): the Homogeneous coordinates to be converted.

    Returns:
        ``torch.Tensor``: the coordinates in Cartesian space.

    Example:
        >>> points = torch.tensor([[4., 3., 2., 1.], [8., 6., 4., 2.]])
        >>> homo2cart(points)
        tensor([[4., 3., 2.],
                [4., 3., 2.]])
    """
    tiny = torch.finfo(coordinates.dtype).tiny
    denum = coordinates[..., -1:].abs().clamp_(min=tiny)
    denum = pm(coordinates[..., -1:]) * denum
    return coordinates[..., :-1] / denum


HANDLED_FUNCTIONS = ['__getitem__', '__setitem__', 'cpu', 'cuda', 'float', 'double', 'to', 'detach', 'view', 'view_as', 'squeeze', 'unsqueeze', 'cat', 'stack', 'split', 'hsplit', 'dsplit', 'vsplit', 'tensor_split', 'chunk', 'concat', 'column_stack', 'dstack', 'vstack', 'hstack', 'index_select', 'masked_select', 'movedim', 'moveaxis', 'narrow', 'permute', 'reshape', 'row_stack', 'scatter', 'scatter_add', 'clone', 'swapaxes', 'swapdims', 'take', 'take_along_dim', 'tile', 'copy', 'transpose', 'unbind', 'gather', 'repeat', 'expand', 'expand_as', 'index_select', 'masked_select', 'index_copy', 'index_copy_', 'select', 'select_scatter', 'index_put', 'index_put_', 'copy_']

