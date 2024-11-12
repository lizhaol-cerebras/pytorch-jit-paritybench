
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


import numpy as np


import tensorflow as tf


import collections


import inspect


from abc import abstractmethod


from typing import Callable


from typing import Optional


from typing import Tuple


from torch.utils.data import Dataset


import logging


import random


from types import SimpleNamespace


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


import copy


import itertools


import types


from copy import deepcopy


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


from enum import IntEnum


from enum import unique


from typing import Iterator


from typing import List


from typing import Union


from torchvision.ops.boxes import box_area


from typing import Any


from typing import Sequence


from torch.nn import functional as F


from typing import Dict


from torchvision.ops import RoIAlign


import time


from collections import Counter


from collections import Mapping


from torch.nn.utils import clip_grad


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from collections import OrderedDict


from collections import abc


import torch.nn.functional as F


from torch import nn


from abc import ABCMeta


import torch.nn as nn


from torch.autograd.function import Function


from torch.nn.modules.utils import _ntuple


from torch import Tensor


import warnings


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from functools import lru_cache


from torch.nn.modules.utils import _pair


from collections import namedtuple


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torchvision.ops import boxes as box_ops


from torchvision.ops import nms


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.optim.lr_scheduler import OneCycleLR


from torch.optim.lr_scheduler import _LRScheduler


from typing import Set


from torch import optim


from itertools import zip_longest


from torch.nn.parallel import DataParallel


import re


from collections import defaultdict


import torchvision


import functools


from typing import IO


from typing import MutableMapping


from scipy.optimize import linear_sum_assignment


from torch import stack as tstack


from abc import abstractproperty


import collections.abc


import torch.utils.checkpoint as checkpoint


from torch.cuda.amp import autocast


from functools import wraps


from torch.nn.init import normal_


from torch.nn.init import zeros_


from torch.nn.init import kaiming_normal_


import uuid


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


class _SymEig3x3(nn.Module):
    """
    Optimized implementation of eigenvalues and eigenvectors computation for symmetric 3x3
     matrices.

    Please see https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
     and https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    """

    def __init__(self, eps: 'Optional[float]'=None) ->None:
        """
        Args:
            eps: epsilon to specify, if None then use torch.float eps
        """
        super().__init__()
        self.register_buffer('_identity', torch.eye(3))
        self.register_buffer('_rotation_2d', torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.register_buffer('_rotations_3d', self._create_rotation_matrices(self._rotation_2d))
        self._eps = eps or torch.finfo(torch.float).eps

    @staticmethod
    def _create_rotation_matrices(rotation_2d) ->torch.Tensor:
        """
        Compute rotations for later use in U V computation

        Args:
            rotation_2d: a Ï�/2 rotation matrix.

        Returns:
            a (3, 3, 3) tensor containing 3 rotation matrices around each of the coordinate axes
            by Ï�/2
        """
        rotations_3d = torch.zeros((3, 3, 3))
        rotation_axes = set(range(3))
        for rotation_axis in rotation_axes:
            rest = list(rotation_axes - {rotation_axis})
            rotations_3d[rotation_axis][rest[0], rest] = rotation_2d[0]
            rotations_3d[rotation_axis][rest[1], rest] = rotation_2d[1]
        return rotations_3d

    def forward(self, inputs: 'torch.Tensor', eigenvectors: 'bool'=True) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute eigenvalues and (optionally) eigenvectors

        Args:
            inputs: symmetric matrices with shape of (..., 3, 3)
            eigenvectors: whether should we compute only eigenvalues or eigenvectors as well

        Returns:
            Either a tuple of (eigenvalues, eigenvectors) or eigenvalues only, depending on
             given params. Eigenvalues are of shape (..., 3) and eigenvectors (..., 3, 3)
        """
        if inputs.shape[-2:] != (3, 3):
            raise ValueError('Only inputs of shape (..., 3, 3) are supported.')
        inputs_diag = inputs.diagonal(dim1=-2, dim2=-1)
        inputs_trace = inputs_diag.sum(-1)
        q = inputs_trace / 3.0
        p1 = ((inputs ** 2).sum(dim=(-1, -2)) - (inputs_diag ** 2).sum(-1)) / 2
        p2 = ((inputs_diag - q[..., None]) ** 2).sum(dim=-1) + 2.0 * p1.clamp(self._eps)
        p = torch.sqrt(p2 / 6.0)
        B = (inputs - q[..., None, None] * self._identity) / p[..., None, None]
        r = torch.det(B) / 2.0
        r = r.clamp(-1.0 + self._eps, 1.0 - self._eps)
        phi = torch.acos(r) / 3.0
        eig1 = q + 2 * p * torch.cos(phi)
        eig2 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
        eig3 = 3 * q - eig1 - eig2
        eigenvals = torch.stack((eig2, eig3, eig1), dim=-1)
        diag_soft_cond = torch.exp(-(p1 / (6 * self._eps)) ** 2).detach()[..., None]
        diag_eigenvals, _ = torch.sort(inputs_diag, dim=-1)
        eigenvals = diag_soft_cond * diag_eigenvals + (1.0 - diag_soft_cond) * eigenvals
        if eigenvectors:
            eigenvecs = self._construct_eigenvecs_set(inputs, eigenvals)
        else:
            eigenvecs = None
        return eigenvals, eigenvecs

    def _construct_eigenvecs_set(self, inputs: 'torch.Tensor', eigenvals: 'torch.Tensor') ->torch.Tensor:
        """
        Construct orthonormal set of eigenvectors by given inputs and pre-computed eigenvalues

        Args:
            inputs: tensor of symmetric matrices of shape (..., 3, 3)
            eigenvals: tensor of pre-computed eigenvalues of of shape (..., 3, 3)

        Returns:
            Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
             set
        """
        eigenvecs_tuple_for_01 = self._construct_eigenvecs(inputs, eigenvals[..., 0], eigenvals[..., 1])
        eigenvecs_for_01 = torch.stack(eigenvecs_tuple_for_01, dim=-1)
        eigenvecs_tuple_for_21 = self._construct_eigenvecs(inputs, eigenvals[..., 2], eigenvals[..., 1])
        eigenvecs_for_21 = torch.stack(eigenvecs_tuple_for_21[::-1], dim=-1)
        eigenvecs_cond = (eigenvals[..., 1] - eigenvals[..., 0] > eigenvals[..., 2] - eigenvals[..., 1]).detach()
        eigenvecs = torch.where(eigenvecs_cond[..., None, None], eigenvecs_for_01, eigenvecs_for_21)
        return eigenvecs

    def _construct_eigenvecs(self, inputs: 'torch.Tensor', alpha0: 'torch.Tensor', alpha1: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct an orthonormal set of eigenvectors by given pair of eigenvalues.

        Args:
            inputs: tensor of symmetric matrices of shape (..., 3, 3)
            alpha0: first eigenvalues of shape (..., 3)
            alpha1: second eigenvalues of shape (..., 3)

        Returns:
            Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
             set
        """
        ev0 = self._get_ev0(inputs - alpha0[..., None, None] * self._identity)
        u, v = self._get_uv(ev0)
        ev1 = self._get_ev1(inputs - alpha1[..., None, None] * self._identity, u, v)
        ev2 = torch.cross(ev0, ev1, dim=-1)
        return ev0, ev1, ev2

    def _get_ev0(self, char_poly: 'torch.Tensor') ->torch.Tensor:
        """
        Construct the first normalized eigenvector given a characteristic polynomial

        Args:
            char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)

        Returns:
            Tensor of first eigenvectors of shape (..., 3)
        """
        r01 = torch.cross(char_poly[..., 0, :], char_poly[..., 1, :], dim=-1)
        r12 = torch.cross(char_poly[..., 1, :], char_poly[..., 2, :], dim=-1)
        r02 = torch.cross(char_poly[..., 0, :], char_poly[..., 2, :], dim=-1)
        cross_products = torch.stack((r01, r12, r02), dim=-2)
        cross_products += self._eps * self._sign_without_zero(cross_products[..., :1, :])
        norms_sq = (cross_products ** 2).sum(dim=-1)
        max_norms_index = norms_sq.argmax(dim=-1)
        max_cross_products = self._gather_by_index(cross_products, max_norms_index[..., None, None], -2)
        max_norms_sq = self._gather_by_index(norms_sq, max_norms_index[..., None], -1)
        return max_cross_products / torch.sqrt(max_norms_sq[..., None])

    def _gather_by_index(self, source: 'torch.Tensor', index: 'torch.Tensor', dim: 'int') ->torch.Tensor:
        """
        Selects elements from the given source tensor by provided index tensor.
        Number of dimensions should be the same for source and index tensors.

        Args:
            source: input tensor to gather from
            index: index tensor with indices to gather from source
            dim: dimension to gather across

        Returns:
            Tensor of shape same as the source with exception of specified dimension.
        """
        index_shape = list(source.shape)
        index_shape[dim] = 1
        return source.gather(dim, index.expand(index_shape)).squeeze(dim)

    def _get_uv(self, w: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes unit-length vectors U and V such that {U, V, W} is a right-handed
        orthonormal set.

        Args:
            w: eigenvector tensor of shape (..., 3)

        Returns:
            Tuple of U and V unit-length vector tensors of shape (..., 3)
        """
        min_idx = w.abs().argmin(dim=-1)
        rotation_2d = self._rotations_3d[min_idx]
        u = F.normalize((rotation_2d @ w[..., None])[..., 0], dim=-1)
        v = torch.cross(w, u, dim=-1)
        return u, v

    def _get_ev1(self, char_poly: 'torch.Tensor', u: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
        """
        Computes the second normalized eigenvector given a characteristic polynomial
        and U and V vectors

        Args:
            char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)
            u: unit-length vectors from _get_uv method
            v: unit-length vectors from _get_uv method

        Returns:
            desc
        """
        j = torch.stack((u, v), dim=-1)
        m = j.transpose(-1, -2) @ char_poly @ j
        is_acute_sign = self._sign_without_zero((m[..., 0, :] * m[..., 1, :]).sum(dim=-1)).detach()
        rowspace = m[..., 0, :] + is_acute_sign[..., None] * m[..., 1, :]
        rowspace += self._eps * self._sign_without_zero(rowspace[..., :1])
        return (j @ F.normalize(rowspace @ self._rotation_2d, dim=-1)[..., None])[..., 0]

    @staticmethod
    def _sign_without_zero(tensor):
        """
        Args:
            tensor: an arbitrary shaped tensor

        Returns:
            Tensor of the same shape as an input, but with 1.0 if tensor > 0.0 and -1.0
             otherwise
        """
        return 2.0 * (tensor > 0.0) - 1.0


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor(torch.Tensor): float matrix of Nx4.
    """
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor[0], np.ndarray):
                tensor = np.array(tensor)
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device('cpu'))
        else:
            tensor = tensor
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4))
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor

    def clone(self) ->'Boxes':
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: 'str') ->'Boxes':
        return Boxes(self.tensor)

    def area(self) ->torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: 'BoxSizeType') ->None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), 'Box tensor contains infinite or NaN!'
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: 'int'=0) ->torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: 'Union[int, slice, torch.BoolTensor]') ->'Boxes':
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
        with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, 'Indexing on Boxes with {} failed to return a matrix!'.format(item)
        return Boxes(b)

    def __len__(self) ->int:
        return self.tensor.shape[0]

    def __repr__(self) ->str:
        return 'Boxes(' + str(self.tensor) + ')'

    def inside_box(self, box_size: 'BoxSizeType', boundary_threshold: 'int'=0) ->torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (self.tensor[..., 0] >= -boundary_threshold) & (self.tensor[..., 1] >= -boundary_threshold) & (self.tensor[..., 2] < width + boundary_threshold) & (self.tensor[..., 3] < height + boundary_threshold)
        return inds_inside

    def get_centers(self) ->torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: 'float', scale_y: 'float') ->None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: "List['Boxes']") ->'Boxes':
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert all(isinstance(box, Boxes) for box in boxes_list)
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) ->torch.device:
        return self.tensor.device

    def __iter__(self) ->Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def _create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    shifts_x = torch.arange(shifts_start, grid_width * stride + shifts_start, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(shifts_start, grid_height * stride + shifts_start, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class DefaultAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        sizes = config.sizes
        aspect_ratios = config.aspect_ratios
        self.strides = [x.stride for x in input_shape]
        self.offset = config.offset
        assert 0.0 <= self.offset < 1.0, self.offset
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes to use
            for the i-th feature map. If len(sizes) == 1, then the same list of
            anchor sizes, given by sizes[0], is used for all feature maps. Anchor
            sizes are given in absolute lengths in units of the input image;
            they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]]): aspect_ratios[i] is the list of
            anchor aspect ratios to use for the i-th feature map. If
            len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
            given by aspect_ratios[0], is used for all feature maps.
        strides (list[int]): stride of each input feature.
        """
        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

    def _calculate_anchors(self, sizes, aspect_ratios):
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)
        cell_anchors = [self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)]
        return BufferList(cell_anchors)

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.

                In standard RPN models, `num_cell_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)
        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors


class RotatedBoxes(Boxes):
    """
    This structure stores a list of rotated boxes as a Nx5 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    """

    def __init__(self, tensor: 'torch.Tensor'):
        """
        Args:
            tensor (Tensor[float]): a Nx5 matrix.  Each row is
                (x_center, y_center, width, height, angle),
                in which angle is represented in degrees.
                While there's no strict range restriction for it,
                the recommended principal range is between [-180, 180) degrees.

        Assume we have a horizontal box B = (x_center, y_center, width, height),
        where width is along the x-axis and height is along the y-axis.
        The rotated box B_rot (x_center, y_center, width, height, angle)
        can be seen as:

        1. When angle == 0:
           B_rot == B
        2. When angle > 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CCW;
        3. When angle < 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CW.

        Mathematically, since the right-handed coordinate system for image space
        is (y, x), where y is top->down and x is left->right, the 4 vertices of the
        rotated rectangle :math:`(yr_i, xr_i)` (i = 1, 2, 3, 4) can be obtained from
        the vertices of the horizontal rectangle (y_i, x_i) (i = 1, 2, 3, 4)
        in the following way (:math:`\\theta = angle*\\pi/180` is the angle in radians,
        (y_c, x_c) is the center of the rectangle):

        .. math::

            yr_i = \\cos(\\theta) (y_i - y_c) - \\sin(\\theta) (x_i - x_c) + y_c,

            xr_i = \\sin(\\theta) (y_i - y_c) + \\cos(\\theta) (x_i - x_c) + x_c,

        which is the standard rigid-body rotation transformation.

        Intuitively, the angle is
        (1) the rotation angle from y-axis in image space
        to the height vector (top->down in the box's local coordinate system)
        of the box in CCW, and
        (2) the rotation angle from x-axis in image space
        to the width vector (left->right in the box's local coordinate system)
        of the box in CCW.

        More intuitively, consider the following horizontal box ABCD represented
        in (x1, y1, x2, y2): (3, 2, 7, 4),
        covering the [3, 7] x [2, 4] region of the continuous coordinate system
        which looks like this:

        .. code:: none

            O--------> x
            |
            |  A---B
            |  |   |
            |  D---C
            |
            v y

        Note that each capital letter represents one 0-dimensional geometric point
        instead of a 'square pixel' here.

        In the example above, using (x, y) to represent a point we have:

        .. math::

            O = (0, 0), A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)

        We name vector AB = vector DC as the width vector in box's local coordinate system, and
        vector AD = vector BC as the height vector in box's local coordinate system. Initially,
        when angle = 0 degree, they're aligned with the positive directions of x-axis and y-axis
        in the image space, respectively.

        For better illustration, we denote the center of the box as E,

        .. code:: none

            O--------> x
            |
            |  A---B
            |  | E |
            |  D---C
            |
            v y

        where the center E = ((3+7)/2, (2+4)/2) = (5, 3).

        Also,

        .. math::

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Therefore, the corresponding representation for the same shape in rotated box in
        (x_center, y_center, width, height, angle) format is:

        (5, 3, 4, 2, 0),

        Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
        CCW (counter-clockwise) by definition. It looks like this:

        .. code:: none

            O--------> x
            |   B-C
            |   | |
            |   |E|
            |   | |
            |   A-D
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CCW with regard to E:
        A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)

        Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
        vector AD or vector BC (the top->down height vector in box's local coordinate system),
        or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
        width vector in box's local coordinate system).

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
        by definition? It looks like this:

        .. code:: none

            O--------> x
            |   D-A
            |   | |
            |   |E|
            |   | |
            |   C-B
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CW with regard to E:
        A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
        will be 1. However, these two will generate different RoI Pooling results and
        should not be treated as an identical box.

        On the other hand, it's easy to see that (X, Y, W, H, A) is identical to
        (X, Y, W, H, A+360N), for any integer N. For example (5, 3, 4, 2, 270) would be
        identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
        equivalent to rotating the same shape 90 degrees CW.

        We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):

        .. code:: none

            O--------> x
            |
            |  C---D
            |  | E |
            |  B---A
            |
            v y

        .. math::

            A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Finally, this is a very inaccurate (heavily quantized) illustration of
        how (5, 3, 4, 2, 60) looks like in case anyone wonders:

        .. code:: none

            O--------> x
            |     B            |    /  C
            |   /E /
            |  A  /
            |   `D
            v y

        It's still a rectangle with center of (5, 3), width of 4 and height of 2,
        but its angle (and thus orientation) is somewhere between
        (5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 5, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()
        self.tensor = tensor

    def clone(self) ->'RotatedBoxes':
        """
        Clone the RotatedBoxes.

        Returns:
            RotatedBoxes
        """
        return RotatedBoxes(self.tensor.clone())

    def to(self, device: 'str') ->'RotatedBoxes':
        return RotatedBoxes(self.tensor)

    def area(self) ->torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = box[:, 2] * box[:, 3]
        return area

    def normalize_angles(self) ->None:
        """
        Restrict angles to the range of [-180, 180) degrees
        """
        self.tensor[:, 4] = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0

    def clip(self, box_size: 'Boxes.BoxSizeType', clip_angle_threshold: 'float'=1.0) ->None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        For RRPN:
        Only clip boxes that are almost horizontal with a tolerance of
        clip_angle_threshold to maintain backward compatibility.

        Rotated boxes beyond this threshold are not clipped for two reasons:

        1. There are potentially multiple ways to clip a rotated box to make it
           fit within the image.
        2. It's tricky to make the entire rectangular box fit within the image
           and still be able to not leave out pixels of interest.

        Therefore we rely on ops like RoIAlignRotated to safely handle this.

        Args:
            box_size (height, width): The clipping box's size.
            clip_angle_threshold:
                Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
                we do the clipping as horizontal boxes.
        """
        h, w = box_size
        self.normalize_angles()
        idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]
        x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0
        y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0
        x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0
        y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0
        x1.clamp_(min=0, max=w)
        y1.clamp_(min=0, max=h)
        x2.clamp_(min=0, max=w)
        y2.clamp_(min=0, max=h)
        self.tensor[idx, 0] = (x1 + x2) / 2.0
        self.tensor[idx, 1] = (y1 + y2) / 2.0
        self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)
        self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)

    def nonempty(self, threshold: 'int'=0) ->torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor: a binary vector which represents
            whether each box is empty (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2]
        heights = box[:, 3]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: 'Union[int, slice, torch.BoolTensor]') ->'RotatedBoxes':
        """
        Returns:
            RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, 'Indexing on RotatedBoxes with {} failed to return a matrix!'.format(item)
        return RotatedBoxes(b)

    def __len__(self) ->int:
        return self.tensor.shape[0]

    def __repr__(self) ->str:
        return 'RotatedBoxes(' + str(self.tensor) + ')'

    def inside_box(self, box_size: 'Boxes.BoxSizeType', boundary_threshold: 'int'=0) ->torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box covering
                [0, width] x [0, height]
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        For RRPN, it might not be necessary to call this function since it's common
        for rotated box to extend to outside of the image boundaries
        (the clip function only clips the near-horizontal boxes)

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        cnt_x = self.tensor[..., 0]
        cnt_y = self.tensor[..., 1]
        half_w = self.tensor[..., 2] / 2.0
        half_h = self.tensor[..., 3] / 2.0
        a = self.tensor[..., 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))
        max_rect_dx = c * half_w + s * half_h
        max_rect_dy = c * half_h + s * half_w
        inds_inside = (cnt_x - max_rect_dx >= -boundary_threshold) & (cnt_y - max_rect_dy >= -boundary_threshold) & (cnt_x + max_rect_dx < width + boundary_threshold) & (cnt_y + max_rect_dy < height + boundary_threshold)
        return inds_inside

    def get_centers(self) ->torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return self.tensor[:, :2]

    def scale(self, scale_x: 'float', scale_y: 'float') ->None:
        """
        Scale the rotated box with horizontal and vertical scaling factors
        Note: when scale_factor_x != scale_factor_y,
        the rotated box does not preserve the rectangular shape when the angle
        is not a multiple of 90 degrees under resize transformation.
        Instead, the shape is a parallelogram (that has skew)
        Here we make an approximation by fitting a rotated rectangle to the parallelogram.
        """
        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y
        theta = self.tensor[:, 4] * math.pi / 180.0
        c = torch.cos(theta)
        s = torch.sin(theta)
        self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)
        self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)
        self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi

    @classmethod
    def cat(cls, boxes_list: "List['RotatedBoxes']") ->'RotatedBoxes':
        """
        Concatenates a list of RotatedBoxes into a single RotatedBoxes

        Arguments:
            boxes_list (list[RotatedBoxes])

        Returns:
            RotatedBoxes: the concatenated RotatedBoxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, RotatedBoxes) for box in boxes_list)
        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) ->str:
        return self.tensor.device

    def __iter__(self) ->Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (5,) at a time.
        """
        yield from self.tensor


class RotatedAnchorGenerator(nn.Module):
    """
    The anchor generator used by Rotated RPN (RRPN).
    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        sizes = config.sizes
        aspect_ratios = config.aspect_ratios
        angles = config.angles
        self.strides = [x.stride for x in input_shape]
        self.offset = config.offset
        assert 0.0 <= self.offset < 1.0, self.offset
        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)

    def _calculate_anchors(self, sizes, aspect_ratios, angles):
        """
        Args:
            sizes (list[list[int]]): sizes[i] is the list of anchor sizes to use
                for the i-th feature map. If len(sizes) == 1, then the same list of
                anchor sizes, given by sizes[0], is used for all feature maps. Anchor
                sizes are given in absolute lengths in units of the input image;
                they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]]): aspect_ratios[i] is the list of
                anchor aspect ratios to use for the i-th feature map. If
                len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
                given by aspect_ratios[0], is used for all feature maps.
            angles (list[list[float]]): angles[i] is the list of
                anchor angles to use for the i-th feature map. If
                len(angles) == 1, then the same list of anchor angles,
                given by angles[0], is used for all feature maps.
        """
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        if len(angles) == 1:
            angles *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)
        assert self.num_features == len(angles)
        cell_anchors = [self.generate_cell_anchors(size, aspect_ratio, angle).float() for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)]
        return BufferList(cell_anchors)

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 5

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios, 2 sizes and 5 angles, the number of anchors is 30.

                In standard RRPN models, `num_cell_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            zeros = torch.zeros_like(shift_x)
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)
            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))
        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), angles=(-90, -60, -30, 0, 30, 60, 90)):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.
            angles (tuple[float]]): Angles of boxes indicating how many degrees
                the boxes are rotated counter-clockwise.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
                storing anchor boxes in (x_ctr, y_ctr, w, h, angle) format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                anchors.extend([0, 0, w, h, a] for a in angles)
        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[RotatedBoxes]]:
                a list of #image elements. Each is a list of #feature level RotatedBoxes.
                The RotatedBoxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = RotatedBoxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)
        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors


class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        self.num_shifts = config.num_shifts
        self.strides = [x.stride for x in input_shape]
        self.offset = config.offset
        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)
            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))
        return shifts_over_all

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.

        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)
        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm), 'SyncBatchNorm does not support empty inputs!'
        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(self.norm, torch.nn.GroupNorm), 'GroupNorm does not support empty inputs in PyTorch <=1.4!'
            output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for i, p, di, k, s in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SwishImplementation(torch.autograd.Function):
    """
    Swish activation function memory-efficient implementation.

    This implementation explicitly processes the gradient, it keeps a copy of the input tensor,
    and uses it to calculate the gradient during the back-propagation phase.
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Conv2dSamePadding(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support "SAME" padding mode and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        self.padding_method = kwargs.pop('padding', None)
        if self.padding_method is None:
            if len(args) >= 5:
                self.padding_method = args[4]
            else:
                self.padding_method = 0
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == 'SAME':
                super().__init__(*args, **kwargs, padding=0)
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
                if isinstance(self.dilation, int):
                    self.dilation = [self.dilation] * 2
                elif len(self.dilation) == 1:
                    self.dilation = [self.dilation[0]] * 2
            else:
                raise ValueError('Unknown padding method: {}'.format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == 'SAME':
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size
                dilation_h, dilation_w = self.dilation
                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)
                padding_needed_h = max(0, (output_h - 1) * stride_h + (kernel_size_h - 1) * dilation_h + 1 - input_h)
                padding_needed_w = max(0, (output_w - 1) * stride_w + (kernel_size_w - 1) * dilation_w + 1 - input_w)
                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top
                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError('Unknown padding method: {}'.format(self.padding_method))
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConvBlock(torch.nn.Module):
    """
    Depthwise seperable convolution block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None, activation=None):
        """
        Args:
            in_channels (int): the number of input tensor channels.
            out_channels (int):the number of output tensor channels.
            kernel_size (int): the kernel size.
            stride (int or tuple or list): the stride.
            bias (bool): if `True`, the pointwise conv applies bias.
            apply_bn (bool): if `True`, apply BN layer after conv layer.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        super(SeparableConvBlock, self).__init__()
        self.norm = norm
        self.activation = activation
        self.depthwise = Conv2dSamePadding(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if bias:
            self.bias = self.pointwise.bias

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Swish(nn.Module):
    """
    Implement the Swish activation function.
    See: https://arxiv.org/abs/1710.05941 for more details.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


logger = logging.getLogger('efg.data.datasets.coco')


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """
    _version = 3

    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = self.running_mean.clone().detach()
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = self.running_var.clone().detach()
        else:
            if version < 2:
                if prefix + 'running_mean' not in state_dict:
                    state_dict[prefix + 'running_mean'] = torch.zeros_like(self.running_mean)
                if prefix + 'running_var' not in state_dict:
                    state_dict[prefix + 'running_var'] = torch.ones_like(self.running_var)
            if version < 3:
                logger.info('FrozenBatchNorm {} is upgraded to version 3.'.format(prefix.rstrip('.')))
                state_dict[prefix + 'running_var'] -= self.eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = bn_module.BatchNorm2d, bn_module.SyncBatchNorm
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, list):
        norm, args = norm
    else:
        args = None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm2d, 'BN1d': BatchNorm1d, 'SyncBN': NaiveSyncBatchNorm, 'SyncBN1d': NaiveSyncBatchNorm1d, 'FrozenBN': FrozenBatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32, channels), 'nnSyncBN': nn.SyncBatchNorm}[norm]
    if args:
        return norm(out_channels, **args)
    else:
        return norm(out_channels)


class BiFPNLayer(nn.Module):
    """
    This module implements one layer of BiFPN, and BiFPN can be obtained
    by stacking this module multiple times.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, input_size, in_channels_list, out_channels, fuse_type='fast', norm='BN', memory_efficient=True):
        """
        input_size (int): the input image size.
        in_channels_list (list): the number of input tensor channels per level.
        out_channels (int): the number of output tensor channels.
        fuse_type (str): now only support three weighted fusion approaches:
            * fast:    Output = sum(Input_i * w_i / sum(w_j))
            * sotfmax: Output = sum(Input_i * e ^ w_i / sum(e ^ w_j))
            * sum:     Output = sum(Input_i) / len(Input_i)
        norm (str): the normalization to use.
        memory_efficient (bool): use `MemoryEfficientSwish` or `Swish` as activation function.
        """
        super(BiFPNLayer, self).__init__()
        assert fuse_type in ('fast', 'softmax', 'sum'), f'Unknown fuse method: {fuse_type}. Please select in [fast, sotfmax, sum].'
        self.input_size = input_size
        self.in_channels_list = in_channels_list
        self.fuse_type = fuse_type
        self.levels = len(in_channels_list)
        self.nodes_input_offsets = [[3, 4], [2, 5], [1, 6], [0, 7], [1, 7, 8], [2, 6, 9], [3, 5, 10], [4, 11]]
        self.nodes_strides = [(2 ** x) for x in [6, 5, 4, 3, 4, 5, 6, 7]]
        self.resample_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            resample_convs_i = nn.ModuleList()
            for input_offset in node_i_input_offsets:
                if self.in_channels_list[input_offset] != out_channels:
                    resample_conv = Conv2d(self.in_channels_list[input_offset], out_channels, kernel_size=1, stride=1, padding=0, norm=get_norm(norm, out_channels), activation=None)
                else:
                    resample_conv = nn.Identity()
                self.in_channels_list.append(out_channels)
                resample_convs_i.append(resample_conv)
            self.resample_convs.append(resample_convs_i)
        self.edge_weights = nn.ParameterList()
        for node_i_input_offsets in self.nodes_input_offsets:
            if fuse_type == 'fast' or fuse_type == 'softmax':
                weights_i = nn.Parameter(torch.ones(len(node_i_input_offsets), dtype=torch.float32), requires_grad=True)
            elif fuse_type == 'sum':
                weights_i = nn.Parameter(torch.ones(len(node_i_input_offsets), dtype=torch.float32), requires_grad=False)
            else:
                raise ValueError('Unknown fuse method: {}'.format(self.fuse_type))
            self.edge_weights.append(weights_i)
        self.combine_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            combine_conv = SeparableConvBlock(out_channels, out_channels, kernel_size=3, padding='SAME', norm=get_norm(norm, out_channels), activation=None)
            self.combine_convs.append(combine_conv)
        self.act = MemoryEfficientSwish() if memory_efficient else Swish()
        self.down_sampling = MaxPool2d(kernel_size=3, stride=2, padding='SAME')
        self.up_sampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        self.nodes_features = inputs
        for node_idx, (node_i_input_offsets, node_i_stride) in enumerate(zip(self.nodes_input_offsets, self.nodes_strides)):
            if self.fuse_type == 'fast':
                weights_i = F.relu(self.edge_weights[node_idx])
            elif self.fuse_type == 'softmax':
                weights_i = self.edge_weights[node_idx].softmax(dim=0)
            elif self.fuse_type == 'sum':
                weights_i = self.edge_weights[node_idx]
            target_width = self.input_size / node_i_stride
            edge_features = []
            for offset_idx, offset in enumerate(node_i_input_offsets):
                edge_feature = self.nodes_features[offset]
                resample_conv = self.resample_convs[node_idx][offset_idx]
                edge_feature = resample_conv(edge_feature)
                width = edge_feature.size(-1)
                if width > target_width:
                    assert width / target_width == 2.0
                    edge_feature = self.down_sampling(edge_feature)
                elif width < target_width:
                    assert target_width / width == 2.0
                    edge_feature = self.up_sampling(edge_feature)
                edge_feature = edge_feature * (weights_i[offset_idx] / (weights_i.sum() + 0.0001))
                edge_features.append(edge_feature)
            node_i_feature = sum(edge_features)
            node_i_feature = self.act(node_i_feature)
            node_i_feature = self.combine_convs[node_idx](node_i_feature)
            self.nodes_features.append(node_i_feature)
        assert len(self.nodes_features) == 13
        return self.nodes_features[-5:]


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], 'Strides {} {} are not log2 contiguous'.format(stride, strides[i - 1])


class BiFPN(Backbone):
    """
    This module implements the BIFPN module in EfficientDet.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, input_size, bottom_up, in_features, out_channels, num_bifpn_layers, fuse_type='weighted_sum', top_block=None, norm='BN', bn_momentum=0.01, bn_eps=0.001, memory_efficient=True):
        """
        input_size (int): the input image size.
        bottom_up (Backbone): module representing the bottom up subnetwork.
            Must be a subclass of :class:`Backbone`. The multi-scale feature
            maps generated by the bottom up network, and listed in `in_features`,
            are used to generate FPN levels.
        in_features (list[str]): names of the input feature maps coming
            from the backbone to which FPN is attached. For example, if the
            backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
            of these may be used; order must be from high to low resolution.
        out_channels (int): the number of channels in the output feature maps.
        num_bifpn_layers (str): the number of bifpn layer.
        fuse_type (str): weighted feature fuse type. see: `BiFPNLayer`
        top_block (nn.Module or None): if provided, an extra operation will
            be performed on the output of the last (smallest resolution)
            FPN output, and the result will extend the result list. The top_block
            further downsamples the feature map. It must have an attribute
            "num_levels", meaning the number of extra FPN levels added by
            this block, and "in_feature", which is a string representing
            its input feature (e.g., p5).
        norm (str): the normalization to use.
        bn_momentum (float): the `momentum` parameter of the norm module.
        bn_eps (float): the `eps` parameter of the norm module.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        self.bottom_up = bottom_up
        self.top_block = top_block
        self.in_features = in_features
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(in_strides)
        self._out_feature_strides = {'p{}'.format(int(math.log2(s))): s for s in in_strides}
        if self.top_block is not None:
            s = int(math.log2(in_strides[-1]))
            for i in range(self.top_block.num_levels):
                self._out_feature_strides[f'p{s + i + 1}'] = 2 ** (s + i + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.bifpn_layers = nn.ModuleList()
        for idx in range(num_bifpn_layers):
            if idx == 0:
                bifpn_layer_in_channels = in_channels + [out_channels] * self.top_block.num_levels
            else:
                bifpn_layer_in_channels = [out_channels] * len(self._out_features)
            bifpn_layer = BiFPNLayer(input_size, bifpn_layer_in_channels, out_channels, fuse_type, norm, memory_efficient)
            self.bifpn_layers.append(bifpn_layer)
        self._size_divisibility = in_strides[-1]
        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization as per Tensorflow official implementations.
        See: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/init_ops.py
             #L437
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stddev = math.sqrt(1.0 / max(1.0, fan_in))
                m.weight.data.normal_(0, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if self.bn_momentum is not None and self.bn_eps is not None:
                    m.momentum = self.bn_momentum
                    m.eps = self.bn_eps
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *args, **kwargs):
        bottom_up_features = self.bottom_up(*args, **kwargs)
        results = [bottom_up_features[f] for f in self.in_features]
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            results.extend(self.top_block(top_block_in_feature))
        for bifpn_layer in self.bifpn_layers:
            results = bifpn_layer(results)
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))


class BiFPNP6P7(nn.Module):
    """
    This module is used in BiFPN to generate extra layers,
    P6 and P7 from EfficientNet "stage8" feature.
    """

    def __init__(self, in_channels, out_channels, norm='BN', in_feature='p4'):
        """
        Args:
            in_channels (int): the number of input tensor channels.
            out_channels (int): the number of output tensor channels.
            norm (str): the normalization to use.
        """
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=get_norm(norm, out_channels), activation=None)
        self.down_sampling = MaxPool2d(kernel_size=3, stride=2, padding='SAME')

    def forward(self, x):
        x = self.p6_conv(x)
        p6 = self.down_sampling(x)
        p7 = self.down_sampling(p6)
        return [p6, p7]


class Sequential(torch.nn.Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    To make it easier to understand, given is a small example::
        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError('kwargs only supported in py36+')
            if name in self._modules:
                raise ValueError('name exists.')
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class RPN(nn.Module):

    def __init__(self, cfg):
        super(RPN, self).__init__()
        self._layer_strides = cfg.ds_layer_strides
        self._num_filters = cfg.ds_num_filters
        self._layer_nums = cfg.layer_nums
        self._upsample_strides = cfg.us_layer_strides
        self._num_upsample_filters = cfg.us_num_filters
        self._num_input_features = cfg.num_input_features
        self.num_channels = sum(self._num_upsample_filters)
        self._norm_cfg = cfg.norm
        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)
        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)
        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(self._upsample_strides[i] / np.prod(self._layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]
        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], self._num_filters[i], layer_num, stride=self._layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = Sequential(nn.ConvTranspose2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), get_norm(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx]), nn.ReLU())
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(nn.Conv2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), get_norm(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx]), nn.ReLU())
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(nn.ZeroPad2d(1), nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False), get_norm(self._norm_cfg, planes), nn.ReLU())
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(get_norm(self._norm_cfg, planes))
            block.add(nn.ReLU())
        return block, planes

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'BN1d': ('bn1d', nn.BatchNorm1d), 'GN': ('gn', nn.GroupNorm)}
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


class RPNFixBNMom(nn.Module):

    def __init__(self, cfg):
        super(RPNFixBNMom, self).__init__()
        self._layer_strides = cfg.ds_layer_strides
        self._num_filters = cfg.ds_num_filters
        self._layer_nums = cfg.layer_nums
        self._upsample_strides = cfg.us_layer_strides
        self._num_upsample_filters = cfg.us_num_filters
        self._num_input_features = cfg.num_input_features
        self.num_channels = sum(self._num_upsample_filters)
        self._norm_cfg = dict(type='BN', eps=0.001, momentum=0.01)
        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)
        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)
        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(self._upsample_strides[i] / np.prod(self._layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]
        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], self._num_filters[i], layer_num, stride=self._layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = Sequential(nn.ConvTranspose2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1], nn.ReLU())
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(nn.Conv2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1], nn.ReLU())
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(nn.ZeroPad2d(1), nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False), build_norm_layer(self._norm_cfg, planes)[1], nn.ReLU())
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())
        return block, planes

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x


class FPN(Backbone):

    def __init__(self, bottom_up, in_features, out_channels, norm='', top_block=None, fuse_type='sum'):
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        self.out_channels = out_channels
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []
        use_bias = norm == ''
        for idx, in_channel in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)
            lateral_conv = Conv2d(in_channel, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module('fpn_lateral{}'.format(stage), lateral_conv)
            self.add_module('fpn_output{}'.format(stage), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        self._out_feature_strides = {'p{}'.format(int(math.log2(s))): s for s in in_strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides['p{}'.format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {'avg', 'sum'}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, *args, **kwargs):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(*args, **kwargs)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(x[1:], self.lateral_convs[1:], self.output_convs[1:]):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode='nearest')
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == 'avg':
                prev_features /= 2
            results.insert(0, output_conv(prev_features))
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    PN+1 feature from PN.
    """

    def __init__(self, in_feature='p5'):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in Retinanet and follow-up network to generate extra layers
    P6 and P7 from C5/P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature='res5'):
        """
        Args:
            in_feature: input feature name, e.g. "res5" stands for C5 features,
                "p5" stands for P5 feature.
        """
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class ResNetBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


def get_activation(activation):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if activation is None:
        return None
    atype = activation.type
    inplace = activation.inplace
    act = {'ReLU': nn.ReLU, 'ReLU6': nn.ReLU6}[atype]
    return act(inplace=inplace)


class BasicBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, stride=1, norm='BN', activation=None, **kwargs):
        """
        The standard block type for ResNet18 and ResNet34.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): A callable that takes the number of
                channels and returns a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        self.activation = get_activation(activation)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, norm=get_norm(norm, out_channels))
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.activation(out)
        return out


class BottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', activation=None, stride_in_1x1=False, dilation=1):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.activation = get_activation(activation)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.activation(out)
        return out


class AVDBottleneckBlock(BottleneckBlock):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', activation=None, stride_in_1x1=False, dilation=1, avd=False, avg_down=False, radix=1, bottleneck_width=64):
        super().__init__(in_channels=in_channels, out_channels=out_channels, bottleneck_channels=bottleneck_channels, stride=stride, num_groups=num_groups, norm=norm, activation=activation, stride_in_1x1=stride_in_1x1, dilation=dilation)
        self.avd = avd and stride > 1
        self.avg_down = avg_down
        self.radix = radix
        cardinality = num_groups
        group_width = int(bottleneck_channels * (bottleneck_width / 64.0)) * cardinality
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        if in_channels != out_channels and self.avg_down:
            assert self.shortcut is not None
            self.shortcut_avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, norm=get_norm(norm, out_channels))
        if self.radix > 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3, stride=1 if self.avd else stride_3x3, padding=dilation, dilation=dilation, groups=cardinality, bias=False, radix=self.radix, norm=norm)
        else:
            assert hasattr(self, 'conv2')
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        if self.radix > 1:
            for layer in [self.conv1, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        if self.radix > 1:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.activation(out)
        if self.avd:
            out = self.avd_layer(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            if self.avg_down:
                x = self.shortcut_avgpool(x)
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.activation(out)
        return out


class _DeformConv(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(_DeformConv._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            _C.deform_conv_forward(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_conv_backward_input(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_filter(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size

    @staticmethod
    @lru_cache(maxsize=128)
    def _cal_im2col_step(input_size, default_size):
        """
        Calculate proper im2col step size, which should be divisible by input_size and not larger
        than prefer_size. Meanwhile the step size should be as large as possible to be more
        efficient. So we choose the largest one among all divisors of input_size which are smaller
        than prefer_size.
        :param input_size: input batch size .
        :param default_size: default preferred im2col step size.
        :return: the largest proper step size.
        """
        if input_size <= default_size:
            return input_size
        best_step = 1
        for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
            if input_size % step == 0:
                if input_size // step <= default_size:
                    return input_size // step
                best_step = step
        return best_step


deform_conv = _DeformConv.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False, norm=None, activation=None):
        """
        Deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.norm = norm
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.bias = None
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x, offset):
        if x.numel() == 0:
            output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for i, p, di, k, s in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)
        x = deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = 'in_channels=' + str(self.in_channels)
        tmpstr += ', out_channels=' + str(self.out_channels)
        tmpstr += ', kernel_size=' + str(self.kernel_size)
        tmpstr += ', stride=' + str(self.stride)
        tmpstr += ', padding=' + str(self.padding)
        tmpstr += ', dilation=' + str(self.dilation)
        tmpstr += ', groups=' + str(self.groups)
        tmpstr += ', deformable_groups=' + str(self.deformable_groups)
        tmpstr += ', bias=False'
        return tmpstr


class _ModulatedDeformConv(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(_ModulatedDeformConv._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = _ModulatedDeformConv.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, norm=None, activation=None):
        """
        Modulated deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask):
        if x.numel() == 0:
            output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for i, p, di, k, s in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)
        x = modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = 'in_channels=' + str(self.in_channels)
        tmpstr += ', out_channels=' + str(self.out_channels)
        tmpstr += ', kernel_size=' + str(self.kernel_size)
        tmpstr += ', stride=' + str(self.stride)
        tmpstr += ', padding=' + str(self.padding)
        tmpstr += ', dilation=' + str(self.dilation)
        tmpstr += ', groups=' + str(self.groups)
        tmpstr += ', deformable_groups=' + str(self.deformable_groups)
        tmpstr += ', bias=' + str(self.with_bias)
        return tmpstr


class DeformBottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, *, bottleneck_channels, stride=1, num_groups=1, norm='BN', activation=None, stride_in_1x1=False, dilation=1, deform_modulated=False, deform_num_groups=1):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.activation = get_activation(activation)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18
        self.conv2_offset = Conv2d(bottleneck_channels, offset_channels * deform_num_groups, kernel_size=3, stride=stride_3x3, padding=1 * dilation, dilation=dilation)
        self.conv2 = deform_conv_op(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, deformable_groups=deform_num_groups, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))
        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)
        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.activation(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.activation(out)
        return out


class BasicStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, norm='BN', activation=None, deep_stem=False, stem_width=32):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.deep_stem = deep_stem
        if self.deep_stem:
            self.conv1_1 = Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, norm=get_norm(norm, stem_width))
            self.conv1_2 = Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, stem_width))
            self.conv1_3 = Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm(norm, stem_width * 2))
            for layer in [self.conv1_1, self.conv1_2, self.conv1_3]:
                if layer is not None:
                    weight_init.c2_msra_fill(layer)
        else:
            self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
            weight_init.c2_msra_fill(self.conv1)
        self.activation = get_activation(activation)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.deep_stem:
            x = self.conv1_1(x)
            x = self.activation(x)
            x = self.conv1_2(x)
            x = self.activation(x)
            x = self.conv1_3(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
        x = self.max_pool(x)
        return x

    @property
    def out_channels(self):
        if self.deep_stem:
            return self.conv1_3.out_channels
        else:
            return self.conv1.out_channels

    @property
    def stride(self):
        return 4


class ResNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None, zero_init_residual=False):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes
        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}
        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = 'res' + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.join(children))
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.conv3.norm.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}


class SparseResNet(Backbone):

    def __init__(self, stem, stages, out_channels=None, out_features=None, norm=None):
        super(SparseResNet, self).__init__()
        self.stem = stem
        self.out_channels = out_channels
        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}
        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, SparseResNetBlockBase), block
            stage = spconv.SparseSequential(*blocks)
            name = 'res' + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.join(children))
        out_channels_multiplier = [6, 3, 2]
        for idx, out_feature in enumerate(self._out_features):
            channels = self._out_feature_channels[out_feature]
            out_layer = spconv.SparseSequential(SparseConv3d(channels, channels, (3, 1, 1), (2, 1, 1), padding=(1, 0, 0), bias=False), get_norm(norm, channels), nn.ReLU())
            self.add_module(out_feature + '_out', out_layer)
            self._out_feature_channels[out_feature] *= out_channels_multiplier[idx]

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        for out_feature in self._out_features:
            out = getattr(self, out_feature + '_out')(outputs[out_feature])
            out = out.dense()
            N, C, D, H, W = out.shape
            out = out.view(N, C * D, H, W)
            outputs[out_feature] = out
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)


def replace_feature(out, new_features):
    if 'replace_feature' in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SpMiddleResNetFHD(Backbone):

    def __init__(self, num_input_features=128, out_features=['res3'], norm='BN1d'):
        super(SpMiddleResNetFHD, self).__init__()
        self.conv_input = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, bias=False, indice_key='res0'), get_norm(norm, 16), nn.ReLU(inplace=True))
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(16, 16, norm=norm, indice_key='res0'), SparseBasicBlock(16, 16, norm=norm, indice_key='res0'))
        self.conv2 = spconv.SparseSequential(SparseConv3d(16, 32, 3, 2, padding=1, bias=False), get_norm(norm, 32), nn.ReLU(inplace=True), SparseBasicBlock(32, 32, norm=norm, indice_key='res1'), SparseBasicBlock(32, 32, norm=norm, indice_key='res1'))
        self.conv3 = spconv.SparseSequential(SparseConv3d(32, 64, 3, 2, padding=1, bias=False), get_norm(norm, 64), nn.ReLU(inplace=True), SparseBasicBlock(64, 64, norm=norm, indice_key='res2'), SparseBasicBlock(64, 64, norm=norm, indice_key='res2'))
        self.conv4 = spconv.SparseSequential(SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1], bias=False), get_norm(norm, 128), nn.ReLU(inplace=True), SparseBasicBlock(128, 128, norm=norm, indice_key='res3'), SparseBasicBlock(128, 128, norm=norm, indice_key='res3'))
        self.extra_conv = spconv.SparseSequential(SparseConv3d(128, 128, (3, 1, 1), (2, 1, 1), bias=False), get_norm(norm, 128), nn.ReLU())

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        x = self.conv_input(ret)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        ret = self.extra_conv(x_conv4)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class MaxPool2dSamePadding(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support "SAME" padding mode and more features.

    See: https://github.com/pytorch/pytorch/issues/3867
    """

    def __init__(self, *args, **kwargs):
        self.padding_method = kwargs.pop('padding', None)
        if self.padding_method is None:
            if len(args) >= 3:
                self.padding_method = args[2]
            else:
                self.padding_method = 0
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == 'SAME':
                super().__init__(*args, **kwargs, padding=0)
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
            else:
                raise ValueError('Unknown padding method: {}'.format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == 'SAME':
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size
                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)
                padding_needed_h = max(0, (output_h - 1) * stride_h + (kernel_size_h - 1) + 1 - input_h)
                padding_needed_w = max(0, (output_w - 1) * stride_w + (kernel_size_w - 1) + 1 - input_w)
                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top
                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError('Unknown padding method: {}'.format(self.padding_method))
        x = super().forward(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def c2_msra_fill(module: 'nn.Module'):
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_xavier_fill(module: 'nn.Module'):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvFCHead(nn.Module):

    def __init__(self, input_dim, conv_dims=[], fc_dims=[], conv_norm='BN1d', fc_norm='LN'):
        super().__init__()
        conv_norm_relus = OrderedDict()
        for k, conv_dim in enumerate(conv_dims):
            conv_norm_relus[f'conv{k}'] = nn.Conv1d(input_dim, conv_dim, kernel_size=1, bias=not conv_norm)
            input_dim = conv_dim
            if conv_norm == 'BN1d':
                conv_norm_relus[f'conv_bn{k}'] = nn.BatchNorm1d(conv_dim)
            conv_norm_relus[f'conv_relu{k}'] = nn.ReLU()
        self.conv_norm_relus = nn.Sequential(conv_norm_relus)
        fcs = OrderedDict()
        for j, fc_dim in enumerate(fc_dims):
            fcs[f'fc{j}'] = nn.Linear(input_dim, fc_dim)
            input_dim = fc_dim
            if fc_norm == 'LN':
                fcs[f'fc_ln{j}'] = nn.LayerNorm(fc_dim)
            fcs[f'fc_relu{j}'] = nn.ReLU()
        self.fcs = nn.Sequential(fcs)
        for layer in self.conv_norm_relus:
            if isinstance(layer, nn.Conv1d):
                kaiming_init(layer)
                c2_msra_fill(layer)
        for layer in self.fcs:
            if isinstance(layer, nn.Linear):
                c2_xavier_fill(layer)

    def forward(self, x):
        y = self.conv_norm_relus(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        z = self.fcs(y)
        return z


class Head(nn.Module):

    def __init__(self, num_input, num_pred, num_cls, use_dir=False, num_dir=0, header=True, norm=None, name='', focal_loss_init=False, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir
        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)
        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)

    def forward(self, x):
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        ret_dict = {'box_preds': box_preds, 'cls_preds': cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_preds
        return ret_dict


class MultiGroupHead(nn.Module):

    def __init__(self, cfg, box_code_sizes, encode_background_as_zeros, reg_class_agnostic=False):
        super(MultiGroupHead, self).__init__()
        self.use_dir = cfg.MODEL.HEAD.LOSS_AUX.get('ENABLED', True)
        tasks = cfg.MODEL.HEAD.TASKS
        num_classes = [len(t['class_names']) for t in tasks]
        self.num_anchor_per_locs = [(2 * n) for n in num_classes]
        self.norm_cfg = cfg.MODEL.HEAD.get('NORM', None)
        self.in_channels = cfg.MODEL.HEAD.IN_CHANNES
        num_clss = []
        num_preds = []
        num_dirs = []
        for num_c, num_a, box_cs in zip(num_classes, self.num_anchor_per_locs, box_code_sizes):
            if encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)
            num_pred = num_a * box_cs
            num_preds.append(num_pred)
            num_dir = num_a * 2
            num_dirs.append(num_dir)
        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(Head(self.in_channels, num_pred, num_cls, use_dir=self.use_dir, num_dir=num_dirs[task_id], header=False, norm=self.norm_cfg))

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts


class Mlp1d(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp2d(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps: 'float'=1e-05, elementwise_affine: 'bool'=True) ->None:
        super(LayerNorm2d, self).__init__()
        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return '{channels}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LayerNorm1d(nn.Module):

    def __init__(self, channels, eps: 'float'=1e-05, elementwise_affine: 'bool'=True) ->None:
        super(LayerNorm1d, self).__init__()
        self.channels = channels
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(1, channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        mean = input.mean(1, keepdim=True)
        std = torch.sqrt(input.var(1, unbiased=False, keepdim=True) + self.eps)
        out = (input - mean) / std
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return '{channels}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Attention2d(nn.Module):

    def __init__(self, dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, out_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).flatten(-2)
        qkv = qkv.reshape(B, 3, self.num_heads, self.out_dim // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-2, -1).reshape(B, self.out_dim, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        L = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * L - 1, 2 * u - 1)
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


class AttentionWithRelPos(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_map_dim=None, num_cls_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_cls_tokens = num_cls_tokens
        if attn_map_dim is not None:
            one_dim = attn_map_dim[0]
            rel_pos_dim = 2 * one_dim - 1
            self.rel_pos = nn.Parameter(torch.zeros(num_heads, rel_pos_dim ** 2))
            tmp = torch.arange(rel_pos_dim ** 2).reshape((rel_pos_dim, rel_pos_dim))
            out = []
            offset_x = offset_y = one_dim // 2
            for y in range(one_dim):
                for x in range(one_dim):
                    for dy in range(one_dim):
                        for dx in range(one_dim):
                            out.append(tmp[dy - y + offset_y, dx - x + offset_x])
            self.rel_pos_index = torch.tensor(out, dtype=torch.long)
            trunc_normal_(self.rel_pos, std=0.02)
        else:
            self.rel_pos = None

    def forward(self, x, patch_attn=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        if self.rel_pos is not None and patch_attn:
            rel_pos = self.rel_pos[:, self.rel_pos_index].reshape(self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens)
            attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] = attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] + rel_pos
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Multilayer perceptron."""

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


class R2LAttentionPlusFFN(nn.Module):

    def __init__(self, input_channels, dim_hidden, kernel_size, num_heads, mlp_ratio=1.0, qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0.0, drop=0.0, cls_attn=True):
        super().__init__()
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [(kernel_size, kernel_size), (kernel_size, kernel_size), 0]
        self.kernel_size = kernel_size
        if cls_attn:
            self.norm0 = norm_layer(input_channels)
        else:
            self.norm0 = None
        self.norm1 = norm_layer(input_channels)
        self.attn = AttentionWithRelPos(input_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_map_dim=(kernel_size[0][0], kernel_size[0][1]), num_cls_tokens=1)
        self.norm2 = norm_layer(input_channels)
        self.mlp = Mlp(in_features=input_channels, hidden_features=int(dim_hidden * mlp_ratio), out_features=dim_hidden, act_layer=act_layer, drop=drop)
        self.expand = nn.Sequential(norm_layer(input_channels), act_layer(), nn.Linear(input_channels, dim_hidden)) if input_channels != dim_hidden else None
        self.linear = nn.Linear(dim_hidden, input_channels)

    def forward(self, xs):
        out, B, H, W, mask = xs
        cls_tokens = out[:, 0:1, ...]
        C = cls_tokens.shape[-1]
        cls_tokens = cls_tokens.reshape(B, -1, C)
        if self.norm0 is not None:
            cls_tokens = cls_tokens + self.attn(self.norm0(cls_tokens))
        cls_tokens = cls_tokens.reshape(-1, 1, C)
        out = torch.cat((cls_tokens, out[:, 1:, ...]), dim=1)
        tmp = out
        tmp = tmp + self.attn(self.norm1(tmp), patch_attn=True, mask=mask)
        identity = self.expand(tmp) if self.expand is not None else tmp
        tmp = identity + self.mlp(self.norm2(tmp))
        return self.linear(tmp)


class Projection(nn.Module):

    def __init__(self, input_channels, output_channels, act_layer, mode='sc'):
        super().__init__()
        tmp = []
        if 'c' in mode:
            ks = 2 if 's' in mode else 1
            if ks == 2:
                stride = ks
                ks = ks + 1
                padding = ks // 2
            else:
                stride = ks
                padding = 0
            if input_channels == output_channels and ks == 1:
                tmp.append(nn.Identity())
            else:
                tmp.extend([LayerNorm2d(input_channels), act_layer()])
                tmp.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=ks, stride=stride, padding=padding, groups=input_channels))
        self.proj = nn.Sequential(*tmp)
        self.proj_cls = self.proj

    def forward(self, xs):
        cls_tokens, patch_tokens = xs
        cls_tokens = self.proj_cls(cls_tokens)
        patch_tokens = self.proj(patch_tokens)
        return cls_tokens, patch_tokens


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm='BN1d', last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = get_norm(norm, self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PillarFeatureNet(nn.Module):

    def __init__(self, num_input_features=4, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), norm='BN1d'):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        assert len(num_filters) > 0
        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm=norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].unsqueeze(1) * self.vy + self.y_offset)
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze()


class PointPillarsScatter(nn.Module):

    def __init__(self, num_input_features=64, norm='BN1d', **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas


class VoxelMeanFeatureExtractor(nn.Module):

    def __init__(self, num_input_features, norm='BN1d'):
        super(VoxelMeanFeatureExtractor, self).__init__()
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class DynamicMeanVFE(nn.Module):

    def __init__(self, num_input_features, voxel_size, point_cloud_range):
        super(DynamicMeanVFE, self).__init__()
        self.num_point_features = num_input_features
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        grid_size = grid_size.int()
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_size = grid_size

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
        Returns:
            vfe_features: (num_voxels, C)
        """
        points = batch_dict['points']
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = points[:, 0].int() * self.scale_xyz + point_coords[:, 0] * self.scale_yz + point_coords[:, 1] * self.scale_z + point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        points_mean = scatter_mean(points_data, unq_inv, dim=0)
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz, unq_coords % self.scale_xyz // self.scale_yz, unq_coords % self.scale_yz // self.scale_z, unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict


class GroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


class Empty(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


class MSDeformAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output.contiguous(), ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1],
            top-left (0,0), bottom-right (1, 1), including padding area
            or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ...,
                                                            H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l), True for padding elements,
            False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class _dynamic_scatter(Function):

    @staticmethod
    def forward(ctx, feats, coors, reduce_type='max'):
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats: [N, C] float tensor. points features to be reduced
                into voxels.
            coors: [N, ndim] int tensor. corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type: str. reduce op. support 'max', 'sum' and 'mean'
        Returns:
            tuple
            voxel_feats: [M, C] float tensor. reduced features. input features
                that shares the same voxel coordinates are reduced to one row
            coordinates: [M, ndim] int tensor, voxel coordinates.
        """
        results = dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx, grad_voxel_feats, grad_voxel_coors=None):
        feats, voxel_feats, point2voxel_map, voxel_points_count = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        dynamic_point_to_voxel_backward(grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats, point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter = _dynamic_scatter.apply


class DynamicScatter(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, average_points: 'bool'):
        super(DynamicScatter, self).__init__()
        """Scatters points into voxels, used in the voxel encoder with
           dynamic voxelization

        **Note**: The CPU and GPU implementation get the same output, but
        have numerical difference after summation and division (e.g., 5e-7).

        Args:
            average_points (bool): whether to use avg pooling to scatter
                points into voxel voxel_size (list): list [x, y, z] size
                of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points, coors):
        """
        Args:
            input: NC points
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)
            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', average_points=' + str(self.average_points)
        tmpstr += ')'
        return tmpstr


class _Voxelization(Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000):
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
            voxel_num = hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3)
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        return voxelization(input, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class AutoAssignHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super(AutoAssignHead, self).__init__()
        in_channels = input_shape[0].channels
        num_classes = config.model.fcos.num_classes
        num_convs = config.model.fcos.num_convs
        prior_prob = config.model.fcos.prior_prob
        self.fpn_strides = config.model.fcos.fpn_strides
        self.norm_reg_targets = config.model.fcos.norm_reg_targets
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.obj_score = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.obj_score]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.bbox_pred.bias, 4.0)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
        """
        logits = []
        bbox_reg = []
        obj_logits = []
        for feature, stride, scale in zip(features, self.fpn_strides, self.scales):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            obj_logits.append(self.obj_score(bbox_subnet))
            bbox_pred = scale(self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * stride)
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, obj_logits


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)
    """

    def __init__(self, tensor: 'torch.Tensor', image_sizes: 'List[Tuple[int, int]]'):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w).
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) ->int:
        return len(self.image_sizes)

    def __getitem__(self, idx: 'Union[int, slice]') ->torch.Tensor:
        """
        Access the individual image in its original size.

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., :size[0], :size[1]]

    def to(self, *args: Any, **kwargs: Any) ->'ImageList':
        cast_tensor = self.tensor
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) ->torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(tensors: 'Sequence[torch.Tensor]', size_divisibility: 'int'=0, pad_ref_long: 'bool'=False, pad_value: 'float'=0.0) ->'ImageList':
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded with `pad_value`
                so that they will have the same shape.
            size_divisibility (int): If `size_divisibility > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisibility`
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        max_size = list(max(s) for s in zip(*[img.shape for img in tensors]))
        if pad_ref_long:
            max_size_max = max(max_size[-2:])
            max_size[-2:] = [max_size_max] * 2
        max_size = tuple(max_size)
        if size_divisibility > 0:
            import math
            stride = size_divisibility
            max_size = list(max_size)
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)
            max_size = tuple(max_size)
        image_sizes = [im.shape[-2:] for im in tensors]
        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            if all(x == 0 for x in padding_size):
                batched_imgs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return ImageList(batched_imgs.contiguous(), image_sizes)


class Instances:

    def __init__(self, **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._fields: 'Dict[str, Any]' = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) ->Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: 'str', val: 'Any') ->None:
        if name.startswith('_'):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: 'str') ->Any:
        if name == '_fields' or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: 'str', value: 'Any') ->None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields) and name not in ['pose', 'new_boxes']:
            assert len(self) == data_len, 'Adding a field of length {} to a Instances of length {}'.format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: 'str') ->bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: 'str') ->None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: 'str') ->Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) ->Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    def to(self, *args: Any, **kwargs: Any) ->'Instances':
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, 'to'):
                v = v
            ret.set(k, v)
        return ret

    def __getitem__(self, item: 'Union[int, slice, torch.BoolTensor]') ->'Instances':
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError('Instances index out of range!')
            else:
                item = slice(item, None, len(self))
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) ->int:
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError('Empty Instances does not support __len__!')

    def __iter__(self):
        raise NotImplementedError('`Instances` object is not iterable!')

    @staticmethod
    def cat(instance_lists: "List['Instances']") ->'Instances':
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]
        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), 'cat'):
                values = type(v0).cat(values)
            else:
                raise ValueError('Unsupported type {} for concatenation'.format(type(v0)))
            ret.set(k, values)
        return ret


class Shift2BoxTransform(object):

    def __init__(self, weights):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dl, dt, dr, db) deltas.
        """
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.

        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)
        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts), dim=-1) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        assert torch.isfinite(deltas).all().item()
        shifts = shifts
        if deltas.numel() == 0:
            return torch.empty_like(deltas)
        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(self.weights)
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2], shifts.unsqueeze(-2) + deltas[..., 2:]), dim=-1).view(deltas.size()[:-2] + (-1,))
        return boxes


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    blocks = []
    for i in range(num_blocks):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs['in_channels'] = kwargs['out_channels']
    return blocks


def build_resnet_backbone(config, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    depth = config.model.resnets.depth
    stem_width = {(18): 32, (34): 32, (50): 32, (101): 64, (152): 64, (200): 64, (269): 64}[depth]
    deep_stem = config.model.resnets.deep_stem
    if not deep_stem:
        assert getattr(config.model.resnets, 'radix', 1) <= 1, 'config.model.resnets.radix: {} > 1'.format(config.model.resnets.radix)
    norm = config.model.resnets.norm
    activation = config.model.resnets.activation
    stem = BasicStem(in_channels=input_shape.channels, out_channels=config.model.resnets.stem_out_channels, norm=norm, activation=activation, deep_stem=deep_stem, stem_width=stem_width)
    freeze_at = config.model.backbone.freeze_at
    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)
    out_features = config.model.resnets.out_features
    num_groups = config.model.resnets.num_groups
    width_per_group = config.model.resnets.width_per_group
    bottleneck_channels = num_groups * width_per_group
    in_channels = config.model.resnets.stem_out_channels
    out_channels = config.model.resnets.res2_out_channels
    stride_in_1x1 = config.model.resnets.stride_in_1x1
    res5_dilation = config.model.resnets.res5_dilation
    num_classes = config.model.resnets.num_classes
    zero_init_residual = config.model.resnets.zero_init_residual
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(res5_dilation)
    num_blocks_per_stage = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [3, 24, 36, 3], (269): [3, 30, 48, 8]}[depth]
    out_stage_idx = [{'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5, 'linear': 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    deform_on_per_stage = getattr(config.model.resnets, 'deform_on_per_stage', [False] * (max_stage_idx - 1))
    if depth in [18, 34]:
        assert out_channels == 64, 'Must set model.resnets.res2_out_channels = 64 for R18/R34'
        assert not any(deform_on_per_stage), 'model.resnets.deform_on_per_stage unsupported for R18/R34'
        assert res5_dilation == 1, 'Must set model.resnets.res5_dilation = 1 for R18/R34'
        assert num_groups == 1, 'Must set model.resnets.num_groups = 1 for R18/R34'
    stages = []
    logger = logging.getLogger(__name__)
    if not stride_in_1x1 and 'torchvision' not in config.model.weights:
        logger.warning('Using pretrain weight not from torchvision with model.resnets.stride_in_1x1 == False')
    elif stride_in_1x1 and 'torchvision' in config.model.weights and config.model.weights:
        logger.warning('Using pretrain weight from torchvision with model.resnets.stride_in_1x1 == True')
    in_channels = 2 * stem_width if deep_stem else in_channels
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or stage_idx == 5 and dilation == 2 else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'first_stride': first_stride, 'in_channels': in_channels, 'out_channels': out_channels, 'norm': norm, 'activation': activation}
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs['block_class'] = DeformBottleneckBlock
                stage_kargs['deform_modulated'] = config.model.resnets.deform_modulated
                stage_kargs['deform_num_groups'] = config.model.resnets.deform_num_groups
            elif hasattr(config.model.resnets, 'RADIX'):
                radix = config.model.resnets.radix
                avd = config.model.resnets.avd or radix > 1
                avg_down = config.model.resnets.avg_down or radix > 1
                bottleneck_width = config.model.resnets.bottleneck_width
                stage_kargs['block_class'] = AVDBottleneckBlock
                stage_kargs['avd'] = avd
                stage_kargs['avg_down'] = avg_down
                stage_kargs['radix'] = radix
                stage_kargs['bottleneck_width'] = bottleneck_width
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, num_classes=num_classes, out_features=out_features, zero_init_residual=zero_init_residual)


def build_retinanet_resnet_fpn_backbone(config, input_shape: 'ShapeSpec'):
    """
    Args:
        config: a OmegaConf config dict

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(config, input_shape)
    in_features = config.model.fpn.in_features
    out_channels = config.model.fpn.out_channels
    block_in_feature = config.model.fpn.block_in_features
    if block_in_feature == 'p5':
        in_channels_p6p7 = out_channels
    elif block_in_feature == 'res5':
        in_channels_p6p7 = bottom_up.output_shape()[block_in_feature].channels
    else:
        raise ValueError(block_in_feature)
    backbone = FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, norm=config.model.fpn.norm, top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=block_in_feature), fuse_type=config.model.fpn.fuse_type)
    return backbone


def build_backbone(config, input_shape=None):
    """
    Build a backbone from `config.model.backbone.name`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(config.model.pixel_mean))
    backbone = build_retinanet_resnet_fpn_backbone(config, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_shift_generator(config, input_shape):
    return ShiftGenerator(config.model.shift_generator, input_shape)


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


BYTES_PER_FLOAT = 4


GPU_MEM_LIMIT = 1024 ** 3


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
    N = masks.shape[0]
    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)
    img_masks = F.grid_sample(masks, grid, align_corners=False)
    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2], 'Only square mask predictions are supported'
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape
    img_h, img_w = image_shape
    if device.type == 'cpu':
        num_chunks = N
    else:
        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert num_chunks <= N, 'Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it'
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
    img_masks = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == 'cpu')
        if threshold >= 0:
            masks_chunk = masks_chunk >= threshold
        else:
            masks_chunk = masks_chunk * 255
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = output_width / results.image_size[1], output_height / results.image_size[0]
    results = Instances((output_height, output_width), **results.get_fields())
    if results.has('pred_boxes'):
        output_boxes = results.pred_boxes
    elif results.has('proposal_boxes'):
        output_boxes = results.proposal_boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    results = results[output_boxes.nonempty()]
    if results.has('pred_masks'):
        results.pred_masks = paste_masks_in_image(results.pred_masks[:, 0, :, :], results.pred_boxes, results.image_size, threshold=mask_threshold)
    if results.has('pred_keypoints'):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y
    return results


def iou_loss_v2(inputs, targets, weight=None, box_mode='xyxy', loss_type='iou', smooth=False, reduction='none'):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == 'ltrb':
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != 'xyxy':
        raise NotImplementedError
    eps = torch.finfo(torch.float32).eps
    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) * (targets[..., 3] - targets[..., 1]).clamp_(min=0)
    w_intersect = (torch.min(inputs[..., 2], targets[..., 2]) - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3]) - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)
    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)
    if loss_type == 'iou':
        loss = -ious.clamp(min=eps).log()
    elif loss_type == 'linear_iou':
        loss = 1 - ious
    elif loss_type == 'giou':
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == 'mean':
            loss = loss.sum() / max(weight.sum().item(), eps)
    elif reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    return loss


_LOG_COUNTER = Counter()


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join('utils', 'writer', 'logger.') not in code.co_filename:
            mod_name = frame.f_globals['__name__']
            if mod_name == '__main__':
                mod_name = 'efg'
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def log_first_n(lvl, msg, n=1, *, name=None, key='caller'):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = key,
    assert len(key) > 0
    caller_module, caller_key = _find_caller()
    hash_key = ()
    if 'caller' in key:
        hash_key = hash_key + caller_key
    if 'message' in key:
        hash_key = hash_key + (msg,)
    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def negative_bag_loss(logits, gamma):
    return logits ** gamma * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction='none')


def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu) ** 2 / (2 * sigma ** 2)).exp()


def normalize(x):
    return (x - x.min() + 1e-12) / (x.max() - x.min() + 1e-12)


def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    return _C.box_iou_rotated(boxes1, boxes2)


def pairwise_iou(boxes1: 'RotatedBoxes', boxes2: 'RotatedBoxes') ->None:
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).

    Args:
        boxes1, boxes2 (RotatedBoxes):
            two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


def positive_bag_loss(logits, mask, gaussian_probs):
    weight = (3 * logits).exp() * gaussian_probs * mask
    w = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    bag_prob = (w * logits).sum(dim=1)
    return F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')


class AutoAssign(nn.Module):
    """
    Implement AutoAssign (https://arxiv.org/abs/2007.03496).
    """

    def __init__(self, config):
        super(AutoAssign, self).__init__()
        self.device = torch.device(config.model.device)
        self.num_classes = config.model.fcos.num_classes
        self.in_features = config.model.fcos.in_features
        self.fpn_strides = config.model.fcos.fpn_strides
        self.focal_loss_alpha = config.model.fcos.focal_loss_alpha
        self.focal_loss_gamma = config.model.fcos.focal_loss_gamma
        self.iou_loss_type = config.model.fcos.iou_loss_type
        self.reg_weight = config.model.fcos.reg_weight
        self.score_threshold = config.model.fcos.score_thresh_test
        self.topk_candidates = config.model.fcos.topk_candidates_test
        self.nms_threshold = config.model.fcos.nms_thresh_test
        self.max_detections_per_image = config.dataset.test.detections_per_image
        self.backbone = build_backbone(config, input_shape=ShapeSpec(channels=len(config.model.pixel_mean)))
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = AutoAssignHead(config, feature_shapes)
        self.shift_generator = build_shift_generator(config, feature_shapes)
        self.shift2box_transform = Shift2BoxTransform(weights=config.model.fcos.bbox_reg_weights)
        self.mu = nn.Parameter(torch.zeros(80, 2))
        self.sigma = nn.Parameter(torch.ones(80, 2))
        pixel_mean = torch.Tensor(config.model.pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(config.model.pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'] for x in batched_inputs]
        elif 'targets' in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [x['targets'] for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)
        if self.training:
            return self.losses(shifts, gt_instances, box_cls, box_delta, box_center)
        else:
            results = self.inference(box_cls, box_delta, box_center, shifts, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def losses(self, shifts, gt_instances, box_cls, box_delta, box_center):
        box_cls_flattened = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
        pred_class_logits = cat(box_cls_flattened, dim=1)
        pred_shift_deltas = cat(box_delta_flattened, dim=1)
        pred_obj_logits = cat(box_center_flattened, dim=1)
        pred_class_probs = pred_class_logits.sigmoid()
        pred_obj_probs = pred_obj_logits.sigmoid()
        pred_box_probs = []
        num_foreground = pred_class_logits.new_zeros(1)
        num_background = pred_class_logits.new_zeros(1)
        positive_losses = []
        gaussian_norm_losses = []
        for shifts_per_image, gt_instances_per_image, pred_class_probs_per_image, pred_shift_deltas_per_image, pred_obj_probs_per_image in zip(shifts, gt_instances, pred_class_probs, pred_shift_deltas, pred_obj_probs):
            locations = torch.cat(shifts_per_image, dim=0)
            labels = gt_instances_per_image.gt_classes
            gt_boxes = gt_instances_per_image.gt_boxes
            target_shift_deltas = self.shift2box_transform.get_deltas(locations, gt_boxes.tensor.unsqueeze(1))
            is_in_boxes = target_shift_deltas.min(dim=-1).values > 0
            foreground_idxs = torch.nonzero(is_in_boxes, as_tuple=True)
            with torch.no_grad():
                predicted_boxes_per_image = self.shift2box_transform.apply_deltas(pred_shift_deltas_per_image, locations)
                gt_pred_iou = pairwise_iou(gt_boxes, Boxes(predicted_boxes_per_image)).max(dim=0, keepdim=True).values.repeat(len(gt_instances_per_image), 1)
                pred_box_prob_per_image = torch.zeros_like(pred_class_probs_per_image)
                box_prob = 1 / (1 - gt_pred_iou[foreground_idxs]).clamp_(1e-12)
                for i in range(len(gt_instances_per_image)):
                    idxs = foreground_idxs[0] == i
                    if idxs.sum() > 0:
                        box_prob[idxs] = normalize(box_prob[idxs])
                pred_box_prob_per_image[foreground_idxs[1], labels[foreground_idxs[0]]] = box_prob
                pred_box_probs.append(pred_box_prob_per_image)
            normal_probs = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                gt_shift_deltas = self.shift2box_transform.get_deltas(shifts_i, gt_boxes.tensor.unsqueeze(1))
                distances = (gt_shift_deltas[..., :2] - gt_shift_deltas[..., 2:]) / 2
                normal_probs.append(normal_distribution(distances / stride, self.mu[labels].unsqueeze(1), self.sigma[labels].unsqueeze(1)))
            normal_probs = torch.cat(normal_probs, dim=1).prod(dim=-1)
            composed_cls_prob = pred_class_probs_per_image[:, labels] * pred_obj_probs_per_image
            loss_box_reg = iou_loss_v2(pred_shift_deltas_per_image.unsqueeze(0), target_shift_deltas, box_mode='ltrb', loss_type=self.iou_loss_type, reduction='none') * self.reg_weight
            pred_reg_probs = (-loss_box_reg).exp()
            positive_losses.append(positive_bag_loss(composed_cls_prob.permute(1, 0) * pred_reg_probs, is_in_boxes.float(), normal_probs))
            num_foreground += len(gt_instances_per_image)
            num_background += normal_probs[foreground_idxs].sum().item()
            gaussian_norm_losses.append(len(gt_instances_per_image) / normal_probs[foreground_idxs].sum().clamp_(1e-12))
        if dist.is_initialized():
            dist.all_reduce(num_foreground)
            num_foreground /= dist.get_world_size()
            dist.all_reduce(num_background)
            num_background /= dist.get_world_size()
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_foreground)
        pred_box_probs = torch.stack(pred_box_probs, dim=0)
        negative_loss = negative_bag_loss(pred_class_probs * pred_obj_probs * (1 - pred_box_probs), self.focal_loss_gamma).sum() / max(1, num_background)
        loss_pos = positive_loss * self.focal_loss_alpha
        loss_neg = negative_loss * (1 - self.focal_loss_alpha)
        loss_norm = torch.stack(gaussian_norm_losses).mean() * (1 - self.focal_loss_alpha)
        return {'loss_pos': loss_pos, 'loss_neg': loss_neg, 'loss_norm': loss_norm}

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`AutoAssignHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            results_per_image = self.inference_single_image(box_cls_per_image, box_reg_per_image, box_ctr_per_image, shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(box_cls, box_delta, box_center, shifts):
            box_cls_i = (box_cls_i.sigmoid_() * box_ctr_i.sigmoid_()).flatten()
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            predicted_boxes = self.shift2box_transform.apply_deltas(box_reg_i, shifts_i)
            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[:self.max_detections_per_image]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'] for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        in_channels = input_shape[0].channels
        num_classes = config.model.fcos.num_classes
        num_convs = config.model.fcos.num_convs
        prior_prob = config.model.fcos.prior_prob
        num_shifts = build_shift_generator(config, input_shape).num_cell_shifts
        self.fpn_strides = config.model.fcos.fpn_strides
        self.centerness_on_reg = config.model.fcos.centerness_on_reg
        self.norm_reg_targets = config.model.fcos.norm_reg_targets
        assert len(set(num_shifts)) == 1, 'using differenct num_shifts value is not supported'
        num_shifts = num_shifts[0]
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_shifts * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_shifts * 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, num_shifts * 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits = []
        bbox_reg = []
        centerness = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))
            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


def cluster_nms(boxes, scores, iou_threshold):
    last_keep = torch.ones(*scores.shape)
    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]
    origin_iou_matrix = box_ops.box_iou(boxes, boxes).tril(diagonal=-1).transpose(1, 0)
    while True:
        iou_matrix = torch.mm(torch.diag(last_keep.float()), origin_iou_matrix)
        keep = iou_matrix.max(dim=0)[0] <= iou_threshold
        if (keep == last_keep).all():
            return idx[keep.nonzero(as_tuple=False)]
        last_keep = keep


def batched_clusternms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = cluster_nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def iou(boxes, top_box):
    x1 = boxes[:, 0].clamp(min=top_box[0])
    y1 = boxes[:, 1].clamp(min=top_box[1])
    x2 = boxes[:, 2].clamp(max=top_box[2])
    y2 = boxes[:, 3].clamp(max=top_box[3])
    inters = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = (top_box[2] - top_box[0]) * (top_box[3] - top_box[1]) + areas - inters
    return inters / unions


def scale_by_iou(ious, sigma, soft_mode='gaussian'):
    if soft_mode == 'linear':
        scale = ious.new_ones(ious.size())
        scale[ious >= sigma] = 1 - ious[ious >= sigma]
    else:
        scale = torch.exp(-ious ** 2 / sigma)
    return scale


def softnms(boxes, scores, sigma, score_threshold, soft_mode='gaussian'):
    assert soft_mode in ['linear', 'gaussian']
    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        top_box = boxes[idx]
        undone_mask[idx] = False
        _boxes = boxes[undone_mask]
        ious = iou(_boxes, top_box)
        scales = scale_by_iou(ious, sigma, soft_mode)
        scores[undone_mask] *= scales
        undone_mask[scores < score_threshold] = False
    return scores


def batched_softnms(boxes, scores, idxs, iou_threshold, score_threshold=0.001, soft_mode='gaussian'):
    assert soft_mode in ['linear', 'gaussian']
    assert boxes.shape[-1] == 4
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        scores[mask] = softnms(boxes[mask], scores[mask], iou_threshold, score_threshold, soft_mode)
    keep = (scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def generalized_batched_nms(boxes, scores, idxs, iou_threshold, score_threshold=0.001, nms_type='normal'):
    assert boxes.shape[-1] == 4
    if nms_type == 'normal':
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    elif nms_type.startswith('softnms'):
        keep = batched_softnms(boxes, scores, idxs, iou_threshold, score_threshold=score_threshold, soft_mode=nms_type.lstrip('softnms-'))
    elif nms_type == 'cluster':
        keep = batched_clusternms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError('NMS type not implemented: "{}"'.format(nms_type))
    return keep


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


def sigmoid_focal_loss(logits, targets, alpha: 'float'=-1, gamma: 'float'=2, reduction: 'str'='none'):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        logits: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as logits. Stores the binary
                 classification label for each element in logits
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


sigmoid_focal_loss_jit = torch.jit.script(sigmoid_focal_loss)


class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.model.device)
        self.num_classes = config.model.fcos.num_classes
        self.in_features = config.model.fcos.in_features
        self.fpn_strides = config.model.fcos.fpn_strides
        self.focal_loss_alpha = config.model.fcos.focal_loss_alpha
        self.focal_loss_gamma = config.model.fcos.focal_loss_gamma
        self.iou_loss_type = config.model.fcos.iou_loss_type
        self.center_sampling_radius = config.model.fcos.center_sampling_radius
        self.score_threshold = config.model.fcos.score_thresh_test
        self.topk_candidates = config.model.fcos.topk_candidates_test
        self.nms_threshold = config.model.fcos.nms_thresh_test
        self.nms_type = config.model.nms_type
        self.max_detections_per_image = config.dataset.test.detections_per_image
        self.backbone = build_backbone(config, input_shape=ShapeSpec(channels=len(config.model.pixel_mean)))
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(config, feature_shapes)
        self.shift_generator = build_shift_generator(config, feature_shapes)
        self.shift2box_transform = Shift2BoxTransform(weights=config.model.fcos.bbox_reg_weights)
        self.object_sizes_of_interest = config.model.fcos.object_sizes_of_interest
        pixel_mean = torch.Tensor(config.model.pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(config.model.pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'] for x in batched_inputs]
        elif 'targets' in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [x['targets'] for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)
        if self.training:
            gt_classes, gt_shifts_reg_deltas, gt_centerness = self.get_ground_truth(shifts, gt_instances)
            return self.losses(gt_classes, gt_shifts_reg_deltas, gt_centerness, box_cls, box_delta, box_center)
        else:
            results = self.inference(box_cls, box_delta, box_center, shifts, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def losses(self, gt_classes, gt_shifts_deltas, gt_centerness, pred_class_logits, pred_shift_deltas, pred_centerness):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas, pred_centerness = permute_all_cls_and_box_to_N_HWA_K_and_concat(pred_class_logits, pred_shift_deltas, pred_centerness, self.num_classes)
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = comm.all_reduce(num_foreground_centerness) / float(comm.get_world_size())
        loss_cls = sigmoid_focal_loss_jit(pred_class_logits[valid_idxs], gt_classes_target[valid_idxs], alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction='sum') / max(1.0, num_foreground)
        loss_box_reg = iou_loss_v2(pred_shift_deltas[foreground_idxs], gt_shifts_deltas[foreground_idxs], gt_centerness[foreground_idxs], box_mode='ltrb', loss_type=self.iou_loss_type, reduction='sum') / max(1.0, num_targets)
        loss_centerness = F.binary_cross_entropy_with_logits(pred_centerness[foreground_idxs], gt_centerness[foreground_idxs], reduction='sum') / max(1, num_foreground)
        return {'loss_cls': loss_cls, 'loss_box_reg': loss_box_reg, 'loss_centerness': loss_centerness}

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.

        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []
        for shifts_per_image, targets_per_image in zip(shifts, targets):
            object_sizes_of_interest = torch.cat([shifts_i.new_tensor([float(s) for s in size]).unsqueeze(0).expand(shifts_i.size(0), -1) for shifts_i, size in zip(shifts_per_image, self.object_sizes_of_interest)], dim=0)
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)
            gt_boxes = targets_per_image.gt_boxes
            deltas = self.shift2box_transform.get_deltas(shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))
            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((torch.max(centers - radius, gt_boxes.tensor[:, :2]), torch.min(centers + radius, gt_boxes.tensor[:, 2:])), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                is_in_boxes = deltas.min(dim=-1).values > 0
            max_deltas = deltas.max(dim=-1).values
            is_cared_in_the_level = (max_deltas >= object_sizes_of_interest[None, :, 0]) & (max_deltas <= object_sizes_of_interest[None, :, 1])
            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt((left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0) * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0))
            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)
        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas), torch.stack(gt_centerness)

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            results_per_image = self.inference_single_image(box_cls_per_image, box_reg_per_image, box_ctr_per_image, shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(box_cls, box_delta, box_center, shifts):
            box_cls_i = box_cls_i.flatten().sigmoid_()
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            predicted_boxes = self.shift2box_transform.apply_deltas(box_reg_i, shifts_i)
            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)
            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
        keep = generalized_batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold, nms_type=self.nms_type)
        keep = keep[:self.max_detections_per_image]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'] for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, 'inference mode with training=True'
        assert len(batched_inputs) == 1, 'inference image number > 1'
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)
        results = self.inference(box_cls, box_delta, box_center, shifts, images)
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


def batch_dice_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, 1 - targets)
    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_mask: 'float'=1, cost_dice: 'float'=1, num_points: 'int'=0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            tgt_ids = targets[b]['labels']
            cost_class = -out_prob[:, tgt_ids]
            out_mask = outputs['pred_masks'][b]
            tgt_mask = targets[b]['masks']
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_mask = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = ['cost_class: {}'.format(self.cost_class), 'cost_mask: {}'.format(self.cost_mask), 'cost_dice: {}'.format(self.cost_dice)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        src_widths = src_boxes[..., 2] - src_boxes[..., 0]
        src_heights = src_boxes[..., 3] - src_boxes[..., 1]
        src_ctr_x = src_boxes[..., 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[..., 1] + 0.5 * src_heights
        target_widths = target_boxes[..., 2] - target_boxes[..., 0]
        target_heights = target_boxes[..., 3] - target_boxes[..., 1]
        target_ctr_x = target_boxes[..., 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[..., 1] + 0.5 * target_heights
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        deltas = torch.stack((dx, dy, dw, dh), dim=-1)
        assert (src_widths > 0).all().item(), 'Input boxes to Box2BoxTransform are not valid!'
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        assert torch.isfinite(deltas).all().item(), 'Box regression deltas become infinite or NaN!'
        boxes = boxes
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
        pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds, labels, allow_low_quality_matches=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float('inf'))
        thresholds.append(float('inf'))
        assert all(low <= high for low, high in zip(thresholds[:-1], thresholds[1:]))
        assert all(label in [-1, 0, 1] for label in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
            return default_matches, default_match_labels
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None], as_tuple=False)
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        match_labels[pred_inds_to_update] = 1


def build_anchor_generator(config, input_shape):
    return DefaultAnchorGenerator(config.model.anchor_generator, input_shape)


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.

    """

    def __init__(self, config, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        in_channels = input_shape[0].channels
        num_classes = config.model.retinanet.num_classes
        num_convs = config.model.retinanet.num_convs
        prior_prob = config.model.retinanet.prior_prob
        num_anchors = build_anchor_generator(config, input_shape).num_cell_anchors
        assert len(set(num_anchors)) == 1, 'Using different number of anchors between levels is not currently supported!'
        num_anchors = num_anchors[0]
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


def smooth_l1_loss(input, target, beta: 'float', reduction: 'str'='none', size_average=False):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-05:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean' or size_average:
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class RetinaNet(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.model.device)
        self.num_classes = config.model.retinanet.num_classes
        self.in_features = config.model.retinanet.in_features
        self.focal_loss_alpha = config.model.retinanet.focal_loss_alpha
        self.focal_loss_gamma = config.model.retinanet.focal_loss_gamma
        self.smooth_l1_loss_beta = config.model.retinanet.smooth_l1_loss_beta
        self.score_threshold = config.model.retinanet.score_thresh_test
        self.topk_candidates = config.model.retinanet.topk_candidates_test
        self.nms_threshold = config.model.retinanet.nms_thresh_test
        self.nms_type = config.model.nms_type
        self.max_detections_per_image = config.dataset.test.detections_per_image
        self.backbone = build_backbone(config, input_shape=ShapeSpec(channels=len(config.model.pixel_mean)))
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(config, feature_shapes)
        self.anchor_generator = build_anchor_generator(config, feature_shapes)
        self.box2box_transform = Box2BoxTransform(weights=config.model.retinanet.bbox_reg_weights)
        self.matcher = Matcher(config.model.retinanet.iou_thresholds, config.model.retinanet.iou_labels, allow_low_quality_matches=True)
        pixel_mean = torch.Tensor(config.model.pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(config.model.pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'] for x in batched_inputs]
        elif 'targets' in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [x['targets'] for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)
        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
        else:
            results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(pred_class_logits, pred_anchor_deltas, self.num_classes)
        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)
        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        loss_cls = sigmoid_focal_loss_jit(pred_class_logits[valid_idxs], gt_classes_target[valid_idxs], alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction='sum') / max(1, num_foreground)
        loss_box_reg = smooth_l1_loss(pred_anchor_deltas[foreground_idxs], gt_anchors_deltas[foreground_idxs], beta=self.smooth_l1_loss_beta, reduction='sum') / max(1, num_foreground)
        return {'loss_cls': loss_cls, 'loss_box_reg': loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
            has_gt = len(targets_per_image) > 0
            if has_gt:
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(anchors_per_image.tensor, matched_gt_boxes.tensor)
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                gt_classes_i[anchor_labels == 0] = self.num_classes
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)
            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(image_sizes)
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(box_cls_per_image, box_reg_per_image, anchors_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            box_cls_i = box_cls_i.flatten().sigmoid_()
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
        keep = generalized_batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold, nms_type=self.nms_type)
        keep = keep[:self.max_detections_per_image]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'] for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, 'inference mode with training=True'
        assert len(batched_inputs) == 1, 'inference image number > 1'
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)
        results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class SepHead(nn.Module):

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, bn=None, init_bias=-2.19):
        super().__init__()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if bn is not None:
                    fc.add(get_norm(bn, head_conv))
                fc.add(nn.ReLU())
            fc.add(nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        weight_init.kaiming_init(m)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


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


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version. Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        """
        Arguments:
            out, target: B x C x H x W
            ind, mask: B x M
            cat (category id for peaks): B x M
        """
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()
        pos_pred_pix = _transpose_and_gather_feat(out, ind)
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
        """
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)
        loss = F.l1_loss(pred * mask, target * mask, reduction='none')
        loss = loss / (mask.sum() + 0.0001)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    keep = torch.from_numpy(keep).long()
    return keep


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            try:
                dict_[k] = v[inds]
            except IndexError:
                dict_[k] = v[inds[len(v)]]


def collate(batch_list, device):
    targets_merged = collections.defaultdict(list)
    for targets in batch_list:
        for target in targets:
            for k, v in target.items():
                targets_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    for key, elems in targets_merged.items():
        if key in ['voxels', 'num_points_per_voxel', 'num_voxels']:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in ['gt_boxes', 'labels', 'gt_names', 'difficulty', 'num_points_in_gt']:
            max_gt = -1
            for k in range(batch_size):
                max_gt = max(max_gt, len(elems[k]))
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, *elems[0].shape[1:]), dtype=elems[0].dtype)
            for i in range(batch_size):
                batch_gt_boxes3d[i, :len(elems[i])] = elems[i]
            if key != 'gt_names':
                batch_gt_boxes3d = torch.tensor(batch_gt_boxes3d, device=device)
            ret[key] = batch_gt_boxes3d
        elif key == 'calib':
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ['coordinates', 'points']:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def flatten(box):
    return np.concatenate(box, axis=0)


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def merge_multi_group_label(gt_classes, num_classes_by_task):
    num_task = len(gt_classes)
    flag = 0
    for i in range(num_task):
        gt_classes[i] += flag
        flag += num_classes_by_task[i]
    return flatten(gt_classes)


class VoxelNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(VoxelNet, self).__init__()
        self.config = config
        self.device = torch.device(config.model.device)
        self.reader = VoxelMeanFeatureExtractor(**config.model.reader)
        self.backbone = SpMiddleResNetFHD(**config.model.backbone)
        self.neck = RPN(config.model.neck)
        self.center_head = CenterHead(config)
        assigner_config = config.model.loss
        self.out_size_factor = assigner_config.out_size_factor
        self.tasks = config.model.head.tasks
        self.gaussian_overlap = assigner_config.gaussian_overlap
        self._max_objs = assigner_config.max_objs
        self._min_radius = assigner_config.min_radius
        self.class_names_plain = list(itertools.chain(*[t['class_names'] for t in self.tasks]))
        self

    def assign_one(self, datas, info, data_id):
        max_objs = self._max_objs
        class_names_by_task = [t['class_names'] for t in self.tasks]
        num_classes_by_task = [t['num_classes'] for t in self.tasks]
        grid_size = datas['shape'][data_id]
        pc_range = datas['range'][data_id]
        voxel_size = datas['size'][data_id]
        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}
        gt_dict = info['annotations']
        gt_boxes_mask = np.array([(n in self.class_names_plain) for n in gt_dict['gt_names']], dtype=np.bool_)
        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array([(self.class_names_plain.index(n) + 1) for n in gt_dict['gt_names']], dtype=np.int32)
        gt_dict['gt_classes'] = gt_classes
        task_masks = []
        flag = 0
        for class_name in class_names_by_task:
            task_masks.append([np.where(gt_dict['gt_classes'] == class_name.index(i) + 1 + flag) for i in class_name])
            flag += len(class_name)
        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict['gt_boxes'][m])
                task_class.append(gt_dict['gt_classes'][m] - flag2)
                task_name.append(gt_dict['gt_names'][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)
        for task_box in task_boxes:
            task_box[:, -1] = box_ops.limit_period(task_box[:, -1], offset=0.5, period=np.pi * 2)
        gt_dict['gt_classes'] = task_classes
        gt_dict['gt_names'] = task_names
        gt_dict['gt_boxes'] = task_boxes
        draw_gaussian = draw_umich_gaussian
        hms, anno_boxs, inds, masks, cats = [], [], [], [], []
        for idx, task in enumerate(self.tasks):
            hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]), dtype=np.float32)
            anno_box = np.zeros((max_objs, 10), dtype=np.float32)
            ind = np.zeros(max_objs, dtype=np.int64)
            mask = np.zeros(max_objs, dtype=np.uint8)
            cat = np.zeros(max_objs, dtype=np.int64)
            num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)
            for k in range(num_objs):
                cls_id = gt_dict['gt_classes'][idx][k] - 1
                L, W = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4]
                L, W = L / voxel_size[0] / self.out_size_factor, W / voxel_size[1] / self.out_size_factor
                if L > 0 and W > 0:
                    radius = gaussian_radius((L, W), min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))
                    x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], gt_dict['gt_boxes'][idx][k][2]
                    coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, (y - pc_range[1]) / voxel_size[1] / self.out_size_factor
                    ct = np.array([coor_x, coor_y], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue
                    draw_gaussian(hm[cls_id], ct, radius)
                    new_idx = k
                    x, y = ct_int[0], ct_int[1]
                    cat[new_idx] = cls_id
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                    rot = gt_dict['gt_boxes'][idx][k][-1]
                    anno_box[new_idx] = np.concatenate((ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]), np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
            hms.append(hm)
            anno_boxs.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            cats.append(cat)
        boxes = flatten(gt_dict['gt_boxes'])
        classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)
        gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
        boxes_and_cls = np.concatenate((boxes, classes.reshape(-1, 1).astype(np.float32)), axis=1)
        num_obj = len(boxes_and_cls)
        assert num_obj <= max_objs
        boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
        gt_boxes_and_cls[:num_obj] = boxes_and_cls
        example.update({'gt_boxes_and_cls': gt_boxes_and_cls, 'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        return example

    def label_assign(self, datas, infos):
        targets_list = []
        for data_id, info in enumerate(infos):
            example = self.assign_one(datas, info, data_id)
            targets_list.append(example)
        return targets_list

    def forward(self, batched_inputs):
        """
        Data:   dict_keys(['voxels', 'coordinates', 'num_points_per_voxel', 'num_voxels', 'shape'])
        Infos:  dict_keys(['image', 'point_cloud', 'calib', 'annotations', 'root_path'])
        """
        with torch.no_grad():
            data = [bi[0] for bi in batched_inputs]
            infos = [bi[1] for bi in batched_inputs]
            datas = collate(data, self.device)
            voxels = datas['voxels']
            coordinates = datas['coordinates']
            num_points_in_voxel = datas['num_points_per_voxel']
            num_voxels = datas['num_voxels']
            input_shape = datas['shape'][0]
            batch_size = len(num_voxels)
        input_features = self.reader(voxels, num_points_in_voxel)
        x = self.backbone(input_features, coordinates, batch_size, input_shape)
        x = self.neck(x)
        preds = self.center_head(x)
        if self.training:
            with torch.no_grad():
                targets_list = self.label_assign(datas, infos)
                targets = collate(targets_list, self.device)
                targets.update(datas)
            return self.center_head.loss(targets, preds)
        else:
            return self.center_head.predict(datas, preds, self.config.model.post_process)


def get_world_size() ->int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricBase:
    """Base class to be inherited by all metrics registered to Pythia. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, params={}):
        self.name = name
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def calculate(self, output, target, *args, **kwargs):
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            metric = self.calculate(*args, **kwargs) / self.iter_per_update
            output = {self.name: metric}
            output = reduce_dict(output)
        return output


class Accuracy(MetricBase):

    def __init__(self, iter_per_update=1):
        defaults = dict(iter_per_update=iter_per_update)
        super().__init__('accuracy', defaults)

    def calculate(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return torch.zeros([], device=output.device)
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for src, _ in indices])
    return batch_idx, src_idx


class ClassificationLoss(nn.Module):

    def __init__(self, focal_alpha):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None

    def forward(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_onehot = torch.zeros_like(src_logits)
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        self.target_classes = target_classes_o
        if 'topk_indexes' in outputs.keys():
            topk_indexes = outputs['topk_indexes']
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx]
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1
        else:
            self.src_logits = src_logits[idx]
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=self.focal_alpha, gamma=2.0, reduction='sum') / num_boxes
        losses = {'loss_ce': loss_ce}
        return losses


def box_cxcyczlwh_to_xyxyxy(x):
    x_c, y_c, z_c, l, w, h = x.unbind(-1)
    b = [x_c - 0.5 * l, y_c - 0.5 * w, z_c - 0.5 * h, x_c + 0.5 * l, y_c + 0.5 * w, z_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_intersect_wo_angle(boxes1, boxes2):
    ltb = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    rbf = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])
    lwh = (rbf - ltb).clamp(min=0)
    inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]
    return inter


def box_vol_wo_angle(boxes):
    vol = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])
    return vol


def box_iou_wo_angle(boxes1, boxes2):
    vol1 = box_vol_wo_angle(boxes1)
    vol2 = box_vol_wo_angle(boxes2)
    inter = box_intersect_wo_angle(boxes1, boxes2)
    union = vol1[:, None] + vol2 - inter
    iou = inter / union
    return iou, union


def generalized_box3d_iou(boxes1, boxes2):
    boxes1 = torch.nan_to_num(boxes1)
    boxes2 = torch.nan_to_num(boxes2)
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou_wo_angle(boxes1, boxes2)
    ltb = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rbf = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])
    whl = (rbf - ltb).clamp(min=0)
    vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]
    return iou - (vol - union) / vol


class RegressionLoss(nn.Module):

    def forward(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = _get_src_permutation_idx(indices)
        if 'topk_indexes' in outputs.keys():
            pred_boxes = torch.gather(outputs['pred_boxes'], 1, outputs['topk_indexes'].expand(-1, -1, outputs['pred_boxes'].shape[-1]))
        else:
            pred_boxes = outputs['pred_boxes']
        target_boxes = torch.cat([t['gt_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes, src_rads = pred_boxes[idx].split(6, dim=-1)
        target_boxes, target_rads = target_boxes.split(6, dim=-1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_rad = F.l1_loss(src_rads, target_rads, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box3d_iou(box_cxcyczlwh_to_xyxyxy(src_boxes), box_cxcyczlwh_to_xyxyxy(target_boxes)))
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes, 'loss_rad': loss_rad.sum() / num_boxes}
        return losses


class Det3DLoss(nn.Module):

    def __init__(self, matcher, weight_dict, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.det3d_losses = nn.ModuleDict()
        self.det3d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == 'boxes':
                self.det3d_losses[loss] = RegressionLoss()
                self.det3d_enc_losses[loss + '_enc'] = RegressionLoss()
            elif loss == 'focal_labels':
                self.det3d_losses[loss] = ClassificationLoss(0.25)
                self.det3d_enc_losses[loss + '_enc'] = ClassificationLoss(0.25)
            else:
                raise ValueError(f'Only boxes|focal_labels are supported for det3d losses. Found {loss}')

    def get_target_classes(self):
        for k in self.det3d_losses.keys():
            if 'labels' in k:
                return self.det3d_losses[k].src_logits, self.det3d_losses[k].target_classes

    def forward(self, outputs, targets):
        num_boxes = sum([len(t['labels']) for t in targets])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.det3d_losses[loss](aux_outputs, targets, indices, num_boxes)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        indices = self.matcher(outputs, targets)
        for loss in self.losses:
            losses.update(self.det3d_losses[loss](outputs, targets, indices, num_boxes))
        return losses


class HungarianMatcher3d(nn.Module):

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=1, cost_giou: 'float'=1, cost_rad: 'float'=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rad = cost_rad
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        if 'topk_indexes' in outputs.keys():
            pred_logits = torch.gather(outputs['pred_logits'], 1, outputs['topk_indexes'].expand(-1, -1, outputs['pred_logits'].shape[-1]))
            pred_boxes = torch.gather(outputs['pred_boxes'], 1, outputs['topk_indexes'].expand(-1, -1, outputs['pred_boxes'].shape[-1]))
        else:
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
        bs, num_queries = pred_logits.shape[:2]
        out_prob = pred_logits.sigmoid()
        out_bbox, out_rad = pred_boxes.split(6, dim=-1)
        tgt_ids = [v['labels'] for v in targets]
        tgt_bbox = [v['gt_boxes'][..., :6] for v in targets]
        tgt_rad = [v['gt_boxes'][..., 6:] for v in targets]
        alpha = 0.25
        gamma = 2.0
        C = []
        for i in range(bs):
            with torch.amp.autocast(enabled=False):
                out_prob_i = out_prob[i].float()
                out_bbox_i = out_bbox[i].float()
                out_rad_i = out_rad[i].float()
                tgt_bbox_i = tgt_bbox[i].float()
                tgt_rad_i = tgt_rad[i].float()
                cost_giou = -generalized_box3d_iou(box_cxcyczlwh_to_xyxyxy(out_bbox[i]), box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]))
                neg_cost_class = (1 - alpha) * out_prob_i ** gamma * -(1 - out_prob_i + 1e-08).log()
                pos_cost_class = alpha * (1 - out_prob_i) ** gamma * -(out_prob_i + 1e-08).log()
                cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]]
                cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)
                cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)
            C_i = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_rad * cost_rad
            C_i = C_i.view(num_queries, -1).cpu()
            C.append(C_i)
        indices = [linear_sum_assignment(c) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def extra_repr(self):
        s = 'cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, cost_rad={cost_rad}'
        return s.format(**self.__dict__)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-05):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class Det3DHead(nn.Module):

    def __init__(self, config, with_aux=False, with_metrics=False, num_classes=3, num_layers=1):
        super().__init__()
        hidden_dim = config.model.hidden_dim
        class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.layers[-1].bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        self.class_embed = get_clones(class_embed, num_layers)
        self.bbox_embed = get_clones(bbox_embed, num_layers)
        matcher_config = config.model.loss.matcher
        matcher = HungarianMatcher3d(cost_class=matcher_config.class_weight, cost_bbox=matcher_config.bbox_weight, cost_giou=matcher_config.giou_weight, cost_rad=matcher_config.rad_weight)
        weight_dict = {'loss_ce': config.model.loss.class_loss_coef, 'loss_bbox': config.model.loss.bbox_loss_coef, 'loss_giou': config.model.loss.giou_loss_coef, 'loss_rad': config.model.loss.rad_loss_coef}
        losses = ['focal_labels', 'boxes']
        self.losses = Det3DLoss(matcher=matcher, weight_dict=weight_dict, losses=losses)
        if with_aux:
            aux_weight_dict = {}
            num_layers = config.model.transformer.dec_layers
            if hasattr(self.losses, 'weight_dict'):
                aux_weight_dict.update({(k + '_enc_0'): v for k, v in self.losses.weight_dict.items()})
                for i in range(num_layers - 1):
                    aux_weight_dict.update({(k + f'_{i}'): v for k, v in self.losses.weight_dict.items()})
                self.losses.weight_dict.update(aux_weight_dict)
        if with_metrics:
            if not isinstance(config.model.metrics, collections.abc.Sequence):
                metrics = config.model.metrics,
            else:
                metrics = config.model.metrics
            module_metrics = {}
            for metric in metrics:
                module_metric = Accuracy(**metric['params'])
                module_metrics[metric['type']] = module_metric
            self.metrics = module_metrics
        self.config = config

    def forward(self, embed, anchors, layer_idx=0):
        cls_logits = self.class_embed[layer_idx](embed)
        box_coords = (self.bbox_embed[layer_idx](embed) + inverse_sigmoid(anchors)).sigmoid()
        return cls_logits, box_coords

    def compute_losses(self, outputs, targets):
        loss_dict = self.losses(outputs, targets)
        weight_dict = self.losses.weight_dict
        for k, v in loss_dict.items():
            if k in weight_dict:
                loss_dict[k] = v * weight_dict[k]
        if hasattr(self, 'metrics'):
            for name, metric in self.metrics.items():
                if name == 'accuracy':
                    loss_dict.update(metric(*self.losses.get_target_classes()))
                else:
                    loss_dict.update(metric(outputs, targets))
        return loss_dict


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = 'Positional encoding ' + self.__class__.__name__
        body = ['num_pos_feats: {}'.format(self.num_pos_feats), 'temperature: {}'.format(self.temperature), 'normalize: {}'.format(self.normalize), 'scale: {}'.format(self.scale)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def build_position_encoding(position_embedding, hidden_dim):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f'not supported {position_embedding}')
    return position_embedding


class Backbone3d(nn.Module):

    def __init__(self, hidden_dim, reader, extractor, position_encoding, out_features=[]):
        super(Backbone3d, self).__init__()
        self.reader = reader
        self.extractor = extractor
        self.position_encoding = build_position_encoding(position_encoding, hidden_dim)
        self.out_features = out_features
        self.num_channels = [extractor.out_channels] * len(out_features)

    def forward(self, voxels, coordinates, num_points_per_voxel, batch_size, input_shape):
        encoded_input = self.reader(voxels, num_points_per_voxel, coordinates)
        backbone_features = self.extractor(encoded_input, coordinates, batch_size, input_shape)
        outputs = []
        for of in self.out_features:
            out = backbone_features[of]
            pos = self.position_encoding(out).type_as(out)
            outputs.append((out, pos))
        return outputs


class BoxAttnFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _C.box_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @custom_bwd
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.box_attn_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class Box3dAttention(nn.Module):

    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=5):
        super(Box3dAttention, self).__init__()
        assert d_model % num_head == 0, 'd_model should be divided by num_head'
        num_variable = 5 if with_rotation else 4
        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.with_rotation = with_rotation
        self.num_variable = num_variable
        self.kernel_size = kernel_size
        self.num_point = kernel_size ** 2
        self.linear_box_weight = nn.Parameter(torch.zeros(num_level * num_head * num_variable, d_model))
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * num_variable))
        self.linear_attn_weight = nn.Parameter(torch.zeros(num_head * num_level * self.num_point, d_model))
        self.linear_attn_bias = nn.Parameter(torch.zeros(num_head * num_level * self.num_point))
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self._create_kernel_indices(kernel_size, 'kernel_indices')
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2
            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2
            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing='ij')
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        B, L = ref_windows.shape[:2]
        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(B, L, self.num_head, self.num_level, self.num_variable)
        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)
        ref_boxes = ref_windows[..., [0, 1, 3, 4]]
        ref_angles = ref_windows[..., [6]]
        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(B, L, self.num_head, self.num_level, 1)
        boxes = ref_boxes + offset_boxes / 8 * ref_boxes[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)
        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot_matrix = rot_matrix.view(B, L, self.num_head, self.num_level, 1, 2, 2)
        grid = self.kernel_indices * torch.relu(size)
        grid = center + (grid.unsqueeze(-2) * rot_matrix).sum(-1)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios
        return grid.contiguous()

    def forward(self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows):
        B, LQ = query.shape[:2]
        LV = value.shape[1]
        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(B, LV, self.num_head, self.head_dim)
        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(B, LQ, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(B, LQ, self.num_head, self.num_level, self.kernel_size, self.kernel_size)
        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step)
        output = self.out_proj(output)
        return output, attn_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
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
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, token, src, pos=None):
        token_list = []
        output = src
        for layer in self.layers:
            output, token = layer(token, output, pos=pos)
            token_list.append(token)
        if self.norm is not None:
            output = self.norm(output)
        return token_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, token, src, pos=None):
        src_mix = self.point_attn(query=src.permute(1, 0, 2), key=src.permute(1, 0, 2), value=src.permute(1, 0, 2))[0]
        src_mix = src_mix.permute(1, 0, 2)
        src = src + self.dropout1(src_mix)
        src = self.norm1(src)
        src_mix = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src_mix)
        src = self.norm2(src)
        src_summary = self.self_attn(token.permute(1, 0, 2), key=src.permute(1, 0, 2), value=src.permute(1, 0, 2))[0]
        src_summary = src_summary.permute(1, 0, 2)
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        return src, token

    def forward(self, token, src, pos=None):
        return self.forward_post(token, src, pos)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class BoxCoder:
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, **kwargs):
        return self._encode(boxes, **kwargs)

    def decode(self, rel_codes, **kwargs):
        return self._decode(rel_codes, **kwargs)

    @abstractmethod
    def _encode(self, boxes, **kwargs):
        pass

    @abstractmethod
    def _decode(self, rel_codes, **kwargs):
        pass


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.             Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.
    Returns:
        torch.Tensor: Value in the range of             [-offset * period, (1-offset) * period)
    """
    if isinstance(val, torch.Tensor):
        is_tensor = True
    elif isinstance(val, np.ndarray):
        is_tensor = False
        val = torch.from_numpy(val).float()
    else:
        raise ValueError('Only support tensor or ndarray!')
    val = val - torch.floor(val / period + offset) * period
    if not ((val >= -offset * period) & (val <= offset * period)).all().item():
        val = torch.clamp(val, min=-offset * period, max=offset * period)
    return val if is_tensor else val.numpy()


def normalize_period(x, offset, period):
    return (x + offset * period) / period


class VoxelBoxCoder3D(BoxCoder):

    def __init__(self, voxel_size, pc_range, n_dim=7, device=torch.device('cpu'), **opts):
        self.device = device
        self.voxel_size = torch.tensor(voxel_size, device=device)
        self.pc_range = torch.tensor(pc_range, device=device)
        self.pc_size = self.pc_range[3:] - self.pc_range[:3]
        self.z_normalizer = 10.0
        self.grid_size = self.pc_size.div(self.voxel_size, rounding_mode='trunc')
        self.n_dim = n_dim
        for k, v in opts.items():
            setattr(self, k, v)

    @property
    def code_size(self):
        return self.n_dim

    def _encode(self, target):
        target['labels'] -= 1
        target['gt_boxes'][:, :2] -= self.pc_range[:2]
        target['gt_boxes'][:, :2] /= self.pc_size[:2]
        target['gt_boxes'][:, 2] -= -1 * self.z_normalizer
        target['gt_boxes'][:, 2] /= 2 * self.z_normalizer
        target['gt_boxes'][:, 3:5] /= self.pc_size[:2]
        target['gt_boxes'][:, 5] /= 2 * self.z_normalizer
        target['gt_boxes'][:, -1] = limit_period(target['gt_boxes'][:, -1], offset=0.5, period=np.pi * 2)
        target['gt_boxes'] = target['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]]
        target['gt_boxes'][:, -1] = normalize_period(target['gt_boxes'][:, -1], offset=0.5, period=np.pi * 2)
        assert ((target['gt_boxes'] >= 0) & (target['gt_boxes'] <= 1)).all().item()
        return target

    def _decode(self, pred_boxes):
        pred_boxes[..., :2] = pred_boxes[..., :2] * self.pc_size[:2] + self.pc_range[:2]
        pred_boxes[..., 2] = pred_boxes[..., 2] * 2 * self.z_normalizer + -1 * self.z_normalizer
        pred_boxes[..., 3:5] = pred_boxes[..., 3:5] * self.pc_size[:2]
        pred_boxes[..., 5] = pred_boxes[..., 5] * 2 * self.z_normalizer
        pred_boxes[..., -1] = pred_boxes[..., -1] * np.pi * 2 - np.pi
        return pred_boxes


def build_sparse_resnet_backbone(config, in_channels):
    depth = config.depth
    stem_width = {(18): 16, '18b': 24, '18c': 32, (34): 16, '34b': 24, '34c': 32, (50): 16}[depth]
    norm = config.norm
    if not isinstance(norm, str):
        norm = OmegaConf.to_container(norm)
    activation = config.activation
    stem = SparseBasicStem(in_channels=in_channels, out_channels=config.stem_out_channels, norm=norm, activation=activation, stem_width=stem_width, indice_key='stem')
    out_features = config.out_features
    num_groups = config.num_groups
    width_per_group = config.width_per_group
    bottleneck_channels = num_groups * width_per_group
    in_channels = config.stem_out_channels
    out_channels = config.res1_out_channels
    num_blocks_per_stage = {(18): [2, 2, 2, 2], '18b': [2, 2, 2, 2], '18c': [2, 2, 2, 2], (34): [3, 4, 6, 3], '34b': [3, 4, 6, 3], '34c': [3, 4, 6, 3], (50): [3, 4, 6, 3]}[depth]
    out_stage_idx = [{'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5, 'linear': 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    stages = []
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        first_stride = 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'first_stride': first_stride, 'in_channels': in_channels, 'out_channels': out_channels, 'norm': norm, 'activation': activation, 'indice_key': 'res' + str(stage_idx)}
        if depth in [18, '18b', '18c', 21, 34]:
            stage_kargs['block_class'] = SparseBasicResBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['block_class'] = SparseBottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return SparseResNet(stem, stages, out_features=out_features, norm=norm)


def build_resnet_fpn_backbone(config, input_shape):
    """
    Args:
        config (OmegaConf): an efg config instance
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_sparse_resnet_backbone(config.resnet, input_shape)
    in_features = config.fpn.in_features
    out_channels = config.fpn.out_channels
    backbone = FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, norm=config.fpn.norm, top_block=LastLevelMaxPool(in_feature=config.fpn.top_block_in_feature), fuse_type=config.fpn.fuse_type)
    return backbone


class VoxelDETR(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.model.device)
        self.hidden_dim = config.model.hidden_dim
        self.aux_loss = config.model.aux_loss
        self.num_classes = len(config.dataset.classes)
        self.num_queries = config.model.transformer.num_queries
        input_dim = len(config.dataset.format) if config.dataset.nsweeps == 1 else len(config.dataset.format) + 1
        reader = VoxelMeanFeatureExtractor(**config.model.backbone.reader, num_input_features=input_dim)
        extractor = build_resnet_fpn_backbone(config.model.backbone.extractor, input_dim)
        self.backbone = Backbone3d(config.model.backbone.hidden_dim, reader, extractor, config.model.backbone.position_encoding, out_features=config.model.backbone.out_features)
        in_channels = self.backbone.num_channels
        self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels[i], self.hidden_dim, kernel_size=1), nn.GroupNorm(32, self.hidden_dim)) for i in range(len(self.backbone.out_features))])
        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.transformer = Transformer(d_model=config.model.transformer.hidden_dim, nhead=config.model.transformer.nhead, nlevel=len(config.model.backbone.out_features), num_encoder_layers=config.model.transformer.enc_layers, num_decoder_layers=config.model.transformer.dec_layers, dim_feedforward=config.model.transformer.dim_feedforward, dropout=config.model.transformer.dropout, num_queries=config.model.transformer.num_queries)
        self.transformer.proposal_head = Det3DHead(config, with_aux=False, with_metrics=False, num_classes=1, num_layers=1)
        self.transformer.decoder.detection_head = Det3DHead(config, with_aux=True, with_metrics=True, num_classes=len(config.dataset.classes), num_layers=config.model.transformer.dec_layers)
        self.box_coder = VoxelBoxCoder3D(config.dataset.voxel_size, config.dataset.pc_range, device=self.device)
        self.config = config
        self

    def forward(self, batched_inputs):
        batch_size = len(batched_inputs)
        samples = collate([bi[0] for bi in batched_inputs], self.device)
        if self.training:
            targets = [bi[1]['annotations'] for bi in batched_inputs]
            for key in ['gt_boxes', 'difficulty', 'num_points_in_gt', 'labels']:
                for i in range(batch_size):
                    targets[i][key] = torch.tensor(targets[i][key], device=self.device)
            targets = [self.box_coder.encode(tgt) for tgt in targets]
        else:
            targets = None
        voxels, coords, num_points_per_voxel, input_shape = samples['voxels'], samples['coordinates'], samples['num_points_per_voxel'], samples['shape'][0]
        ms_backbone_features_with_pos_embed = self.backbone(voxels, coords, num_points_per_voxel, batch_size, input_shape)
        features = []
        pos_encodings = []
        for idx, feat_pos in enumerate(ms_backbone_features_with_pos_embed):
            features.append(self.input_proj[idx](feat_pos[0]))
            pos_encodings.append(feat_pos[1])
        outputs = self.transformer(features, pos_encodings)
        hidden_state, init_reference, inter_references, src_embed, src_ref_windows, src_indexes = outputs
        outputs_classes = []
        outputs_coords = []
        for idx in range(hidden_state.shape[0]):
            if idx == 0:
                reference = init_reference
            else:
                reference = inter_references[idx - 1]
            outputs_class, outputs_coord = self.transformer.decoder.detection_head(hidden_state[idx], reference, idx)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.training:
            losses = {}
            enc_class, enc_coords = self.transformer.proposal_head(src_embed, src_ref_windows)
            bin_targets = copy.deepcopy(targets)
            [tgt['labels'].fill_(0) for tgt in bin_targets]
            enc_outputs = {'topk_indexes': src_indexes, 'pred_logits': enc_class, 'pred_boxes': enc_coords}
            enc_losses = self.transformer.proposal_head.compute_losses(enc_outputs, bin_targets)
            losses.update({(k + '_enc'): v for k, v in enc_losses.items()})
            outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'aux_outputs': self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])}
            dec_losses = self.transformer.decoder.detection_head.compute_losses(outputs, targets)
            losses.update(dec_losses)
            return losses
        else:
            out_logits = outputs_class[-1]
            out_bbox = outputs_coord[-1]
            out_prob = out_logits.sigmoid()
            out_prob = out_prob.view(out_logits.shape[0], -1)
            out_bbox = self.box_coder.decode(out_bbox)

            def _process_output(indices, bboxes):
                topk_boxes = indices.div(out_logits.shape[2], rounding_mode='floor')
                labels = indices % out_logits.shape[2]
                boxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_bbox.shape[-1]))
                return labels + 1, boxes, topk_boxes
            scores, topk_indices = torch.topk(out_prob, 300, dim=1, sorted=False)
            labels, boxes, topk_indices = _process_output(topk_indices.view(1, -1), out_bbox)
            results = [{'scores': s.detach().cpu(), 'labels': l.detach().cpu(), 'boxes3d': b.detach().cpu()} for s, l, b in zip(scores, labels, boxes)]
            return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs['res{}'.format(i + 2)] = out
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class D2SwinTransformer(SwinTransformer, Backbone):

    def __init__(self, config, input_shape):
        pretrain_img_size = config.MODEL.SWIN.PRETRAIN_IMG_SIZE
        patch_size = config.MODEL.SWIN.PATCH_SIZE
        in_chans = 3
        embed_dim = config.MODEL.SWIN.EMBED_DIM
        depths = config.MODEL.SWIN.DEPTHS
        num_heads = config.MODEL.SWIN.NUM_HEADS
        window_size = config.MODEL.SWIN.WINDOW_SIZE
        mlp_ratio = config.MODEL.SWIN.MLP_RATIO
        qkv_bias = config.MODEL.SWIN.QKV_BIAS
        qk_scale = config.MODEL.SWIN.QK_SCALE
        drop_rate = config.MODEL.SWIN.DROP_RATE
        attn_drop_rate = config.MODEL.SWIN.ATTN_DROP_RATE
        drop_path_rate = config.MODEL.SWIN.DROP_PATH_RATE
        norm_layer = nn.LayerNorm
        ape = config.MODEL.SWIN.APE
        patch_norm = config.MODEL.SWIN.PATCH_NORM
        use_checkpoint = config.MODEL.SWIN.USE_CHECKPOINT
        super().__init__(pretrain_img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint=use_checkpoint)
        self._out_features = config.MODEL.SWIN.OUT_FEATURES
        self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        self._out_feature_channels = {'res2': self.num_features[0], 'res3': self.num_features[1], 'res4': self.num_features[2], 'res5': self.num_features[3]}

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f'SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!'
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @property
    def size_divisibility(self):
        return 32


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -torch.abs(gt_class_logits)


def dice_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor', num_masks: 'float'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element
            in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat([point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
    return point_coords


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class NestedTensor(object):

    def __init__(self, tensors, mask: 'Optional[Tensor]'):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]') ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]'):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor', num_masks: 'float'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(src_masks, lambda logits: calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels, num_masks), 'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks)}
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.loss_labels, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = ['matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)), 'losses: {}'.format(self.losses), 'weight_dict: {}'.format(self.weight_dict), 'num_classes: {}'.format(self.num_classes), 'eos_coef: {}'.format(self.eos_coef), 'num_points: {}'.format(self.num_points), 'oversample_ratio: {}'.format(self.oversample_ratio), 'importance_sample_ratio: {}'.format(self.importance_sample_ratio)]
        _repr_indent = 4
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        conv_dim = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        norm = config.MODEL.SEM_SEG_HEAD.NORM
        transformer_dropout = config.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = config.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = config.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        transformer_in_features = config.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        common_stride = config.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim)))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads, dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers, num_feature_levels=self.transformer_num_feature_levels)
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.mask_features)
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        use_bias = norm == ''
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)
            lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward_features(self, features):
        srcs = []
        pos = []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features


def build_pixel_decoder(config, input_shape):
    """
    Build a pixel decoder from `config.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = config.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = MSDeformAttnPixelDecoder(config, input_shape)
    forward_features = getattr(model, 'forward_features', None)
    if not callable(forward_features):
        raise ValueError(f'Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. Please implement forward_features for {name} to only return mask features.')
    return model


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if 'static_query' in k:
                    newk = k.replace('static_query', 'query_feat')
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(f'Weight format of {self.__class__.__name__} have changed! Please upgrade your models. Applying automatic conversion now ...')

    def __init__(self, config, in_channels, mask_classification=True):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        in_channels = in_channels
        mask_classification = mask_classification
        num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = config.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = config.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = config.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = config.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        assert config.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        dec_layers = config.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = config.MODEL.MASK_FORMER.PRE_NORM
        enforce_input_project = config.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        mask_dim = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        assert mask_classification, 'Only support mask classification model'
        self.mask_classification = mask_classification
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_ffn_layers.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm))
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        predictions_class = []
        predictions_mask = []
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](output, src[level_index], memory_mask=attn_mask, memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        assert len(predictions_class) == self.num_layers + 1
        out = {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 'aux_outputs': self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask)}
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode='bilinear', align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]


def build_transformer_decoder(config, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `config.MODEL.INS_EMBED_HEAD.NAME`.
    """
    return MultiScaleMaskedTransformerDecoder(config, in_channels, mask_classification)


class MaskFormerHead(nn.Module):
    _version = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if 'sem_seg_head' in k and not k.startswith(prefix + 'predictor'):
                    newk = k.replace(prefix, prefix + 'pixel_decoder.')
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(f'Weight format of {self.__class__.__name__} have changed! Please upgrade your models. Applying automatic conversion now ...')

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        if config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'transformer_encoder':
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'pixel_embedding':
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'multi_scale_pixel_decoder':
            transformer_predictor_in_channels = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        sorted_input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted_input_shape]
        self.ignore_value = config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.common_stride = 4
        self.loss_weight = config.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.pixel_decoder = build_pixel_decoder(config, input_shape)
        self.predictor = build_transformer_decoder(config, transformer_predictor_in_channels, mask_classification=True)
        self.transformer_in_feature = config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE
        self.num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == 'multi_scale_pixel_decoder':
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        elif self.transformer_in_feature == 'transformer_encoder':
            assert transformer_encoder_features is not None, 'Please use the TransformerEncoderPixelDecoder.'
            predictions = self.predictor(transformer_encoder_features, mask_features, mask)
        elif self.transformer_in_feature == 'pixel_embedding':
            predictions = self.predictor(mask_features, mask_features, mask)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == 'cuda' and hasattr(x, 'to')
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)
        logger = logging.getLogger(__name__)
        logger.info('Attempting to copy inputs of {} to CPU due to CUDA OOM'.format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return wrapped


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, :img_size[0], :img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(result, size=(output_height, output_width), mode='bilinear', align_corners=False)[0]
    return result


class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(self, config):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.device = torch.device(config.model.device)
        input_shape = ShapeSpec(channels=len(config.model.pixel_mean))
        backbone = build_resnet_backbone(config, input_shape)
        sem_seg_head = MaskFormerHead(config, backbone.output_shape())
        deep_supervision = config.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = config.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = config.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = config.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = config.MODEL.MASK_FORMER.MASK_WEIGHT
        matcher = HungarianMatcher(cost_class=class_weight, cost_mask=mask_weight, cost_dice=dice_weight, num_points=config.MODEL.MASK_FORMER.TRAIN_NUM_POINTS)
        weight_dict = {'loss_ce': class_weight, 'loss_mask': mask_weight, 'loss_dice': dice_weight}
        if deep_supervision:
            dec_layers = config.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({(k + f'_{i}'): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['labels', 'masks']
        criterion = SetCriterion(sem_seg_head.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses, num_points=config.MODEL.MASK_FORMER.TRAIN_NUM_POINTS, oversample_ratio=config.MODEL.MASK_FORMER.OVERSAMPLE_RATIO, importance_sample_ratio=config.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO)
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = config.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.object_mask_threshold = config.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
        self.overlap_threshold = config.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD
        size_divisibility = config.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = config.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE or config.MODEL.MASK_FORMER.TEST.PANOPTIC_ON or config.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        self.register_buffer('pixel_mean', torch.Tensor(config.model.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(config.model.pixel_std).view(-1, 1, 1), False)
        self.semantic_on = config.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        self.instance_on = config.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        self.panoptic_on = config.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
        self.test_topk_per_image = config.dataset.test.detections_per_image
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x['image'] for x in batched_inputs]
        images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        if self.training:
            if 'instances' in batched_inputs[0]:
                gt_instances = [x['instances'] for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs['pred_logits']
            mask_pred_results = outputs['pred_masks']
            mask_pred_results = F.interpolate(mask_pred_results, size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode='bilinear', align_corners=False)
            del outputs
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                processed_results.append({})
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, image_size, height, width)
                    mask_cls_result = mask_cls_result
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]['sem_seg'] = r
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]['panoptic_seg'] = panoptic_r
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]['instances'] = instance_r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({'labels': targets_per_image.gt_classes, 'masks': padded_masks})
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        current_segment_id = 0
        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.dataset_meta.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append({'id': current_segment_id, 'isthing': bool(isthing), 'category_id': int(pred_class)})
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.dataset_meta.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-06)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result


class PerPixelBaselineHead(nn.Module):
    _version = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if 'sem_seg_head' in k and not k.startswith(prefix + 'predictor'):
                    newk = k.replace(prefix, prefix + 'pixel_decoder.')
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(f'Weight format of {self.__class__.__name__} have changed! Please upgrade your models. Applying automatic conversion now ...')

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        ignore_value = config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        pixel_decoder = build_pixel_decoder(config, input_shape)
        loss_weight = config.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = pixel_decoder
        self.predictor = Conv2d(self.pixel_decoder.mask_dim, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(x, scale_factor=self.common_stride, mode='bilinear', align_corners=False)
            return x, {}

    def layers(self, features):
        x, _, _ = self.pixel_decoder.forward_features(features)
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        predictions = predictions.float()
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode='bilinear', align_corners=False)
        loss = F.cross_entropy(predictions, targets, reduction='mean', ignore_index=self.ignore_value)
        losses = {'loss_sem_seg': loss * self.loss_weight}
        return losses


class StandardTransformerDecoder(nn.Module):

    def __init__(self, config, in_channels, mask_classification=True):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        in_channels = in_channels
        mask_classification = mask_classification
        num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = config.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = config.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = config.MODEL.MASK_FORMER.NHEADS
        dropout = config.MODEL.MASK_FORMER.DROPOUT
        dim_feedforward = config.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        enc_layers = config.MODEL.MASK_FORMER.ENC_LAYERS
        dec_layers = config.MODEL.MASK_FORMER.DEC_LAYERS
        pre_norm = config.MODEL.MASK_FORMER.PRE_NORM
        deep_supervision = config.MODEL.MASK_FORMER.DEEP_SUPERVISION
        enforce_input_project = config.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        mask_dim = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        self.mask_classification = mask_classification
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        transformer = Transformer(d_model=hidden_dim, dropout=dropout, nhead=nheads, dim_feedforward=dim_feedforward, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, normalize_before=pre_norm, return_intermediate_dec=deep_supervision)
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:])[0]
        pos = self.pe_layer(x, mask)
        src = x
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {'pred_logits': outputs_class[-1]}
        else:
            out = {}
        if self.aux_loss:
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)
            out['pred_masks'] = outputs_seg_masks[-1]
            out['aux_outputs'] = self._set_aux_loss(outputs_class if self.mask_classification else None, outputs_seg_masks)
        else:
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
            out['pred_masks'] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]


class PerPixelBaselinePlusHead(PerPixelBaselineHead):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if 'sem_seg_head' in k and not k.startswith(prefix + 'predictor'):
                    newk = k.replace(prefix, prefix + 'pixel_decoder.')
                    logger.debug(f'{k} ==> {newk}')
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False
            if not scratch:
                logger.warning(f'Weight format of {self.__class__.__name__} have changed! Please upgrade your models. Applying automatic conversion now ...')

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
            deep_supervision: whether or not to add supervision to the output of
                every transformer decoder layer
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        ignore_value = config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = config.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        pixel_decoder = build_pixel_decoder(config, input_shape)
        loss_weight = config.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        transformer_in_feature = config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE
        if config.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'transformer_encoder':
            in_channels = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            in_channels = input_shape[transformer_in_feature].channels
        transformer_predictor = StandardTransformerDecoder(config, in_channels, mask_classification=False)
        deep_supervision = config.MODEL.MASK_FORMER.DEEP_SUPERVISION
        super().__init__(input_shape, num_classes=num_classes, pixel_decoder=pixel_decoder, loss_weight=loss_weight, ignore_value=ignore_value)
        del self.predictor
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        self.deep_supervision = deep_supervision

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x, aux_outputs = self.layers(features)
        if self.training:
            if self.deep_supervision:
                losses = self.losses(x, targets)
                for i, aux_output in enumerate(aux_outputs):
                    losses['loss_sem_seg' + f'_{i}'] = self.losses(aux_output['pred_masks'], targets)['loss_sem_seg']
                return None, losses
            else:
                return None, self.losses(x, targets)
        else:
            x = F.interpolate(x, scale_factor=self.common_stride, mode='bilinear', align_corners=False)
            return x, {}

    def layers(self, features):
        mask_features, transformer_encoder_features, _ = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == 'transformer_encoder':
            assert transformer_encoder_features is not None, 'Please use the TransformerEncoderPixelDecoder.'
            predictions = self.predictor(transformer_encoder_features, mask_features)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features)
        if self.deep_supervision:
            return predictions['pred_masks'], predictions['aux_outputs']
        else:
            return predictions['pred_masks'], None


class BasePixelDecoder(nn.Module):

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        conv_dim = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = config.MODEL.SEM_SEG_HEAD.MASK_DIM
        norm = config.MODEL.SEM_SEG_HEAD.NORM
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        lateral_convs = []
        output_convs = []
        use_bias = norm == ''
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(in_channels, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module('layer_{}'.format(idx + 1), output_conv)
                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)
                lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
                output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
                self.add_module('layer_{}'.format(idx + 1), output_conv)
                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        weight_init.c2_xavier_fill(self.mask_features)
        self.maskformer_num_feature_levels = 3

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode='nearest')
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning('Calling forward() may cause unpredicted behavior of PixelDecoder module.')
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoderPixelDecoder(BasePixelDecoder):

    def __init__(self, config, input_shape: 'Dict[str, ShapeSpec]'):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(config, input_shape)
        input_shape = {k: v for k, v in input_shape.items() if k in config.MODEL.SEM_SEG_HEAD.IN_FEATURES}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        conv_dim = config.MODEL.SEM_SEG_HEAD.CONVS_DIM
        norm = config.MODEL.SEM_SEG_HEAD.NORM
        transformer_dropout = config.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = config.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = config.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        transformer_enc_layers = config.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        transformer_pre_norm = config.MODEL.MASK_FORMER.PRE_NORM
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads, dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers, normalize_before=transformer_pre_norm)
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        use_bias = norm == ''
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, 'layer_{}'.format(len(self.in_features)))
        self.add_module('layer_{}'.format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode='nearest')
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), transformer_encoder_features, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning('Calling forward() may cause unpredicted behavior of PixelDecoder module.')
        return self.forward_features(features)


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
        else:
            self.code_weights = None

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

    def __init__(self, input_dim, joint_feat=False, channels=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
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
        self.init_weights()

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
        return feat, feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)
    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]
    return nn.Sequential(*layers)


class MotionEncoder(nn.Module):

    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(c_in=in_channels, mlp_channels=[hidden_dim] * num_pre_layers, ret_before_act=False)
        self.mlps = build_mlps(c_in=hidden_dim * 2, mlp_channels=[hidden_dim] * (num_layers - num_pre_layers), ret_before_act=False)
        if out_channels is not None:
            self.out_mlps = build_mlps(c_in=hidden_dim, mlp_channels=[hidden_dim, hidden_dim, out_channels], ret_before_act=True)
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid
        feature_buffers = feature_buffers.max(dim=2)[0]
        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=-1) > 0
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers


class TransformerEncoderGlobalLocal(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        token_list = []
        output = src
        for layer in self.layers:
            output = layer(output)
            token_list.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return token_list


class FFN(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, dout=None, activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoderLayerGlobalLocal(nn.Module):

    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0):
        super().__init__()
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.local_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src):
        bs, num_track, candi = src.shape[0], src.shape[1], src.shape[2]
        src_global = src.reshape(bs, -1, src.shape[-1])
        src_mix = self.global_attn(query=src_global.permute(1, 0, 2), key=src_global.permute(1, 0, 2), value=src_global.permute(1, 0, 2))[0]
        src_mix = src_mix.permute(1, 0, 2)
        src_global = self.ffn1(src_global, src_mix)
        src_local = src_global.reshape(bs, num_track, candi, -1).reshape(bs * num_track, candi, -1)
        src_mix = self.local_attn(query=src_local.permute(1, 0, 2), key=src_local.permute(1, 0, 2), value=src_local.permute(1, 0, 2))[0]
        src_mix = src_mix.permute(1, 0, 2)
        src_local = self.ffn2(src_local, src_mix)
        return src_local.reshape(bs, num_track, candi, -1)

    def forward(self, src):
        return self.forward_post(src)


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    overlaps_bev = torch.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()
    _C.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-06)
    return iou3d


def crop_current_frame_points(num_lidar_points, trajectory_rois, points):
    batch_size, traj_length, num_track, candi_length, _ = trajectory_rois.shape
    src = torch.zeros(batch_size, num_track * candi_length, num_lidar_points, 6)
    trajectory_rois = trajectory_rois.reshape(batch_size, traj_length, -1, 8)
    num_rois = num_track * candi_length
    for bs_idx in range(batch_size):
        cur_batch_boxes = trajectory_rois[bs_idx, 0, :, :7].view(-1, 7)
        cur_radiis = torch.sqrt((cur_batch_boxes[:, 3] / 2) ** 2 + (cur_batch_boxes[:, 4] / 2) ** 2) * 1.2
        cur_points = points[points[:, 0] == bs_idx][:, 1:]
        time_mask = cur_points[:, -1] < 1
        cur_points = cur_points[time_mask]
        if cur_batch_boxes.shape[0] > 16:
            length_iter = cur_batch_boxes.shape[0] // 16
            dis_list = []
            for i in range(length_iter + 1):
                dis = torch.norm(cur_points[:, :2].unsqueeze(0) - cur_batch_boxes[16 * i:16 * (i + 1), :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1), dim=2)
                dis_list.append(dis)
            dis = torch.cat(dis_list, 0)
        else:
            dis = torch.norm(cur_points[:, :2].unsqueeze(0) - cur_batch_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1), dim=2)
        point_mask = dis <= cur_radiis.unsqueeze(-1)
        for roi_box_idx in range(0, num_rois):
            cur_roi_points = cur_points[point_mask[roi_box_idx]]
            if cur_roi_points.shape[0] > num_lidar_points:
                np.random.seed(0)
                choice = np.random.choice(cur_roi_points.shape[0], num_lidar_points, replace=False)
                cur_roi_points_sample = cur_roi_points[choice]
            elif cur_roi_points.shape[0] == 0:
                add_zeros = cur_roi_points.new_zeros(num_lidar_points, 6)
                add_zeros[:, :3] = trajectory_rois[bs_idx, 0:1, roi_box_idx, :3].repeat(num_lidar_points, 1)
                cur_roi_points_sample = add_zeros
            else:
                empty_num = num_lidar_points - cur_roi_points.shape[0]
                add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim=0)
            src[bs_idx, roi_box_idx, :num_lidar_points, :] = cur_roi_points_sample
    return src


def decode_torch(box_encodings, anchors):
    encode_angle_by_sincos = False
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    if not encode_angle_by_sincos:
        xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
    else:
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
    diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza
    if encode_angle_by_sincos:
        rg_cos = cost + torch.cos(ra)
        rg_sin = sint + torch.sin(ra)
        rg = torch.atan2(rg_sin, rg_cos)
    else:
        rg = rt + ra
    cgs = [(t + a) for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


def encode_boxes_res_torch(boxes, anchors):
    """
    Args:
        boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

    Returns:

    """
    anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-05)
    boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-05)
    xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)
    diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / dza
    dxt = torch.log(dxg / dxa)
    dyt = torch.log(dyg / dya)
    dzt = torch.log(dzg / dza)
    encode_angle_by_sincos = False
    if encode_angle_by_sincos:
        rt_cos = torch.cos(rg) - torch.cos(ra)
        rt_sin = torch.sin(rg) - torch.sin(ra)
        rts = [rt_cos, rt_sin]
    else:
        rts = [rg - ra]
    cts = [(g - a) for g, a in zip(cgs, cas)]
    return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = boxes3d.new_tensor(([1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1])) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def get_corner_loss_lidar(pred_bbox3d: 'torch.Tensor', gt_bbox3d: 'torch.Tensor'):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]
    pred_box_corners = boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = boxes_to_corners_3d(gt_bbox3d)
    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = boxes_to_corners_3d(gt_bbox3d_flip)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2), torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)
    return corner_loss.mean(dim=1)


def get_corner_loss(rcnn_reg, roi_boxes3d, gt_of_rois_src, fg_mask):
    fg_rcnn_reg = rcnn_reg[fg_mask]
    fg_roi_boxes3d = roi_boxes3d[fg_mask]
    code_size = 7
    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
    batch_anchors = fg_roi_boxes3d.clone().detach()
    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
    batch_anchors[:, :, 0:3] = 0
    rcnn_boxes3d = decode_torch(fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors).view(-1, code_size)
    rcnn_boxes3d = rotate_points_along_z(rcnn_boxes3d.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
    rcnn_boxes3d[:, 0:3] += roi_xyz
    loss_corner = get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
    loss_corner = loss_corner.mean()
    return loss_corner


def get_corner_points(rois, batch_size_rcnn):
    faked_features = rois.new_ones((2, 2, 2))
    dense_idx = faked_features.nonzero()
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()
    local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
    roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) - local_roi_size.unsqueeze(dim=1) / 2
    return roi_grid_points


def get_corner_points_of_roi(rois):
    rois = rois.view(-1, rois.shape[-1])
    batch_size_rcnn = rois.shape[0]
    local_roi_grid_points = get_corner_points(rois, batch_size_rcnn)
    local_roi_grid_points = rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
    return global_roi_grid_points, local_roi_grid_points


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = _C.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out]].contiguous(), None


def reorder_rois(pred_bboxes):
    num_max_rois = max([len(bbox) for bbox in pred_bboxes])
    num_max_rois = max(1, num_max_rois)
    ordered_bboxes = torch.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]])
    valid_mask = torch.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]])
    for bs_idx in range(ordered_bboxes.shape[0]):
        ordered_bboxes[bs_idx, :len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
        valid_mask[bs_idx, :len(pred_bboxes[bs_idx])] = 1
    return ordered_bboxes, valid_mask.bool()


def spherical_coordinate(src, diag_dist):
    assert src.shape[-1] == 27
    device = src.device
    indices_x = torch.LongTensor([0, 3, 6, 9, 12, 15, 18, 21, 24])
    indices_y = torch.LongTensor([1, 4, 7, 10, 13, 16, 19, 22, 25])
    indices_z = torch.LongTensor([2, 5, 8, 11, 14, 17, 20, 23, 26])
    src_x = torch.index_select(src, -1, indices_x)
    src_y = torch.index_select(src, -1, indices_y)
    src_z = torch.index_select(src, -1, indices_z)
    dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
    phi = torch.atan(src_y / (src_x + 1e-05))
    the = torch.acos(src_z / (dis + 1e-05))
    dis = dis / (diag_dist + 1e-05)
    src = torch.cat([dis, phi, the], dim=-1)
    return src


def transform_box_to_global(pred_boxes3d, pred_vels, pose):
    expand_bboxes = np.concatenate([pred_boxes3d[:, :3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)
    expand_vels = np.concatenate([pred_vels[:, 0:2], np.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
    bboxes_global = np.dot(expand_bboxes, pose.T)[:, :3]
    vels_global = np.dot(expand_vels, pose[:3, :3].T)
    moved_bboxes_global = deepcopy(bboxes_global)
    bboxes_pre2cur = np.concatenate([moved_bboxes_global, pred_boxes3d[:, 3:7]], axis=-1)
    bboxes_pre2cur[..., -1] = bboxes_pre2cur[..., -1] + np.arctan2(pose[..., 1, 0], pose[..., 0, 0])
    return torch.tensor(bboxes_pre2cur).float(), torch.tensor(vels_global[:, :2]).float()


def transform_global_to_current_torch(pred_boxes3d, pred_vels, global_from_ref, time_lag=0):
    ref_from_global = torch.linalg.inv(global_from_ref)
    moved_bboxes_global = pred_boxes3d[:, :3].clone()
    vels_global = torch.cat([pred_vels[:, 0:2], torch.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
    if time_lag > 0:
        moved_bboxes_global[:, :2] = moved_bboxes_global[:, :2] + time_lag * vels_global[:, :2]
    expand_bboxes_global = torch.cat([moved_bboxes_global[:, :3], torch.ones((pred_boxes3d.shape[0], 1))], axis=-1)
    bboxes_global2cur = torch.mm(ref_from_global, expand_bboxes_global.t()).t()[:, :3]
    vels_global2cur = torch.mm(ref_from_global[:3, :3], vels_global.t()).t()[:, :2]
    bboxes_global2cur = torch.cat([bboxes_global2cur, pred_boxes3d[:, 3:7]], axis=-1)
    bboxes_global2cur[..., 6] = bboxes_global2cur[..., 6] - torch.atan2(global_from_ref[..., 1, 0], global_from_ref[..., 0, 0])
    return bboxes_global2cur, vels_global2cur


def transform_trajs_to_global_coords(box_seq, center_xyz, center_heading, pred_vel_repeat=None, heading_index=6):
    box_seq_local = box_seq.clone()
    batch_size, len_traj, num_track, num_candi = box_seq.shape[0], box_seq.shape[1], box_seq.shape[2], box_seq.shape[3]
    box_seq_local = rotate_points_along_z(points=box_seq_local.permute(0, 2, 3, 1, 4).reshape(batch_size * num_track * num_candi, -1, box_seq.shape[-1]), angle=center_heading.reshape(-1))
    box_seq_local = box_seq_local.reshape(batch_size, num_track, num_candi, len_traj, -1).permute(0, 3, 1, 2, 4)
    box_seq_local[:, :, :, :, 0:center_xyz.shape[-1]] = box_seq_local[:, :, :, :, 0:center_xyz.shape[-1]] + center_xyz
    box_seq_local[:, :, :, :, heading_index] = box_seq_local[:, :, :, :, heading_index] + center_heading
    if pred_vel_repeat is not None:
        local_vel = rotate_points_along_z(points=pred_vel_repeat.permute(0, 2, 3, 1, 4).reshape(batch_size * num_track * num_candi, -1, pred_vel_repeat.shape[-1]), angle=center_heading.reshape(-1))
        local_vel = local_vel.reshape(batch_size, num_track, num_candi, len_traj, -1).permute(0, 3, 1, 2, 4)
    else:
        local_vel = None
    return box_seq_local, local_vel


def transform_trajs_to_local_coords(box_seq, center_xyz, center_heading, pred_vel_hypo=None, heading_index=8, rot_vel_index=[6, 7]):
    box_seq_local = box_seq.clone()
    box_seq_local_buffer = torch.zeros_like(box_seq)
    valid_mask = torch.logical_and((center_xyz[..., :2].sum(-1) != 0).repeat(1, box_seq.shape[1], 1, 1), box_seq[..., 3:6].sum(-1) != 0)
    batch_size, len_traj, num_track, num_candi = box_seq.shape[0], box_seq.shape[1], box_seq.shape[2], box_seq.shape[3]
    box_seq_local[:, :, :, :, 0:2] = box_seq_local[:, :, :, :, 0:2] - center_xyz[..., :2]
    box_seq_local = rotate_points_along_z(points=box_seq_local.permute(0, 2, 3, 1, 4).reshape(batch_size * num_track * num_candi, -1, box_seq.shape[-1]), angle=-center_heading.reshape(-1))
    box_seq_local = box_seq_local.reshape(batch_size, num_track, num_candi, len_traj, -1).permute(0, 3, 1, 2, 4)
    box_seq_local[:, :, :, :, heading_index] = box_seq_local[:, :, :, :, heading_index] - center_heading
    if pred_vel_hypo is not None:
        local_vel_buffer = torch.zeros_like(pred_vel_hypo)
        local_vel = rotate_points_along_z(points=pred_vel_hypo.permute(0, 2, 3, 1, 4).reshape(batch_size * num_track * num_candi, -1, pred_vel_hypo.shape[-1]), angle=-center_heading.reshape(-1))
        local_vel = local_vel.reshape(batch_size, num_track, num_candi, len_traj, -1).permute(0, 3, 1, 2, 4)
        local_vel_buffer[valid_mask] = local_vel[valid_mask]
    else:
        local_vel_buffer = None
    box_seq_local_buffer[valid_mask] = box_seq_local[valid_mask]
    return box_seq_local_buffer, local_vel_buffer


class TrajectoryFormer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.model.device)
        self.config = config
        self.is_train = config.task == 'train'
        self.hidden_dim = config.model.hidden_dim
        self.seqboxembed = PointNet(config.model.boxes_dim, channels=self.hidden_dim)
        self.velboxembed = MotionEncoder(config.model.motion_input_dim, self.hidden_dim, out_channels=3 * config.model.motion_pred_frames)
        self.traj_length = config.dataset.traj_length
        self.num_lidar_points = config.model.num_lidar_points
        self.num_hypo_det = config.model.num_hypo_det
        self.num_hypo_pred = config.model.num_hypo_pred
        self.num_hypo_train = (self.num_hypo_pred + self.num_hypo_det) * 2
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=None)
        self.point_reg = MLP(self.hidden_dim, self.hidden_dim, 7, 3)
        self.joint_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.point_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.boxes_cls = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        self.cls_embed = MLP(self.hidden_dim * 2 + 3, self.hidden_dim, self.hidden_dim, 3)
        self.up_dimension_geometry = MLP(config.model.point_dim, self.hidden_dim, self.hidden_dim, 3)
        self.dist_thresh = config.model.dist_thresh
        self.token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.token_traj = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.num_encoder_layers = config.model.enc_layers
        self.dim_feedforward = config.model.dim_feedforward
        self.nhead = config.model.nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model=self.hidden_dim, nhead=self.nhead, dim_feedforward=self.dim_feedforward) for i in range(self.num_encoder_layers)]
        encoder_layer_gl = [TransformerEncoderLayerGlobalLocal(self.config, d_model=self.hidden_dim, nhead=self.nhead, dim_feedforward=self.dim_feedforward) for i in range(self.num_encoder_layers)]
        encoder_norm = None
        self.encoder_fg = TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm, self.config)
        self.encoder_globallocal = TransformerEncoderGlobalLocal(encoder_layer_gl, self.num_encoder_layers, encoder_norm, self.config)
        self.car_embed = torch.tensor([1, 0, 0]).float().reshape(1, 1, 3)
        self.ped_embed = torch.tensor([0, 1, 0]).float().reshape(1, 1, 3)
        self.cyc_embed = torch.tensor([0, 0, 1]).float().reshape(1, 1, 3)
        self.train_nms_thresh = self.config.dataset.nms_thresh
        self.train_score_thresh = self.config.dataset.score_thresh
        self.max_id = 0
        self.WAYMO_TRACKING_NAMES = config.dataset.classes
        self.nms_thresh = self.config.model.nms_thresh
        max_dist = config.model.max_dist
        self.num_hypo_inference = config.model.num_hypo_pred_eval
        self.history_traj_frames = config.model.history_frames_eval
        self.keep_thresh_car = config.model.track_score.car
        self.keep_thresh_ped = config.model.track_score.ped
        self.keep_thresh_cyc = config.model.track_score.cyc
        self.new_born_car = config.model.new_born_score.car
        self.new_born_ped = config.model.new_born_score.ped
        self.new_born_cyc = config.model.new_born_score.cyc
        self.new_born_nms_thresh = self.config.model.new_born_nms_thresh
        self
        self.tracker = Tracker(max_dist=max_dist)
        self.load_motion_module = False

    def forward(self, batched_inputs):
        if not self.load_motion_module:
            self.load_pretrain_motionencoder()
        if self.is_train:
            loss_dict = self.forward_train(batched_inputs)
            return loss_dict
        else:
            results = self.forward_inference(batched_inputs)
            return results

    def forward_train(self, batched_inputs):
        self.batch_size = len(batched_inputs)
        samples = collate([bi[0] for bi in batched_inputs], self.device)
        targets = [bi[1]['annotations'] for bi in batched_inputs]
        for key in ['gt_boxes', 'difficulty', 'num_points_in_gt', 'labels']:
            for i in range(self.batch_size):
                targets[i][key] = torch.tensor(targets[i][key], device=self.device)
        if self.is_train:
            load_boxes3d = [torch.from_numpy(bi[1]['annotations']['pred_boxes3d']) for bi in batched_inputs]
            load_scores = [torch.from_numpy(bi[1]['annotations']['pred_scores']) for bi in batched_inputs]
            load_labels = [torch.from_numpy(bi[1]['annotations']['pred_labels']) for bi in batched_inputs]
            pred_boxes3d, pred_labels, det_boxes3d, traj = self.organize_proposals(load_boxes3d, load_scores, load_labels)
            self.num_track = pred_boxes3d.shape[1]
            if self.num_track > 0 and det_boxes3d.shape[1] > 0:
                loss_dict = {}
                hypotheses_aug = self.hypotheses_augment(pred_boxes3d, targets)
                global_trajectory_hypothses, global_candidates = self.generate_trajectory_hypothses(pred_boxes3d, det_boxes3d, traj, self.num_hypo_det, hypotheses_aug)
                point_feat_list = self.get_trajcetory_point_feature(global_trajectory_hypothses, samples)
                point_cls = self.point_cls(torch.cat(point_feat_list, 0)).squeeze(-1)
                boxes_feat = self.get_trajectory_boxes_feature(global_trajectory_hypothses)
                boxes_cls = self.boxes_cls(boxes_feat).reshape(-1, self.num_hypo_train)
                hypotheses_feat = self.get_trajectory_hypotheses_feat(point_feat_list, boxes_feat, pred_labels)
                feat_list = self.encoder_globallocal(hypotheses_feat)
                joint_cls_list = []
                for i in range(self.num_encoder_layers):
                    joint_cls = self.joint_cls(feat_list[i]).squeeze(-1).reshape(-1, self.num_hypo_train)
                    joint_cls_list.append(joint_cls)
                joint_cls = torch.cat(joint_cls_list, 0)
                point_reg = self.point_reg(torch.cat(point_feat_list, 0)).reshape(3, self.batch_size * self.num_track, self.num_hypo_train, 7)
                point_reg = point_reg.reshape(1, -1, 7)
                fg_iou_mask, fg_reg_mask, ious_targets, gt_boxes = self.get_cls_targets(pred_boxes3d, global_candidates, targets)
                rois = global_candidates[..., :7].reshape(-1, 7)
                reg_targets = self.get_reg_targets(rois, gt_boxes)
                loss_cls_sum, loss_reg_sum = self.get_loss(rois, gt_boxes, point_cls, joint_cls, boxes_cls, point_reg, ious_targets, reg_targets, fg_reg_mask, fg_iou_mask)
                if gt_boxes.shape[0] > 0:
                    loss_dict.update({'loss_cls': loss_cls_sum, 'loss_reg': loss_reg_sum})
                else:
                    loss_dict = {'loss_cls': loss_cls_sum, 'loss_reg': torch.tensor([0.0]).reshape(1, -1)}
            else:
                loss_dict = {'loss_cls': torch.tensor([0.0]).reshape(1, -1), 'loss_reg': torch.tensor([0.0]).reshape(1, -1)}
            return loss_dict

    def forward_inference(self, batched_inputs):
        self.batch_size = len(batched_inputs)
        samples = collate([bi[0] for bi in batched_inputs], self.device)
        self.frame_id = int(batched_inputs[0][1]['token'].split('_frame_')[-1].split('.')[0])
        det_boxes3d = [torch.from_numpy(bi[1]['annotations']['pred_boxes3d']) for bi in batched_inputs][0][None]
        det_scores = [torch.from_numpy(bi[1]['annotations']['pred_scores']) for bi in batched_inputs][0][None]
        det_labels = [torch.from_numpy(bi[1]['annotations']['pred_labels']) for bi in batched_inputs][0][None]
        pose = batched_inputs[0][1]['veh_to_global']
        self.pose = pose
        det_boxes = det_boxes3d[:, :, [0, 1, 2, 3, 4, 5, 8]]
        det_vels = det_boxes3d[:, :, [6, 7]]
        if self.frame_id == 0:
            results = []
            track_out, instance, global_boxes, global_vels = self.init_trajectory(pose, det_boxes, det_scores, det_vels, det_labels)
            results.append(track_out)
            tracks = []
            for i in range(instance.pred_boxes.shape[0]):
                tracks.append({'translation': global_boxes[i, :2].cpu().numpy(), 'ct': global_boxes[i, :2].cpu().numpy(), 'velocity': global_vels[i].cpu().numpy(), 'detection_name': self.WAYMO_TRACKING_NAMES[int(det_labels[0, i] - 1)], 'score': instance.scores[i].cpu().numpy(), 'box_id': instance.track_id[i].cpu().numpy(), 'tracking_id': instance.track_id[i].cpu().numpy(), 'label_preds': instance.pred_classes[i].cpu().numpy(), 'active': 1, 'age': 1})
            self.tracker.reset(self.max_id, tracks)
            return results
        else:
            keep = self.class_agnostic_nms(det_boxes[0], det_scores[0].reshape(-1), nms_thresh=self.nms_thresh)
            det_boxes = det_boxes[:, keep]
            det_vels = det_vels[:, keep]
            det_scores = det_scores[:, keep]
            det_labels = det_labels[:, keep]
            if self.instances[-1].track_id.shape[0] == 0:
                if (det_boxes.sum(-1) == 0).all():
                    track_out = {'track_scores': torch.zeros(0), 'track_labels': torch.zeros(0), 'track_boxes3d': torch.zeros(0, 7), 'track_ids': torch.zeros(0).int()}
                    results = []
                    results.append(track_out)
                    return results
                else:
                    track_out = self.init_trajectory(pose, det_boxes, det_scores, det_vels, det_labels)[0]
                    results = []
                    results.append(track_out)
                    return results
            cur_ids = self.instances[-1].track_id
            traj, traj_vels = self.get_history_traj(cur_ids)
            self.num_track, self.num_candi = traj.shape[2], traj.shape[3]
            points, trajectory_hypothese, global_candidates, joint_vels, asso_mask = self.get_point_and_trajectory(traj[:, :self.history_traj_frames - 1], traj_vels[:, :self.history_traj_frames - 1], samples, det_boxes, det_vels, det_labels, det_scores)
            point_feat = self.get_proposal_aware_point_feature(points.reshape(-1, points.shape[-2], points.shape[-1]), trajectory_hypothese[:, 0].reshape(self.batch_size, 1, -1, 8), self.num_track * self.num_candi)
            point_feat = point_feat.reshape(-1, self.num_lidar_points, point_feat.shape[-1])
            boxes_feat = self.get_trajectory_boxes_feature(trajectory_hypothese[:, :self.history_traj_frames])
            token = self.token.repeat(self.batch_size * (self.num_track * self.num_candi), 1, 1)
            point_feat_list = self.encoder_fg(token, point_feat)
            fg_confidence = self.point_cls(point_feat_list[-1]).reshape(self.num_track, self.num_candi)
            fg_confidence = fg_confidence.sigmoid()
            hypotheses_feat = self.get_trajectory_hypotheses_feat_inference(point_feat_list, boxes_feat, self.instances[-1].pred_classes)
            feat_list = self.encoder_globallocal(hypotheses_feat)
            hypotheses_scores = self.joint_cls(feat_list[-1]).reshape(-1, self.num_candi).sigmoid()
            point_reg = self.point_reg(point_feat_list[-1]).reshape(self.batch_size * self.num_track, -1, 7)
            point_pred = self.generate_refined_boxes(global_candidates[..., :7], box_preds=point_reg)
            refined_candidates = point_pred.reshape(self.num_track, self.num_candi, 7)
            keep_mask = self.get_keep_mask(fg_confidence, asso_mask)
            output_new = {'pred_logits': det_scores, 'pred_boxes': det_boxes, 'pred_labels': det_labels, 'pred_vels': det_vels}
            selected = hypotheses_scores.max(-1)[1][keep_mask]
            joint_scores = fg_confidence
            matched_boxes = global_candidates[keep_mask, selected][..., :7]
            refined_matched_boxes = refined_candidates[keep_mask, selected]
            matched_vels = joint_vels[keep_mask, selected]
            matched_scores = joint_scores[keep_mask, selected].reshape(-1)
            matched_labels = self.instances[-1].pred_classes[keep_mask]
            track_id = cur_ids[keep_mask]
            track_new = {'matched_boxes': matched_boxes, 'refined_matched_boxes': refined_matched_boxes, 'matched_vels': matched_vels, 'matched_scores': matched_scores, 'matched_labels': matched_labels, 'track_id': track_id}
            track_out = self.update_trajectory(output_new, track_new)
            results = []
            results.append(track_out)
            return results

    def get_history_traj(self, cur_ids):
        num_frames = self.num_hypo_inference + self.history_traj_frames
        window_list = self.instances[::-1][:num_frames]
        traj = torch.zeros(1, len(window_list), cur_ids.shape[0], 7)
        traj_vels = torch.zeros(1, len(window_list), cur_ids.shape[0], 2)
        pose_cur_cuda = torch.from_numpy(self.pose).float()
        for k, id in enumerate(cur_ids):
            traj_id = self.history_trajectory_bank[id.item()]
            boxes_cat = torch.cat([x for t, x in enumerate(traj_id['track_boxes3d'][:num_frames])], dim=0).reshape(-1, 7).clone()
            vels_cat = torch.cat([x for t, x in enumerate(traj_id['track_vels'][:num_frames])], dim=0).reshape(-1, 2).clone()
            transfered_traj, transfered_vel = transform_global_to_current_torch(boxes_cat, vels_cat, pose_cur_cuda)
            traj[0, :boxes_cat.shape[0], k] = transfered_traj
            traj_vels[0, :vels_cat.shape[0], k] = transfered_vel
        return traj, traj_vels

    def load_pretrain_motionencoder(self):
        ckpt = torch.load(self.config.dataset.motion_model, map_location='cpu')
        if 'model' in ckpt.keys():
            ckpt = ckpt['model']
        motion_module_name = 'velboxembed'
        ckpt_traj = {}
        for k, v in ckpt.items():
            if motion_module_name in k:
                ckpt_traj[k.replace('velboxembed.', '')] = v
        self.velboxembed.load_state_dict(ckpt_traj, True)
        for parm in self.velboxembed.parameters():
            parm.required_grad = False
        self.velboxembed.eval()
        self.load_motion_module = True

    def hypotheses_augment(self, batch_bbox, targets):
        range_config = [[0.5, 0.1, np.pi / 12, 0.7], [0.5, 0.15, np.pi / 12, 0.7], [0.5, 0.15, np.pi / 9, 0.5], [0.5, 0.15, np.pi / 6, 0.3], [0.5, 0.15, np.pi / 3, 0.2]]
        max_aug_iteration = 20
        aug_list = []
        for bs_idx in range(batch_bbox.shape[0]):
            bbox = batch_bbox[bs_idx]
            aug_list_batch = []
            count = 0
            for _ in range(max_aug_iteration):
                idx = np.random.randint(low=0, high=len(range_config), size=(1,))[0]
                pos_shift = torch.from_numpy((np.random.rand(3) - 0.5) / 0.5 * range_config[idx][0]).float()
                hwl_scale = torch.from_numpy((np.random.rand(3) - 0.5) / 0.5 * range_config[idx][1] + 1.0).float()
                angle_rot = torch.from_numpy((np.random.rand(1) - 0.5) / 0.5 * range_config[idx][2]).float()
                aug_box3d = torch.cat([bbox[:, 0:3] + pos_shift[None, :], bbox[:, 3:6] * hwl_scale[None, :], bbox[:, 6:7] + angle_rot[None, :]], -1)
                if aug_box3d.shape[0] > 0 and targets[bs_idx]['gt_boxes'].shape[0] > 0:
                    ious = boxes_iou3d_gpu(aug_box3d.float(), targets[bs_idx]['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]])
                    max_iou = ious.max(-1)[0]
                    if max_iou.mean() < 0.5:
                        count += 1
                        aug_list_batch.append(aug_box3d[:, None, :])
                else:
                    count += 1
                    aug_list_batch.append(aug_box3d[:, None, :])
                if count == 2:
                    break
            if count != 2:
                for _ in range(2 - count):
                    aug_list_batch.append(bbox[:, None, :7])
            aug_list.append(torch.cat(aug_list_batch, 1)[None])
        return torch.cat(aug_list)

    def get_proposal_aware_point_feature(self, src, trajectory_rois, num_rois):
        proposal_aware_polar_point_list = []
        for i in range(trajectory_rois.shape[1]):
            corner_points, _ = get_corner_points_of_roi(trajectory_rois[:, i, :, :].contiguous())
            corner_points = corner_points.view(self.batch_size, num_rois, -1, corner_points.shape[-1])
            corner_points = corner_points.view(self.batch_size * num_rois, -1)
            trajectory_roi_center = trajectory_rois[:, i, :, :].contiguous().reshape(self.batch_size * num_rois, -1)[:, :3]
            corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
            proposal_aware_car_point = src[:, i * self.num_lidar_points:(i + 1) * self.num_lidar_points, :3].repeat(1, 1, 9) - corner_add_center_points.unsqueeze(1).repeat(1, self.num_lidar_points, 1)
            lwh = trajectory_rois[:, i, :, :].reshape(self.batch_size * num_rois, -1)[:, 3:6].unsqueeze(1).repeat(1, proposal_aware_car_point.shape[1], 1)
            diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
            proposal_aware_polar_point = spherical_coordinate(proposal_aware_car_point, diag_dist=diag_dist.unsqueeze(-1))
            proposal_aware_polar_point_list.append(proposal_aware_polar_point)
        proposal_aware_polar_point = torch.cat(proposal_aware_polar_point_list, dim=1)
        proposal_aware_polar_point = torch.cat([proposal_aware_polar_point, src[:, :, 3:]], dim=-1)
        proposal_aware_feat = self.up_dimension_geometry(proposal_aware_polar_point)
        return proposal_aware_feat

    def get_trajcetory_point_feature(self, global_trajectory_hypothses, samples):
        candi_length = global_trajectory_hypothses.shape[-2]
        point = crop_current_frame_points(self.num_lidar_points, global_trajectory_hypothses, samples['points'])
        point_feat = self.get_proposal_aware_point_feature(point.reshape(-1, point.shape[-2], point.shape[-1]), global_trajectory_hypothses[:, 0].reshape(self.batch_size, 1, -1, 8), self.num_track * candi_length)
        point_feat = point_feat.reshape(-1, self.num_lidar_points, point_feat.shape[-1])
        token = self.token.repeat(self.batch_size * self.num_track * self.num_hypo_train, 1, 1)
        point_feat_list = self.encoder_fg(token, point_feat)
        return point_feat_list

    def get_trajectory_boxes_feature(self, traj_rois):
        traj_boxes = traj_rois.clone()
        batch_size = traj_rois.shape[0]
        num_track, num_candi = traj_rois.shape[2], traj_rois.shape[3]
        empty_mask = traj_rois[..., :6].sum(-1) == 0
        traj_boxes[..., 6] = traj_boxes[..., 6] % (2 * np.pi)
        traj_boxes[empty_mask] = 0
        boxes_feat, _ = self.seqboxembed(traj_boxes.permute(0, 2, 3, 4, 1).contiguous().view(-1, traj_boxes.shape[-1], traj_boxes.shape[1]))
        boxes_feat = boxes_feat.reshape(batch_size, num_track, num_candi, boxes_feat.shape[-1])
        return boxes_feat

    def get_trajectory_hypotheses_feat(self, point_feat_list, boxes_feat, pred_labels):
        point_feat = point_feat_list[-1].reshape(self.batch_size, self.num_track, self.num_hypo_train, -1)
        src = torch.cat([point_feat, boxes_feat, torch.zeros_like(point_feat)[..., :3]], -1)
        car_mask = pred_labels == 1
        ped_mask = pred_labels == 2
        cyc_mask = pred_labels == 3
        src[car_mask[:, :, 0]][..., -3:] = self.car_embed
        src[ped_mask[:, :, 0]][..., -3:] = self.ped_embed
        src[cyc_mask[:, :, 0]][..., -3:] = self.cyc_embed
        src = F.relu(self.cls_embed(src))
        return src

    def get_trajectory_hypotheses_feat_inference(self, point_feat_list, boxes_feat, pred_labels):
        point_feat = point_feat_list[-1].reshape(self.batch_size, self.num_track, -1, self.token.shape[-1])
        src = torch.cat([point_feat, boxes_feat, torch.zeros_like(point_feat)[..., :3]], -1)
        self.car_mask = pred_labels == 1
        self.ped_mask = pred_labels == 2
        self.cyc_mask = pred_labels == 3
        src[:, self.car_mask][..., -3:] = self.car_embed
        src[:, self.ped_mask][..., -3:] = self.ped_embed
        src[:, self.cyc_mask][..., -3:] = self.cyc_embed
        src = F.relu(self.cls_embed(src))
        return src

    def organize_proposals(self, pred_boxes3d, pred_scores, pred_labels):
        all_batch_list_boxes = []
        all_batch_list_score = []
        all_batch_list_label = []
        for i in range(len(pred_boxes3d)):
            cur_batch_box = pred_boxes3d[i].reshape(self.traj_length + 1, -1, 9)
            cur_batch_score = pred_scores[i].reshape(self.traj_length + 1, -1)
            cur_batch_label = pred_labels[i].reshape(self.traj_length + 1, -1)
            batch_list = []
            batch_list_score = []
            batch_list_label = []
            for j in range(self.traj_length + 1):
                cur_box = cur_batch_box[j]
                cur_score = cur_batch_score[j]
                cur_label = cur_batch_label[j]
                assert cur_box.shape[0] == cur_score.shape[0]
                mask = self.class_agnostic_nms(cur_box[:, [0, 1, 2, 3, 4, 5, 8]], cur_score.reshape(-1), nms_thresh=self.train_nms_thresh, score_thresh=self.train_score_thresh)
                batch_list.append(cur_box[mask])
                batch_list_score.append(cur_score[mask].reshape(-1, 1))
                batch_list_label.append(cur_label[mask].reshape(-1, 1))
            cur_batch_box, _ = reorder_rois(batch_list)
            all_batch_list_boxes.append(cur_batch_box.reshape(-1, 9))
            cur_batch_score, _ = reorder_rois(batch_list_score)
            all_batch_list_score.append(cur_batch_score.reshape(-1, 1))
            cur_batch_label, _ = reorder_rois(batch_list_label)
            all_batch_list_label.append(cur_batch_label.reshape(-1, 1))
        pred_boxes3d, _ = reorder_rois(all_batch_list_boxes)
        pred_scores, _ = reorder_rois(all_batch_list_score)
        pred_labels, _ = reorder_rois(all_batch_list_label)
        pred_boxes3d_list = pred_boxes3d.reshape(self.batch_size, self.traj_length + 1, -1, 9)
        det_boxes3d = pred_boxes3d_list[:, 0, :, [0, 1, 2, 3, 4, 5, -1]]
        pred_vel = pred_boxes3d_list[:, 1:2, :, [6, 7]]
        traj, valid_mask = self.generate_trajectory(pred_boxes3d_list[:, 1:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]])
        time_sweeps = traj.new_ones(traj.shape[0], traj.shape[1], traj.shape[2], 1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * i * 0.1
        traj = torch.cat([traj[..., [0, 1, 2, 3, 4, 5, 8]], time_sweeps], -1)
        with torch.no_grad():
            pred_hypo = self.get_pred_motion(traj, pred_vel)[:, 0, :, 0]
        traj = traj[:, 1:]
        pred_label_list = pred_labels.reshape(self.batch_size, self.traj_length + 1, -1, 1)
        pred_labels = pred_label_list[:, 1]
        return pred_hypo, pred_labels, det_boxes3d, traj

    def generate_trajectory_hypothses(self, transfered_det, det_boxes3d, traj, num_hypo_det, aug_hypo=None):
        batch_size, num_track = transfered_det.shape[0], transfered_det.shape[1]
        dist = torch.cdist(transfered_det[:, :, :2], det_boxes3d[:, :, :2], 2)
        matched_id = torch.arange(transfered_det.shape[1]).reshape(1, -1, 1).repeat(batch_size, 1, num_hypo_det)
        matched_id[..., :num_hypo_det] = det_boxes3d.shape[1]
        min_value, matched_det_id = torch.topk(-dist, num_hypo_det, -1)
        valid_dist_mask = -min_value < self.dist_thresh
        matched_id[..., :num_hypo_det][valid_dist_mask] = matched_det_id[valid_dist_mask]
        batch_index = torch.arange(batch_size).reshape(-1, 1, 1).repeat(1, num_track, 1)
        det_boxes_with_bg = torch.cat([det_boxes3d, torch.zeros(batch_size, 1, 7)], 1)
        group_det_boxes = det_boxes_with_bg[batch_index, matched_id]
        time = torch.zeros([batch_size, num_track, num_hypo_det, 1])
        group_det_boxes = torch.cat([group_det_boxes, time], -1)
        transfered_det = transfered_det[:, None, :, None, :]
        if aug_hypo is not None:
            aug_hypo = aug_hypo[:, None, :, :, :]
            time = torch.zeros_like(aug_hypo[..., :1])
            aug_hypo = torch.cat([aug_hypo, time], -1)
            transfered_det = torch.cat([transfered_det, aug_hypo], 3)
        global_candidates = torch.cat([transfered_det, group_det_boxes.unsqueeze(1)], 3)
        traj_repeat = traj.unsqueeze(3).repeat(1, 1, 1, global_candidates.shape[3], 1)
        global_trajectory_hypothses = torch.cat([global_candidates, traj_repeat], 1)
        return global_trajectory_hypothses, global_candidates

    def get_cls_targets(self, pred_boxes3d, global_candidates, targets):
        fg_mask_list = []
        ious_targets = []
        reg_mask_list = []
        gt_boxes_list = []
        batch_size, num_track = pred_boxes3d.shape[0], pred_boxes3d.shape[1]
        for i in range(batch_size):
            num_gt = targets[i]['gt_boxes'].shape[0]
            if pred_boxes3d[i].shape[0] > 0 and num_gt > 0:
                rois = global_candidates[i][..., :7].reshape(-1, 7)
                rois_iou = boxes_iou3d_gpu(rois, targets[i]['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]])
                rois_iou = rois_iou.reshape(num_track, self.num_hypo_train, num_gt)
                track_iou = rois_iou[:, 0]
                max_iou, track_id = track_iou.max(-1)
                fg_track_mask = max_iou > 0.5
                reg_mask = rois_iou.max(-1)[0] > 0.5
                group_iou = rois_iou[torch.arange(num_track), :, track_id].reshape(-1, self.num_hypo_train)
                group_iou_labels = self.get_iou_labels(group_iou)
                track_id = rois_iou.reshape(-1, num_gt).max(-1)[1]
                ordered_gt_boxes = targets[i]['gt_boxes'][track_id][:, [0, 1, 2, 3, 4, 5, -1]]
                gt_boxes_list.append(ordered_gt_boxes)
                reg_mask_list.append(reg_mask)
                ious_targets.append(group_iou_labels)
                fg_mask_list.append(fg_track_mask)
            else:
                ious_targets.append(torch.zeros([num_track, self.num_hypo_train]))
                fg_mask_list.append(torch.zeros([num_track]).bool())
                reg_mask_list.append(torch.zeros([num_track, self.num_hypo_train]).bool())
                gt_boxes_list.append(torch.zeros([num_track * self.num_hypo_train, 7]))
        fg_iou_mask = torch.cat(fg_mask_list)
        gt_boxes = torch.cat(gt_boxes_list)
        fg_reg_mask = torch.cat(reg_mask_list)
        fg_reg_mask = fg_reg_mask.reshape(1, -1).repeat(self.num_encoder_layers, 1).reshape(-1)
        ious_targets = torch.cat(ious_targets, 0).reshape(-1).repeat(self.num_encoder_layers)
        return fg_iou_mask, fg_reg_mask, ious_targets, gt_boxes

    def get_reg_targets(self, pred_rois, gt_boxes):
        rois, gt_of_rois = pred_rois[None].clone(), gt_boxes[None].clone()
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry
        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0
        local_rois[:, :, 6] = 0
        gt_of_rois = rotate_points_along_z(points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)).view(1, -1, gt_of_rois.shape[-1])
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, 6] = heading_label
        reg_targets = encode_boxes_res_torch(gt_of_rois[0], local_rois[0])
        reg_targets = reg_targets.repeat(self.num_encoder_layers, 1).reshape(1, -1, 7)
        return reg_targets

    def get_iou_labels(self, cls_iou):
        iou_bg_thresh = 0.25
        iou_fg_thresh = 0.75
        fg_mask = cls_iou > iou_fg_thresh
        bg_mask = cls_iou < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        batch_cls_labels = (fg_mask > 0).float()
        batch_cls_labels[interval_mask] = (cls_iou[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        return batch_cls_labels

    def class_agnostic_nms(self, pred_boxes3d, pred_scores, nms_thresh=0.1, score_thresh=None, nms_pre_maxsize=4096, nms_post_maxsize=500):
        box_preds = pred_boxes3d
        scores = pred_scores
        if score_thresh is not None:
            scores_mask = scores >= score_thresh
            scores = scores[scores_mask]
            box_preds = box_preds[scores_mask]
        rank_scores_nms, indices = torch.topk(scores, k=min(nms_pre_maxsize, scores.shape[0]))
        box_preds_nms = box_preds[indices][:, :7]
        if box_preds_nms.shape[0] > 0:
            keep_idx, _ = nms_gpu(box_preds_nms, rank_scores_nms, thresh=nms_thresh)
            selected = indices[keep_idx[:nms_post_maxsize]]
            if score_thresh is not None:
                original_idxs = scores_mask.nonzero().view(-1)
                selected = original_idxs[selected]
            return selected
        else:
            return torch.tensor([]).long()

    def generate_trajectory(self, proposals_list):
        cur_batch_boxes = proposals_list[:, 0, :, :]
        trajectory_rois = torch.zeros_like(cur_batch_boxes[:, None, :, :]).repeat(1, proposals_list.shape[1], 1, 1)
        trajectory_rois[:, 0, :, :] = proposals_list[:, 0, :, :]
        valid_length = torch.zeros([trajectory_rois.shape[0], trajectory_rois.shape[1], trajectory_rois.shape[2]])
        valid_length[:, 0] = 1
        num_frames = proposals_list.shape[1]
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = trajectory_rois[:, i - 1, :, 0:2] - 0.1 * trajectory_rois[:, i - 1, :, 6:8]
            frame[:, :, 2:] = trajectory_rois[:, i - 1, :, 2:]
            for bs_idx in range(proposals_list.shape[0]):
                iou3d = boxes_iou3d_gpu(frame[bs_idx, :, [0, 1, 2, 3, 4, 5, -1]], proposals_list[bs_idx, i, :, [0, 1, 2, 3, 4, 5, -1]])
                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)
                fg_inds = (max_overlaps >= 0.5).nonzero().view(-1)
                valid_length[bs_idx, i, fg_inds] = 1
                trajectory_rois[bs_idx, i, fg_inds, :] = proposals_list[bs_idx, i, traj_assignment[fg_inds]]
        return trajectory_rois, valid_length

    def get_loss(self, rois, gt_boxes, point_cls, joint_cls, boxes_cls, point_reg, ious_targets, reg_targets, fg_reg_mask, fg_iou_mask):
        loss_reg = self.reg_loss_func(point_reg, reg_targets)[:, fg_reg_mask]
        loss_reg = loss_reg.sum() / max(fg_reg_mask.sum(), 1)
        loss_corner = get_corner_loss(point_reg.reshape(-1, 7), rois.repeat(self.num_encoder_layers, 1), gt_boxes.repeat(self.num_encoder_layers, 1), fg_reg_mask)
        loss_point_cls = F.binary_cross_entropy(point_cls.sigmoid().reshape(-1), ious_targets)
        index = ious_targets.shape[0] // self.num_encoder_layers
        ious_targets = ious_targets[:index].reshape(self.batch_size * self.num_track, self.num_hypo_train)
        loss_box_cls = F.binary_cross_entropy(boxes_cls.sigmoid()[fg_iou_mask], ious_targets[fg_iou_mask])
        fg_mask_repeat = fg_iou_mask.repeat(self.num_encoder_layers)
        group_ious_repeat = ious_targets.repeat(self.num_encoder_layers, 1)
        loss_joint_cls = F.binary_cross_entropy(joint_cls.sigmoid()[fg_mask_repeat], group_ious_repeat[fg_mask_repeat])
        loss_cls = loss_joint_cls + loss_point_cls + loss_box_cls
        loss_reg = loss_reg + loss_corner
        return loss_cls, loss_reg

    def init_trajectory(self, pose, det_boxes, det_scores, det_vels, det_labels):
        self.instances = []
        instance = Instances()
        self.history_trajectory_bank = collections.defaultdict(dict)
        if self.config.model.eval_class == 'VEHICLE':
            score_thresh = self.new_born_car
        elif self.config.model.eval_class == 'PEDESTRIAN':
            score_thresh = self.new_born_ped
        elif self.config.model.eval_class == 'CYCLIST':
            score_thresh = self.new_born_cyc
        else:
            raise NotImplementedError
        score_mask = self.class_agnostic_nms(det_boxes[0], det_scores[0].reshape(-1), nms_thresh=self.nms_thresh, score_thresh=score_thresh)
        instance.track_id = torch.arange(score_mask.shape[0]).reshape(-1)
        instance.pred_boxes = det_boxes[0, score_mask]
        instance.vels = det_vels[0, score_mask]
        instance.scores = det_scores[0, score_mask]
        instance.pred_classes = det_labels[0, score_mask]
        instance.pose = pose
        instance.new_boxes = torch.cat([det_boxes[0, score_mask], det_vels[0, score_mask]], -1)
        self.instances.append(instance)
        self.max_id = score_mask.shape[0]
        track_out = {'track_scores': det_scores[0, score_mask].detach().cpu(), 'track_labels': det_labels[0, score_mask].detach().cpu(), 'track_boxes3d': det_boxes[0, score_mask].detach().cpu(), 'track_ids': torch.arange(score_mask.shape[0]).reshape(-1).int()}
        global_boxes, global_vels = transform_box_to_global(instance.pred_boxes.cpu().numpy(), instance.vels.cpu().numpy(), self.pose)
        for index, track_id in enumerate(track_out['track_ids']):
            track_id = track_id.item()
            self.history_trajectory_bank[track_id]['track_scores'] = []
            self.history_trajectory_bank[track_id]['track_vels'] = []
            self.history_trajectory_bank[track_id]['track_labels'] = []
            self.history_trajectory_bank[track_id]['track_boxes3d'] = []
            self.history_trajectory_bank[track_id]['track_pose'] = []
            self.history_trajectory_bank[track_id]['track_scores'].insert(0, instance.scores[index])
            self.history_trajectory_bank[track_id]['track_vels'].insert(0, global_vels[index])
            self.history_trajectory_bank[track_id]['track_labels'].insert(0, instance.pred_classes[index])
            self.history_trajectory_bank[track_id]['track_boxes3d'].insert(0, global_boxes[index])
            self.history_trajectory_bank[track_id]['track_pose'].insert(0, instance.pose)
        return track_out, instance, global_boxes, global_vels

    def get_point_and_trajectory(self, traj, traj_vels, samples, det_boxes, det_vels, det_labels, det_scores):
        pred_traj = self.get_pred_candi(traj, traj_vels)
        pred_candi = pred_traj[:, :, 0, :7]
        cur_vels = self.instances[-1].vels[None]
        det_candi, det_candi_vel, asso_mask = self.get_det_candi(self.pose, pred_candi, cur_vels, det_boxes, det_vels, det_labels, det_scores, self.frame_id)
        time_sweeps = traj.new_ones(traj.shape[0], traj.shape[1], traj.shape[2], 1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * (i + 1) * 0.1
        traj = torch.cat([traj, time_sweeps], -1)
        trajectory_hypothese, global_candidates, joint_vels = self.genereate_trajcetory_hypotheses_inference(pred_traj, det_candi, traj, cur_vels, det_candi_vel)
        self.num_candi = trajectory_hypothese.shape[3]
        points = crop_current_frame_points(self.num_lidar_points, trajectory_hypothese, samples['points'])
        return points, trajectory_hypothese, global_candidates, joint_vels, asso_mask

    def get_pred_candi(self, traj, traj_vels):
        num_pred = max(1, min(self.num_hypo_inference, traj.shape[1] - 1))
        pred_traj_list = []
        for i in range(num_pred):
            cur_traj = traj[:, i:i + self.history_traj_frames]
            cur_vel = traj_vels[:, i:i + 1]
            pred_traj = self.get_pred_motion(cur_traj, cur_vel)[:, i]
            pred_traj_list.append(pred_traj)
        pred_traj = torch.cat(pred_traj_list, 2)
        empty_mask = pred_traj[..., 3:6].sum(-1) == 0
        pred_traj[empty_mask] = 0
        return pred_traj

    def get_pred_motion(self, traj, pred_vel=None):
        traj_rois = traj.clone().unsqueeze(3)
        batch_size, len_traj, num_track = traj_rois.shape[0], traj_rois.shape[1], traj_rois.shape[2]
        self.num_future = 10
        history_traj = traj_rois
        future_traj_init = traj_rois[:, 0:1].repeat(1, self.num_future, 1, 1, 1)
        future_traj_center = traj_rois[:, 0:1, :, :, :3].repeat(1, self.num_future, 1, 1, 1)
        pred_vel_hypos = 0.1 * pred_vel.unsqueeze(3).repeat(1, len_traj, 1, 1, 1)
        for i in range(future_traj_center.shape[1]):
            future_traj_center[:, i, :, :, :2] += 0.1 * (i + 1) * pred_vel[:, 0].unsqueeze(2)
        future_traj_init[..., :2] = future_traj_center[..., :2]
        empty_mask = (traj_rois[:, 0:1, :, :, 3:6].sum(-1) == 0).repeat(1, traj_rois.shape[1], 1, 1)
        time_sweeps = torch.ones_like(history_traj)[..., :1]
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * i * 0.1
        history_traj = torch.cat([history_traj, time_sweeps], -1)
        history_traj_local, history_vel_local = transform_trajs_to_local_coords(history_traj, center_xyz=history_traj[:, 0:1, :, :, 0:2], center_heading=history_traj[:, 0:1, :, :, 6], pred_vel_hypo=pred_vel_hypos, heading_index=6)
        future_traj_init_local, _ = transform_trajs_to_local_coords(future_traj_init, center_xyz=history_traj[:, 0:1, :, :, 0:2], center_heading=history_traj[:, 0:1, :, :, 6], heading_index=6)
        history_traj_local = torch.cat([history_traj_local[..., :2], history_traj_local[..., 6:7], history_vel_local, history_traj_local[..., 7:8]], -1)
        history_traj_local = history_traj_local.permute(0, 2, 3, 1, 4).reshape(batch_size, num_track * history_traj.shape[3], len_traj, -1)
        valid_mask = ~empty_mask.permute(0, 2, 3, 1).reshape(batch_size, num_track * history_traj.shape[3], len_traj)
        future_traj_pred = self.velboxembed(history_traj_local, valid_mask)
        future_traj_pred = future_traj_pred.reshape(batch_size, num_track, history_traj.shape[3], self.num_future, 3).permute(0, 3, 1, 2, 4)
        future_traj_local = future_traj_init_local.clone()
        future_traj_local[..., [0, 1, 6]] = future_traj_pred + future_traj_init_local[..., [0, 1, 6]].detach()
        future_traj = transform_trajs_to_global_coords(future_traj_local, center_xyz=history_traj[:, 0:1, :, 0:1, 0:2], center_heading=history_traj[:, 0:1, :, 0:1, 6], heading_index=6)[0]
        return future_traj

    def get_det_candi(self, pose, transfered_det, cur_vels, det_boxes, det_vels, det_labels, det_scores, frame_id):
        time_lag = 0.1
        global_boxes, global_vels = transform_box_to_global(det_boxes[0].cpu(), det_vels[0].cpu(), pose)
        current_det = []
        for i in range(det_boxes.shape[1]):
            current_det.append({'translation': global_boxes[i, :2].cpu().numpy(), 'velocity': global_vels[i].cpu().numpy(), 'detection_name': self.WAYMO_TRACKING_NAMES[int(det_labels[0, i] - 1)], 'score': det_scores[0, i].cpu().numpy(), 'box_id': i, 'label_preds': det_labels[0, i].cpu().numpy()})
        outputs = self.tracker.step_centertrack(current_det, time_lag, frame_id)
        tracking_ids = []
        box_ids = []
        for item in outputs:
            if item['active'] == 0:
                continue
            box_ids.append(item['box_id'])
            tracking_ids.append(item['tracking_id'])
        remained_box_ids = np.array(box_ids)
        det_candi = torch.zeros_like(transfered_det)
        det_candi_vel = torch.zeros_like(cur_vels)
        asso_mask = torch.zeros(transfered_det.shape[1]).bool()
        for i in range(remained_box_ids.shape[0]):
            track_id = tracking_ids[i]
            det_candi[0][track_id] = det_boxes[0][remained_box_ids][i]
            det_candi_vel[0][track_id] = det_vels[0][remained_box_ids][i]
            asso_mask[track_id] = True
        return det_candi, det_candi_vel, asso_mask

    def genereate_trajcetory_hypotheses_inference(self, pred_hypo, cp_matched_boxes, traj, cur_vels, det_candi_vel):
        time = torch.zeros_like(pred_hypo)[..., :1]
        pred_hypo = torch.cat([pred_hypo, time], -1).unsqueeze(1)
        group_det_boxes = torch.cat([cp_matched_boxes.unsqueeze(2), torch.zeros([self.batch_size, self.num_track, 1, 1])], -1)
        global_candidates = torch.cat([pred_hypo, group_det_boxes.unsqueeze(1)], 3)
        trajectory_hypotheses = torch.cat([global_candidates, traj.unsqueeze(3).repeat(1, 1, 1, global_candidates.shape[3], 1)], 1)
        vels_hypotheses = cur_vels[:, :, None, :].repeat(1, 1, global_candidates.shape[3] - 1, 1)
        vels_hypotheses = torch.cat([vels_hypotheses, det_candi_vel.unsqueeze(2)], 2)
        vels_hypotheses = vels_hypotheses.reshape(self.num_track, -1, 2)
        global_candidates = global_candidates.reshape(self.num_track, -1, 8)
        return trajectory_hypotheses, global_candidates, vels_hypotheses

    def generate_refined_boxes(self, rois, box_preds=None):
        code_size = rois.shape[-1]
        num_rois = rois.shape[0]
        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0
        batch_box_preds = decode_torch(box_preds, local_rois).view(-1, code_size)
        batch_box_preds = rotate_points_along_z(batch_box_preds.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(num_rois, -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds, rois[:, :, 7:]], -1)
        return batch_box_preds

    def get_keep_mask(self, fg_confidence, asso_mask):
        keep_mask = torch.zeros_like(fg_confidence[:, 0]).bool()
        keep_mask[asso_mask] = True
        track_score_mask = torch.zeros_like(keep_mask)
        track_score_mask[self.car_mask] = fg_confidence[:, 0].reshape(-1)[self.car_mask] > self.keep_thresh_car
        track_score_mask[self.ped_mask] = fg_confidence[:, 0].reshape(-1)[self.ped_mask] > self.keep_thresh_ped
        track_score_mask[self.cyc_mask] = fg_confidence[:, 0].reshape(-1)[self.cyc_mask] > self.keep_thresh_cyc
        keep_mask[~asso_mask] = track_score_mask[~asso_mask]
        return keep_mask

    def update_trajectory(self, output_new, track_new):
        boxes_new = output_new['pred_boxes'].reshape(1, -1, 7)
        scores_new = output_new['pred_logits'].reshape(1, -1)
        vels_new = output_new['pred_vels'].reshape(1, -1, 2)
        labels_new = output_new['pred_labels']
        matched_boxes = track_new['matched_boxes']
        refined_matched_boxes = track_new['refined_matched_boxes']
        track_id = track_new['track_id']
        matched_vels = track_new['matched_vels']
        matched_scores = track_new['matched_scores']
        matched_labels = track_new['matched_labels']
        if self.frame_id > 0 and boxes_new[0].shape[0] > 0 and matched_boxes.shape[0] > 0:
            ious_det2track = boxes_iou3d_gpu(boxes_new[0], matched_boxes)
            mask = ious_det2track.max(-1)[0] > self.new_born_nms_thresh
            scores_new[0][mask] = 0
        scores_new_mask = torch.zeros_like(scores_new).bool()
        new_car_mask = labels_new == 1
        new_ped_mask = labels_new == 2
        new_cyc_mask = labels_new == 3
        scores_new_mask[new_car_mask] = scores_new[new_car_mask] > self.new_born_car
        scores_new_mask[new_ped_mask] = scores_new[new_ped_mask] > self.new_born_ped
        scores_new_mask[new_cyc_mask] = scores_new[new_cyc_mask] > self.new_born_cyc
        if scores_new_mask.sum() > 0:
            new_det_scores_mask = scores_new_mask[0]
            new_det_boxes = boxes_new[0, new_det_scores_mask]
            new_det_scores = scores_new[0, new_det_scores_mask].reshape(-1)
            new_det_vels = vels_new[0, new_det_scores_mask]
            new_det_labels = labels_new[0, new_det_scores_mask]
            new_track_id = self.max_id + 1 + torch.arange(new_det_boxes.shape[0])
            self.max_id = self.max_id + 1 + new_det_boxes.shape[0]
        else:
            new_det_scores_mask = []
            new_det_boxes = torch.tensor([])
            new_det_scores = torch.tensor([])
            new_det_vels = torch.tensor([])
            new_det_labels = torch.tensor([])
            new_track_id = torch.tensor([])
        instance = Instances()
        instance.track_id = torch.cat([track_id, new_track_id], 0)
        instance.pred_boxes = torch.cat([matched_boxes, new_det_boxes], 0)
        instance.refined_pred_boxes = torch.cat([refined_matched_boxes, new_det_boxes], 0)
        instance.new_boxes = boxes_new[0, new_det_scores_mask]
        instance.scores = torch.cat([matched_scores, new_det_scores], 0)
        instance.vels = torch.cat([matched_vels, new_det_vels], 0)
        instance.pred_classes = torch.cat([matched_labels, new_det_labels], 0)
        instance.pose = self.pose
        self.instances.append(instance)
        track_out = {'track_scores': instance.scores.cpu(), 'track_labels': instance.pred_classes.cpu(), 'track_boxes3d': instance.refined_pred_boxes.cpu(), 'track_ids': instance.track_id.detach().cpu().int()}
        global_boxes, global_vels = transform_box_to_global(instance.pred_boxes.cpu().numpy(), instance.vels.cpu().numpy(), self.pose)
        for index, track_id in enumerate(track_out['track_ids']):
            track_id = track_id.item()
            if track_id not in self.history_trajectory_bank.keys():
                self.history_trajectory_bank[track_id]['track_scores'] = []
                self.history_trajectory_bank[track_id]['track_vels'] = []
                self.history_trajectory_bank[track_id]['track_labels'] = []
                self.history_trajectory_bank[track_id]['track_boxes3d'] = []
                self.history_trajectory_bank[track_id]['track_pose'] = []
            self.history_trajectory_bank[track_id]['track_scores'].insert(0, instance.scores[index])
            self.history_trajectory_bank[track_id]['track_vels'].insert(0, global_vels[index])
            self.history_trajectory_bank[track_id]['track_labels'].insert(0, instance.pred_classes[index])
            self.history_trajectory_bank[track_id]['track_boxes3d'].insert(0, global_boxes[index])
            self.history_trajectory_bank[track_id]['track_pose'].insert(0, instance.pose)
        self.update_global_hypotheses_for_dist_asso(global_boxes, global_vels, instance)
        return track_out

    def update_global_hypotheses_for_dist_asso(self, global_boxes, global_vels, instance):
        tracks = []
        for i in range(instance.pred_boxes.shape[0]):
            tracks.append({'translation': global_boxes[i, :2].cpu().numpy(), 'ct': global_boxes[i, :2].cpu().numpy(), 'velocity': global_vels[i].cpu().numpy(), 'detection_name': self.WAYMO_TRACKING_NAMES[int(instance.pred_classes[i] - 1)], 'score': instance.scores[i].cpu().numpy(), 'box_id': i, 'tracking_id': i, 'label_preds': instance.pred_classes[i].cpu().numpy(), 'active': 1, 'age': 1})
        self.tracker.reset(self.max_id, tracks)


class MotionPrediction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.model.device)
        self.config = config
        self.is_train = config.task == 'train'
        self.num_hypo = config.model.num_hypo
        hidden_dim = config.model.hidden_dim
        input_dim = config.model.motion_input_dim
        self.num_future = config.dataset.future_frames
        self.velboxembed = MotionEncoder(input_dim, hidden_dim, out_channels=self.num_future * 3)
        self.traj_length = config.dataset.traj_length
        self.dist_thresh = config.model.dist_thresh
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=None)

    def genereate_trajcetory_hypotheses(self, transfered_det, det_boxes3d, traj, num_hypo):
        batch_size, num_track = transfered_det.shape[0], transfered_det.shape[1]
        dist = torch.cdist(transfered_det[:, :, :2], det_boxes3d[:, :, :2], 2)
        matched_id = torch.arange(transfered_det.shape[1]).reshape(1, -1, 1).repeat(batch_size, 1, num_hypo - 1)
        matched_id[..., :num_hypo - 1] = det_boxes3d.shape[1]
        min_value, matched_det_id = torch.topk(-dist, num_hypo - 1, -1)
        valid_dist_mask = -min_value < self.dist_thresh
        matched_id[..., :num_hypo - 1][valid_dist_mask] = matched_det_id[valid_dist_mask]
        batch_index = torch.arange(batch_size).reshape(-1, 1, 1).repeat(1, num_track, 1)
        det_boxes_with_bg = torch.cat([det_boxes3d, torch.zeros(batch_size, 1, 7)], 1)
        group_det_boxes = det_boxes_with_bg[batch_index, matched_id]
        time = torch.zeros([batch_size, num_track, num_hypo - 1, 1])
        group_det_boxes = torch.cat([group_det_boxes, time], -1)
        transfered_det = transfered_det[:, None, :, None, :]
        global_candidates = torch.cat([transfered_det, group_det_boxes.unsqueeze(1)], 3)
        traj_repeat = traj.unsqueeze(3).repeat(1, 1, 1, global_candidates.shape[3], 1)
        trajcetory_hypotheses = torch.cat([global_candidates, traj_repeat], 1)
        return trajcetory_hypotheses, global_candidates, valid_dist_mask

    def get_pred_traj(self, traj_rois, valid_mask=None, pred_vel=None, det_vel=None, pred_label=None):
        batch_size, len_traj, num_track = traj_rois.shape[0], traj_rois.shape[1], traj_rois.shape[2]
        history_traj = traj_rois.clone()
        future_traj_init = traj_rois.clone()[:, 0:1].repeat(1, self.num_future, 1, 1, 1)
        future_traj_center = traj_rois.clone()[:, 0:1, :, :, :3].repeat(1, self.num_future, 1, 1, 1)
        pred_vel_hypos = 0.1 * pred_vel.unsqueeze(1).unsqueeze(3).repeat(1, len_traj, 1, 2, 1)
        pred_vel_hypos[:, :, :, 1] = 0.1 * det_vel[:, None, :]
        for i in range(future_traj_center.shape[1]):
            future_traj_center[:, i, :, :, :2] += 0.1 * (i + 1) * pred_vel.unsqueeze(2)
        future_traj_init[..., :2] = future_traj_center[..., :2]
        empty_mask = (traj_rois[:, 0:1, :, :, 3:6].sum(-1) == 0).repeat(1, traj_rois.shape[1], 1, 1)
        history_traj_local, history_vel_local = transform_trajs_to_local_coords(history_traj, center_xyz=history_traj[:, 0:1, :, :, 0:2], center_heading=history_traj[:, 0:1, :, :, 6], pred_vel_hypo=pred_vel_hypos, heading_index=6)
        future_traj_init_local, _ = transform_trajs_to_local_coords(future_traj_init, center_xyz=history_traj[:, 0:1, :, :, 0:2], center_heading=history_traj[:, 0:1, :, :, 6], heading_index=6)
        history_traj_local = torch.cat([history_traj_local[..., :2], history_traj_local[..., 6:7], history_vel_local, history_traj_local[..., 7:8]], -1)
        history_traj_local = history_traj_local.permute(0, 2, 3, 1, 4).reshape(batch_size, num_track * history_traj.shape[3], len_traj, -1)
        valid_mask = ~empty_mask.permute(0, 2, 3, 1).reshape(batch_size, num_track * history_traj.shape[3], len_traj)
        future_traj_pred = self.velboxembed(history_traj_local, valid_mask)
        future_traj_pred = future_traj_pred.reshape(batch_size, num_track, history_traj.shape[3], self.num_future, 3).permute(0, 3, 1, 2, 4)
        future_traj = future_traj_init_local.clone()
        future_traj[..., [0, 1, 6]] = future_traj_pred + future_traj_init_local[..., [0, 1, 6]].detach()
        return future_traj

    def organize_proposals(self, batch_size, pred_boxes3d, pred_scores, pred_labels):
        all_batch_list_boxes = []
        all_batch_list_score = []
        all_batch_list_label = []
        for i in range(len(pred_boxes3d)):
            cur_batch_box = pred_boxes3d[i].reshape(self.traj_length + 1, -1, 9)
            cur_batch_score = pred_scores[i].reshape(self.traj_length + 1, -1)
            cur_batch_label = pred_labels[i].reshape(self.traj_length + 1, -1)
            batch_list = []
            batch_list_score = []
            batch_list_label = []
            for j in range(self.traj_length + 1):
                cur_box = cur_batch_box[j]
                cur_score = cur_batch_score[j]
                cur_label = cur_batch_label[j]
                assert cur_box.shape[0] == cur_score.shape[0]
                mask = self.class_agnostic_nms(cur_box[:, [0, 1, 2, 3, 4, 5, 8]], cur_score.reshape(-1), nms_thresh=self.config.dataset.nms_thresh, score_thresh=self.config.dataset.score_thresh)
                batch_list.append(cur_box[mask])
                batch_list_score.append(cur_score[mask].reshape(-1, 1))
                batch_list_label.append(cur_label[mask].reshape(-1, 1))
            cur_batch_box, _ = self.reorder_rois(batch_list)
            all_batch_list_boxes.append(cur_batch_box.reshape(-1, 9))
            cur_batch_score, _ = self.reorder_rois(batch_list_score)
            all_batch_list_score.append(cur_batch_score.reshape(-1, 1))
            cur_batch_label, _ = self.reorder_rois(batch_list_label)
            all_batch_list_label.append(cur_batch_label.reshape(-1, 1))
        pred_boxes3d, _ = self.reorder_rois(all_batch_list_boxes)
        pred_scores, _ = self.reorder_rois(all_batch_list_score)
        pred_labels, _ = self.reorder_rois(all_batch_list_label)
        pred_boxes3d_list = pred_boxes3d.reshape(batch_size, self.traj_length + 1, -1, 9)
        det_boxes3d = pred_boxes3d_list[:, 0, :, [0, 1, 2, 3, 4, 5, -1]]
        pred_boxes3d = pred_boxes3d_list[:, 1, :, [0, 1, 2, 3, 4, 5, -1]]
        traj, valid_mask = self.generate_trajectory(pred_boxes3d_list[:, 1:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]])
        time_sweeps = traj.new_ones(traj.shape[0], traj.shape[1], traj.shape[2], 1)
        for i in range(time_sweeps.shape[1]):
            time_sweeps[:, i] = time_sweeps[:, i] * i * 0.1
        traj = torch.cat([traj[..., [0, 1, 2, 3, 4, 5, 8]], time_sweeps], -1)
        pred_boxes3d = traj[:, 0]
        traj = traj[:, 1:]
        det_vel = pred_boxes3d_list[:, 0, :, [6, 7]]
        pred_vel = pred_boxes3d_list[:, 1, :, [6, 7]]
        pred_score_list = pred_scores.reshape(batch_size, self.traj_length + 1, -1, 1)
        pred_scores = pred_score_list[:, 1]
        pred_label_list = pred_labels.reshape(batch_size, self.traj_length + 1, -1, 1)
        pred_labels = pred_label_list[:, 1]
        return pred_boxes3d, pred_scores, pred_labels, pred_vel, det_boxes3d, det_vel, traj, valid_mask

    def get_targets(self, pred_boxes3d, targets, global_candidates):
        reg_mask_list = []
        gt_boxes_list = []
        gt_future_list = []
        gt_future_list_local = []
        batch_size = pred_boxes3d.shape[0]
        num_track = global_candidates.shape[2]
        for i in range(batch_size):
            if pred_boxes3d[i].shape[0] > 0 and targets[i]['gt_boxes'].shape[0] > 0:
                rois = global_candidates[i][..., :7].reshape(-1, 7)
                track_iou = boxes_iou3d_gpu(rois, targets[i]['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]])
                max_iou, track_id = track_iou.max(-1)
                reg_mask = max_iou > 0.5
                ordered_gt_boxes = targets[i]['gt_boxes'][track_id][:, [0, 1, 2, 3, 4, 5, -1]]
                pred_gt_boxes = targets[i]['future_gt_boxes'].reshape(-1, targets[i]['gt_boxes'].shape[0], 9)
                track_id_vel = track_id.reshape(-1, global_candidates.shape[3])[:, 0].reshape(-1)
                ordered_future_gt_boxes = pred_gt_boxes[:, track_id_vel][None, :, :, [0, 1, 2, 3, 4, 5, -1]].unsqueeze(3)
                ordered_future_gt_boxes_local = transform_trajs_to_local_coords(ordered_future_gt_boxes, center_xyz=ordered_future_gt_boxes[:, 0:1, :, 0:1, :3], center_heading=ordered_future_gt_boxes[:, 0:1, :, 0:1, 6], heading_index=6)[0].squeeze(3)
                gt_boxes_list.append(ordered_gt_boxes)
                gt_future_list_local.append(ordered_future_gt_boxes_local[:, 1:])
                gt_future_list.append(ordered_future_gt_boxes[:, 1:])
                reg_mask_list.append(reg_mask)
            else:
                pad = global_candidates[i][..., :7].reshape(-1, 7)
                reg_mask_list.append(torch.zeros([pad.shape[0]]).bool())
                gt_boxes_list.append(torch.zeros([pad.shape[0], 7]))
                gt_future_list.append(torch.zeros([1, self.num_future, num_track, 1, 7]))
                gt_future_list_local.append(torch.zeros([1, self.num_future, num_track, 7]))
        gt_boxes = torch.cat(gt_boxes_list)
        gt_future_boxes = torch.cat(gt_future_list)
        gt_future_traj_local = torch.cat(gt_future_list_local)
        gt_future_traj_local = gt_future_traj_local[:, :, :, None, [0, 1, 6]]
        fg_reg_mask = torch.cat(reg_mask_list)
        fg_reg_mask = fg_reg_mask.reshape(batch_size, 1, num_track, self.num_hypo)
        return gt_boxes, gt_future_boxes, gt_future_traj_local, fg_reg_mask

    def class_agnostic_nms(self, pred_boxes3d, pred_scores, nms_thresh=0.1, score_thresh=None, nms_pre_maxsize=4096, nms_post_maxsize=500):
        box_preds = pred_boxes3d
        scores = pred_scores
        if score_thresh is not None:
            scores_mask = scores >= score_thresh
            scores = scores[scores_mask]
            box_preds = box_preds[scores_mask]
        rank_scores_nms, indices = torch.topk(scores, k=min(nms_pre_maxsize, scores.shape[0]))
        box_preds_nms = box_preds[indices][:, :7]
        if box_preds_nms.shape[0] > 0:
            keep_idx, _ = nms_gpu(box_preds_nms, rank_scores_nms, thresh=nms_thresh)
            selected = indices[keep_idx[:nms_post_maxsize]]
            if score_thresh is not None:
                original_idxs = scores_mask.nonzero().view(-1)
                selected = original_idxs[selected]
            return selected
        else:
            return torch.tensor([]).long()

    @staticmethod
    def reorder_rois(pred_bboxes):
        num_max_rois = max([len(bbox) for bbox in pred_bboxes])
        num_max_rois = max(1, num_max_rois)
        ordered_bboxes = torch.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]])
        valid_mask = np.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]])
        for bs_idx in range(ordered_bboxes.shape[0]):
            ordered_bboxes[bs_idx, :len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
            valid_mask[bs_idx, :len(pred_bboxes[bs_idx])] = 1
        return ordered_bboxes, torch.from_numpy(valid_mask).bool()

    def generate_trajectory(self, proposals_list):
        cur_batch_boxes = proposals_list[:, 0, :, :]
        trajectory_rois = torch.zeros_like(cur_batch_boxes[:, None, :, :]).repeat(1, proposals_list.shape[1], 1, 1)
        trajectory_rois[:, 0, :, :] = proposals_list[:, 0, :, :]
        valid_length = torch.zeros([trajectory_rois.shape[0], trajectory_rois.shape[1], trajectory_rois.shape[2]])
        valid_length[:, 0] = 1
        num_frames = proposals_list.shape[1]
        for i in range(1, num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:, :, 0:2] = trajectory_rois[:, i - 1, :, 0:2] - 0.1 * trajectory_rois[:, i - 1, :, 6:8]
            frame[:, :, 2:] = trajectory_rois[:, i - 1, :, 2:]
            for bs_idx in range(proposals_list.shape[0]):
                iou3d = boxes_iou3d_gpu(frame[bs_idx, :, [0, 1, 2, 3, 4, 5, -1]], proposals_list[bs_idx, i, :, [0, 1, 2, 3, 4, 5, -1]])
                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)
                fg_inds = (max_overlaps >= 0.5).nonzero().view(-1)
                valid_length[bs_idx, i, fg_inds] = 1
                trajectory_rois[bs_idx, i, fg_inds, :] = proposals_list[bs_idx, i, traj_assignment[fg_inds]]
        return trajectory_rois, valid_length

    def forward(self, batched_inputs):
        batch_size = len(batched_inputs)
        targets = [bi[1]['annotations'] for bi in batched_inputs]
        for key in ['gt_boxes', 'future_gt_boxes', 'difficulty', 'num_points_in_gt', 'labels']:
            for i in range(batch_size):
                targets[i][key] = torch.tensor(targets[i][key], device=self.device)
        pred_boxes3d = [torch.from_numpy(bi[1]['annotations']['pred_boxes3d']) for bi in batched_inputs]
        pred_scores = [torch.from_numpy(bi[1]['annotations']['pred_scores']) for bi in batched_inputs]
        pred_labels = [torch.from_numpy(bi[1]['annotations']['pred_labels']) for bi in batched_inputs]
        outputs = self.organize_proposals(batch_size, pred_boxes3d, pred_scores, pred_labels)
        pred_boxes3d, pred_scores, pred_labels, pred_vel, det_boxes3d, det_vel, traj, valid_mask = outputs
        num_track = pred_boxes3d.shape[1]
        batch_size = pred_boxes3d.shape[0]
        if num_track > 0 and det_boxes3d.shape[1] > 0:
            loss_dict = {}
            trajectory_hypothese, global_candidates, _ = self.genereate_trajcetory_hypotheses(pred_boxes3d, det_boxes3d, traj, self.num_hypo)
            future_traj_local = self.get_pred_traj(trajectory_hypothese, valid_mask, pred_vel, det_vel)
            gt_boxes, gt_future_boxes, gt_future_traj_local, fg_reg_mask = self.get_targets(pred_boxes3d, targets, global_candidates)
            valid_gt_mask = (gt_future_boxes[..., 3:6].sum(-1) > 0).repeat(1, 1, 1, self.num_hypo)
            valid_fg_mask = fg_reg_mask.repeat(1, self.num_future, 1, 1)
            valid_mask = torch.logical_and(valid_gt_mask, valid_fg_mask)
            loss = self.reg_loss_func(future_traj_local[..., [0, 1, 6]], gt_future_traj_local.repeat(1, 1, 1, self.num_hypo, 1))[valid_mask]
            loss = loss.sum() / valid_mask.sum()
            if gt_boxes.shape[0] > 0:
                loss_dict.update({'loss': loss})
            else:
                loss_dict = {'loss': torch.tensor([0.0]).reshape(1, -1)}
        else:
            loss_dict = {'loss': torch.tensor([0.0]).reshape(1, -1)}
        return loss_dict


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvFCHead,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Empty,
     lambda: ([], {}),
     lambda: ([], {})),
    (FFN,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FFNLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupNorm,
     lambda: ([], {'num_channels': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Head,
     lambda: ([], {'num_input': 4, 'num_pred': 4, 'num_cls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LastLevelMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm1d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool2dSamePadding,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp1d,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Mlp2d,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PointNetfeat,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Projection,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'act_layer': torch.nn.ReLU}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {})),
    (RegLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SeparableConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderGlobalLocal,
     lambda: ([], {'encoder_layer': [torch.nn.ReLU()], 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'config': SimpleNamespace(), 'd_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (TransformerEncoderLayerGlobalLocal,
     lambda: ([], {'config': SimpleNamespace(), 'd_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VoxelMeanFeatureExtractor,
     lambda: ([], {'num_input_features': 4}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4])], {})),
    (WeightedSmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

