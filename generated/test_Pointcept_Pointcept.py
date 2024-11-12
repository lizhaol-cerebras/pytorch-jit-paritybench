
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


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from typing import Tuple


import torch.nn as nn


import time


import numpy as np


from copy import deepcopy


from torch.utils.data import Dataset


from functools import partial


import torch.utils.data


from collections.abc import Sequence


import pandas as pd


from itertools import repeat


import copy


import math


import random


import numbers


import scipy


import scipy.ndimage


import scipy.interpolate


import scipy.stats


from collections.abc import Mapping


from torch.utils.data.dataloader import default_collate


from torch.nn.parallel import DistributedDataParallel


import torch.distributed as dist


from uuid import uuid4


from collections import OrderedDict


import logging


import torch.multiprocessing as mp


import torch.nn.functional as F


from typing import Optional


from itertools import filterfalse


from torch.nn.modules.loss import _Loss


from itertools import chain


from typing import List


from typing import Dict


from torch.utils.checkpoint import checkpoint


from typing import Union


import functools


import torch.backends.cudnn as cudnn


from collections import defaultdict


import warnings


from collections import abc


import torch.optim.lr_scheduler as lr_scheduler


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(binary[tuple(slicing)], (k, 0), mode='constant', value=0)
    return shifted


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode='floor')
    return gray


def encode(locs, num_dims, num_bits):
    """Decode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the
     input, excluding the last dimension, which needs to be num_dims.
    """
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    if orig_shape[-1] != num_dims:
        raise ValueError('\n      The shape of locs was surprising in that the last dimension was of size\n      %d, but num_dims=%d.  These need to be equal.\n      ' % (orig_shape[-1], num_dims))
    if num_dims * num_bits > 63:
        raise ValueError("\n      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded\n      into a int64.  Are you sure you need that many points on your Hilbert\n      curve?\n      " % (num_dims, num_bits, num_dims * num_bits))
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = locs_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[..., -num_bits:]
    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], mask[:, None])
            to_flip = torch.logical_and(torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1), torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]))
            gray[:, dim, bit + 1:] = torch.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), 'constant', 0)
    hh_uint8 = (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8)
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()
    return hh_uint64


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'batch' not in self.keys() and 'offset' in self.keys():
            self['batch'] = offset2batch(self.offset)
        elif 'offset' not in self.keys() and 'batch' in self.keys():
            self['offset'] = batch2offset(self.batch)

    def serialization(self, order='z', depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert 'batch' in self.keys()
        if 'grid_coord' not in self.keys():
            assert {'grid_size', 'coord'}.issubset(self.keys())
            self['grid_coord'] = torch.div(self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode='trunc').int()
        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self['serialized_depth'] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16
        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(dim=1, index=order, src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]
        self['serialized_code'] = code
        self['serialized_order'] = order
        self['serialized_inverse'] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {'feat', 'batch'}.issubset(self.keys())
        if 'grid_coord' not in self.keys():
            assert {'grid_size', 'coord'}.issubset(self.keys())
            self['grid_coord'] = torch.div(self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode='trunc').int()
        if 'sparse_shape' in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(features=self.feat, indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(), spatial_shape=sparse_shape, batch_size=self.batch[-1].tolist() + 1)
        self['sparse_shape'] = sparse_shape
        self['sparse_conv_feat'] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert ocnn is not None, 'Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn.'
        assert {'grid_coord', 'feat', 'batch'}.issubset(self.keys())
        if depth is None:
            if 'depth' in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self['depth'] = depth
        assert depth <= 16
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(points=coord, features=self.feat, batch_id=self.batch.unsqueeze(-1), batch_size=self.batch[-1] + 1)
        octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth, batch_size=self.batch[-1] + 1, device=coord.device)
        octree.build_octree(point)
        octree.construct_all_neigh()
        self['octree'] = octree

