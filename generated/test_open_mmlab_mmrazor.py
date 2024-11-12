
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


import warnings


from typing import Any


from typing import Dict


from typing import Generator


import numpy as np


import torch


import copy


from typing import Optional


from typing import Sequence


from typing import List


from typing import Union


import torch.nn as nn


from typing import Tuple


from torch.utils.data import DataLoader


import random


import time


import math


from abc import abstractmethod


import torch.distributed as dist


from torch import Tensor


from torch.nn.modules.batchnorm import _BatchNorm


from torch import distributed as torch_dist


import types


from torch import distributed as dist


import torch.nn.functional as F


from typing import Type


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from typing import OrderedDict


from torch import nn


from collections import OrderedDict


from abc import ABCMeta


from math import log


from typing import Iterator


from typing import Set


from torch.nn import Module


from typing import Callable


import logging


from functools import partial


from torch.nn import LayerNorm


from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm


from itertools import repeat


from typing import Iterable


from torch.nn.modules.conv import _ConvNd


from abc import ABC


from torch.nn import functional


import torch.utils.checkpoint as cp


from inspect import signature


from torch.nn.parameter import Parameter


import inspect


from itertools import product


import abc


from typing import TypeVar


from typing import Generic


from torch.nn import ModuleList


from collections import Counter


import functools


from types import FunctionType


from types import ModuleType


import torch.nn


from abc import abstractclassmethod


import torch.optim as optim


import re


from torch.nn import Conv2d


from torch.nn import Linear


from torch.nn.modules import GroupNorm


from torch.nn.modules.batchnorm import _NormBase


from copy import deepcopy


from torch._C import ScriptObject


from collections import namedtuple


import torch.multiprocessing as mp


import matplotlib.pyplot as plt


import torchvision


import torchvision.datasets as datasets


import torchvision.transforms as transforms


from torch.utils.data import Dataset as TorchDataset


from torch.utils.data import DistributedSampler


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp.api import ShardingStrategy


from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload


from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


import string


from torch.utils.data import Dataset


from torch.optim import SGD


from torch.nn import GroupNorm


from torch.nn import Sequential


from copy import copy


from torch.nn import Parameter


from torch.nn.intrinsic.qat import ConvBnReLU2d


from logging import Logger


class torch_setting:
    """Set the default torch dtype setting."""

    def __init__(self, dtype=None) ->None:
        self.original_dtype = torch.get_default_dtype()
        self.dtype = dtype

    def __enter__(self):
        """Enter."""
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit."""
        torch.set_default_dtype(self.original_dtype)


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    """matmul248 function with matmul_248_kernel."""
    with torch.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    """transpose_matmul248 function with transpose_matmul_248_kernel."""
    with torch.device(input.device):
        output_dim = qweight.shape[0] * 32 // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']),)
        transpose_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits, maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):
    """Custom QuantLinearFunction."""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        """Custom forward."""
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Custom backward."""
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class Quantizer(nn.Module):
    """Quantizer for some basic quantization functions."""

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8, trits=False):
        """Configure qconfig."""
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale)

    def _quantize(self, x, scale, zero, maxq):
        """Fakequant."""
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, weight=False):
        """Observe the specified data and calculate the qparams."""
        dev = x.device
        self.maxq = self.maxq
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        """Fakequant."""
        if self.ready():
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        """Whether is enabled."""
        return self.maxq > 0

    def ready(self):
        """Whether is ready."""
        return torch.all(self.scale != 0)


class BatchNormWrapper(nn.Module):
    """Wrapper for BatchNorm.

    For more information, Please refer to
    https://github.com/NVIDIA/apex/issues/121
    """

    def __init__(self, m):
        super(BatchNormWrapper, self).__init__()
        self.m = m
        self.m.eval()

    def forward(self, x):
        """Convert fp16 to fp32 when forward."""
        input_type = x.dtype
        x = self.m(x.float())
        return x


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise (stride=2).

    Args:
        in_channels (int): number of channels of input tensor.
        out_channels (int): number of channels of output tensor.
        act_cfg (Dict): config to build activation layer.
        norm_cfg (Dict): config to build normalization layer.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', act_cfg: 'Dict'=dict(type='ReLU'), norm_cfg: 'Dict'=dict(type='BN')) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.relu = build_activation_layer(self.act_cfg)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = build_norm_layer(self.norm_cfg, self.out_channels)[1]

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward with factorized reduce."""
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class StandardConv(nn.Module):
    """Standard Convolution in Darts. Basic structure is ReLU-Conv-BN.

    Args:
        in_channels (int): number of channels of input tensor.
        out_channels (int): number of channels of output tensor.
        kernel_size (Union[int, Tuple]): size of the convolving kernel.
        stride (Union[int, Tuple]): controls the stride for the
            cross-correlation, a single number or a one-element tuple.
            Default to 1.
        padding (Union[str, int, Tuple]): Padding added to both sides
            of the input. Default to 0.
        act_cfg (Dict): config to build activation layer.
        norm_cfg (Dict): config to build normalization layer.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple]', stride: 'Union[int, Tuple]'=1, padding: 'Union[str, int, Tuple]'=0, act_cfg: 'Dict'=dict(type='ReLU'), norm_cfg: 'Dict'=dict(type='BN')) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(build_activation_layer(self.act_cfg), nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False), build_norm_layer(self.norm_cfg, self.out_channels)[1])

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward the standard convolution."""
        return self.net(x)


class LinearHeadForTest(Module):

    def __init__(self, in_channel, num_class=1000) ->None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channel, num_class)

    def forward(self, x):
        pool = self.pool(x).flatten(1)
        return self.linear(pool)


class ModuleWithUntracableMethod(nn.Module):

    def __init__(self, in_channel, out_channel) ->None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x: 'torch.Tensor'):
        x = self.conv(x)
        x = self.untracable_method(x)
        x = self.conv2(x)
        return x

    def untracable_method(self, x):
        if x.sum() > 0:
            x = x * 2
        else:
            x = x * -2
        return x


class UntracableModule(nn.Module):

    def __init__(self, in_channel, out_channel) ->None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x: 'torch.Tensor'):
        x = self.conv(x)
        if x.sum() > 0:
            x = x * 2
        else:
            x = x * -2
        x = self.conv2(x)
        return x


def untracable_function(x: 'torch.Tensor'):
    if x.sum() > 0:
        x = x - 1
    else:
        x = x + 1
    return x


class UntracableBackBone(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 2)
        self.untracable_module = UntracableModule(16, 8)
        self.module_with_untracable_method = ModuleWithUntracableMethod(8, 16)

    def forward(self, x):
        x = self.conv(x)
        x = untracable_function(x)
        x = self.untracable_module(x)
        x = self.module_with_untracable_method(x)
        return x


class UntracableModel(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.backbone = UntracableBackBone()
        self.head = LinearHeadForTest(16, 1000)

    def forward(self, x):
        return self.head(self.backbone(x))


class Node(nn.Module):
    """Node structure of DARTS.

    Args:
        node_id (str): key of the node.
        num_prev_nodes (int): number of previous nodes.
        channels (int): number of channels of current node.
        num_downsample_nodes (int): index of downsample node.
        mutable_cfg (Dict): config of `DiffMutableModule`.
        route_cfg (Dict): config of `DiffChoiceRoute`.
    """

    def __init__(self, node_id: 'str', num_prev_nodes: 'int', channels: 'int', num_downsample_nodes: 'int', mutable_cfg: 'Dict', route_cfg: 'Dict') ->None:
        super().__init__()
        edges = nn.ModuleDict()
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_nodes else 1
            edge_id = f'{node_id}_p{i}'
            module_kwargs = dict(in_channels=channels, out_channels=channels, stride=stride)
            mutable_cfg.update(module_kwargs=module_kwargs)
            mutable_cfg.update(alias=edge_id)
            edges.add_module(edge_id, MODELS.build(mutable_cfg))
        route_cfg.update(alias=node_id)
        route_cfg.update(edges=edges)
        self.route = MODELS.build(route_cfg)

    def forward(self, prev_nodes: 'Union[List[Tensor], Tuple[Tensor]]') ->Tensor:
        """Forward with the previous nodes list."""
        return self.route(prev_nodes)


class Cell(nn.Module):
    """Darts cell structure.

    Args:
        num_nodes (int): number of nodes.
        channels (int): number of channels of current cell.
        prev_channels (int): number of channel of previous input.
        prev_prev_channels (int): number of channel of previous previous input.
        reduction (bool): whether to reduce the feature map size.
        prev_reduction (bool): whether to reduce the previous feature map size.
        mutable_cfg (Optional[Dict]): config of `DiffMutableModule`.
        route_cfg (Optional[Dict]): config of `DiffChoiceRoute`.
        act_cfg (Dict): config to build activation layer.
            Defaults to dict(type='ReLU').
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(self, num_nodes: 'int', channels: 'int', prev_channels: 'int', prev_prev_channels: 'int', reduction: 'bool', prev_reduction: 'bool', mutable_cfg: 'Dict', route_cfg: 'Dict', act_cfg: 'Dict'=dict(type='ReLU'), norm_cfg: 'Dict'=dict(type='BN')) ->None:
        super().__init__()
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.reduction = reduction
        self.num_nodes = num_nodes
        if prev_reduction:
            self.preproc0 = FactorizedReduce(prev_prev_channels, channels, self.act_cfg, self.norm_cfg)
        else:
            self.preproc0 = StandardConv(prev_prev_channels, channels, 1, 1, 0, self.act_cfg, self.norm_cfg)
        self.preproc1 = StandardConv(prev_channels, channels, 1, 1, 0, self.act_cfg, self.norm_cfg)
        self.nodes = nn.ModuleList()
        for depth in range(2, self.num_nodes + 2):
            if reduction:
                node_id = f'reduce_n{depth}'
                num_downsample_nodes = 2
            else:
                node_id = f'normal_n{depth}'
                num_downsample_nodes = 0
            self.nodes.append(Node(node_id, depth, channels, num_downsample_nodes, mutable_cfg, route_cfg))

    def forward(self, s0: 'Tensor', s1: 'Tensor') ->Tensor:
        """Forward with the outputs of previous previous cell and previous
        cell."""
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.nodes:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)
        return torch.cat(tensors[2:], dim=1)


class AuxiliaryModule(nn.Module):
    """Auxiliary head in 2/3 place of network to let the gradient flow well.

    Args:
        in_channels (int): number of channels of inputs.
        base_channels (int): number of middle channels of the auxiliary module.
        out_channels (int): number of channels of outputs.
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(self, in_channels: 'int', base_channels: 'int', out_channels: 'int', norm_cfg: 'Dict'=dict(type='BN')) ->None:
        super().__init__()
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(nn.ReLU(), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False), build_norm_layer(self.norm_cfg, base_channels)[1], nn.ReLU(inplace=True), nn.Conv2d(base_channels, out_channels, kernel_size=2, bias=False), build_norm_layer(self.norm_cfg, out_channels)[1], nn.ReLU(inplace=True))

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward the auxiliary module."""
        return self.net(x)


class DartsBackbone(nn.Module):
    """Backbone of Differentiable Architecture Search (DARTS).

    Args:
        in_channels (int): number of channels of input tensor.
        base_channels (int): number of middle channels.
        mutable_cfg (Optional[Dict]): config of `DiffMutableModule`.
        route_cfg (Optional[Dict]): config of `DiffChoiceRoute`.
        num_layers (Optional[int]): number of layers.
            Defaults to 8.
        num_nodes (Optional[int]): number of nodes.
            Defaults to 4.
        stem_multiplier (Optional[int]): multiplier for stem.
            Defaults to 3.
        out_indices (tuple, optional): output indices for auxliary module.
            Defaults to (7, ).
        auxliary (bool, optional): whether use auxliary module.
            Defaults to False.
        aux_channels (Optional[int]): number of middle channels of
            auxliary module. Defaults to None.
        aux_out_channels (Optional[int]): number of output channels of
            auxliary module. Defaults to None.
        act_cfg (Dict): config to build activation layer.
            Defaults to dict(type='ReLU').
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(self, in_channels: 'int', base_channels: 'int', mutable_cfg: 'Dict', route_cfg: 'Dict', num_layers: 'int'=8, num_nodes: 'int'=4, stem_multiplier: 'int'=3, out_indices: 'Union[Tuple, List]'=(7,), auxliary: 'bool'=False, aux_channels: 'Optional[int]'=None, aux_out_channels: 'Optional[int]'=None, act_cfg: 'Dict'=dict(type='ReLU'), norm_cfg: 'Dict'=dict(type='BN')) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.stem_multiplier = stem_multiplier
        self.out_indices = out_indices
        assert self.out_indices[-1] == self.num_layers - 1
        if auxliary:
            assert aux_channels is not None
            assert aux_out_channels is not None
            self.aux_channels = aux_channels
            self.aux_out_channels = aux_out_channels
            self.auxliary_indice = 2 * self.num_layers // 3
        else:
            self.auxliary_indice = -1
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = self.stem_multiplier * self.base_channels
        stem_norm_cfg = copy.deepcopy(self.norm_cfg)
        stem_norm_cfg.update(dict(affine=True))
        self.stem = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=False), build_norm_layer(self.norm_cfg, self.out_channels)[1])
        prev_prev_channels = self.out_channels
        prev_channels = self.out_channels
        self.out_channels = self.base_channels
        self.cells = nn.ModuleList()
        prev_reduction, reduction = False, False
        for i in range(self.num_layers):
            prev_reduction, reduction = reduction, False
            if i in [self.num_layers // 3, 2 * self.num_layers // 3]:
                self.out_channels *= 2
                reduction = True
            cell = Cell(self.num_nodes, self.out_channels, prev_channels, prev_prev_channels, reduction, prev_reduction, mutable_cfg, route_cfg, self.act_cfg, self.norm_cfg)
            self.cells.append(cell)
            prev_prev_channels = prev_channels
            prev_channels = self.out_channels * self.num_nodes
            if i == self.auxliary_indice:
                self.auxliary_module = AuxiliaryModule(prev_channels, self.aux_channels, self.aux_out_channels, self.norm_cfg)

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward the darts backbone."""
        outs = []
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i in self.out_indices:
                outs.append(s1)
            if i == self.auxliary_indice and self.training:
                aux_feature = self.auxliary_module(s1)
                outs.insert(0, aux_feature)
        return tuple(outs)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels
        self.norm1 = nn.BatchNorm2d(self.mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, 1)
        self.conv2 = nn.Conv2d(self.mid_channels, out_channels, 1)
        self.relu = nn.ReLU6()
        self.drop_path = nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.drop_path(out)
            out += identity
            return out
        out = _inner_forward(x)
        out = self.relu(out)
        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck.
        droprate (float, optional): droprate of the layer. Defaults to 0.
        stride (int): stride of the first block. Default: 1.
        conv_cfg (Dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (Dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, block: 'nn.Module', num_blocks: 'int', in_channels: 'int', out_channels: 'int', expansion: 'int', droprate: 'float'=0, stride: 'int'=1, conv_cfg: 'Dict'=None, norm_cfg: 'Dict'=dict(type='BN'), **kwargs):
        self.block = block
        self.droprate = droprate
        self.expansion = expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        layers = []
        layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super(ResLayer, self).__init__(*layers)


class Normalize(nn.Module):
    """normalization layer.

    Args:
        power (int, optional): power. Defaults to 2.
    """

    def __init__(self, power: 'int'=2) ->None:
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out

