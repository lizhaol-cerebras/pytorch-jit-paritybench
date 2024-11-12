
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


import logging


import time


import numpy as np


import torch


from torch.utils._pytree import tree_map


import torch.nn as nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from inspect import getmembers


from inspect import isfunction


import abc


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn.functional as F


from typing import Any


from typing import Callable


from typing import Optional


from typing import Tuple


from typing import List


import copy


from collections import defaultdict


import collections


import functools


import torch.optim


from torch.utils._pytree import tree_flatten


import enum


import random


from matplotlib import cm


from torch import Tensor


from typing import Dict


from typing import Literal


from typing import Type


from typing import Union


from torch.nn import Parameter


import typing


import torch.distributed as dist


from torch.cuda.amp.grad_scaler import GradScaler


from torch.nn.parallel import DistributedDataParallel as DDP


class Funcs:

    def __init__(self, funcs):
        for func in funcs:
            self.__dict__[func.__name__] = func


class BaseBackend(metaclass=abc.ABCMeta):

    def __init__(self):
        backend_name = self._get_backend_name()
        self.funcs = Funcs(self._get_funcs(backend_name))
        self.device = None

    @abc.abstractmethod
    def synchronize(self):
        pass

    @abc.abstractmethod
    def _get_backend_name(self):
        pass

    def _get_funcs(self, backend_name):
        """get functions defined in different backend"""
        funcs = []
        for name, func in getmembers(backend_name):
            if not (str(name).startswith('__') and str(name).endswith('__')):
                funcs.append(func)
        return funcs

    def get_generator(self):
        return self.device.get_generator()


class BaseDevice(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def synchronize(self):
        pass

    @abc.abstractmethod
    def get_generator(self):
        pass


class IntelDevice(BaseDevice):

    def __init__(self) ->None:
        super().__init__()
        self.synchronize = torch.xpu.synchronize
        torch.set_default_tensor_type('torch.xpu.FloatTensor')

    def synchronize(self):
        self.synchronize()

    def get_generator(self):
        return torch.xpu.Generator()


class NvidiaDevice(BaseDevice):

    def synchronize(self):
        torch.cuda.synchronize()

    def get_generator(self):
        return None


class DeviceFactory:
    device_name2class = {'intel': IntelDevice, 'nvidia': NvidiaDevice}

    @classmethod
    def create_device(cls, device_name):
        if device_name not in cls.device_name2class.keys():
            raise Exception('unsupported gpu device type...')
        return cls.device_name2class[device_name]()


class CUDABackend(BaseBackend):

    def __init__(self):
        super().__init__()
        self.gpu_vendor = 'nvidia'
        self.device = DeviceFactory.create_device(self.gpu_vendor)

    def _get_backend_name(self):
        funcs = []
        try:
            backend_name = __import__('_cuda_backend')
        except ImportError:
            backend_name = __import__('_cuda_backend')
        return backend_name

    def synchronize(self):
        self.device.synchronize()


class DPCPPBackend(BaseBackend):

    def __init__(self):
        super().__init__()
        self.gpu_vendor = self.funcs.get_gpu_vendor()
        self.device = DeviceFactory.create_device(self.gpu_vendor)

    def _get_backend_name(self):
        try:
            backend_name = __import__('extensions.dpcpp._dpcpp_backend', fromlist=['extensions.dpcpp'])
        except ImportError:
            backend_name = __import__('extensions.dpcpp._dpcpp_backend', fromlist=['extensions.dpcpp'])
        return backend_name

    def synchronize(self):
        self.device.synchronize()

    def get_gpu_vendor(self):
        return self.gpu_vendor


_gridtype_to_id = {'hash': 0, 'tiled': 1}


_interp_to_id = {'linear': 0, 'smoothstep': 1}


class _grid_encode(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, backend, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        inputs = inputs.contiguous()
        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        S = np.log2(per_level_scale)
        H = base_resolution
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None
        backend.synchronize()
        backend.funcs.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        ctx.backend = backend
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        backend = ctx.backend
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None
        backend.synchronize()
        backend.funcs.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)
        if dy_dx is not None:
            grad_inputs = grad_inputs
        return None, grad_inputs, grad_embeddings, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: 'int' = 256
    net_depth_viewdirs: 'int' = 2
    net_width_viewdirs: 'int' = 256
    skip_layer_dir: 'int' = 0
    num_rgb_channels: 'int' = 3
    deg_view: 'int' = 4
    use_reflections: 'bool' = False
    use_directional_enc: 'bool' = False
    enable_pred_roughness: 'bool' = False
    roughness_bias: 'float' = -1.0
    use_diffuse_color: 'bool' = False
    use_specular_tint: 'bool' = False
    use_n_dot_v: 'bool' = False
    bottleneck_noise: 'float' = 0.0
    density_bias: 'float' = -1.0
    density_noise: 'float' = 0.0
    rgb_premultiplier: 'float' = 1.0
    rgb_bias: 'float' = 0.0
    rgb_padding: 'float' = 0.001
    enable_pred_normals: 'bool' = False
    disable_density_normals: 'bool' = False
    disable_rgb: 'bool' = False
    warp_fn = 'contract'
    num_glo_features: 'int' = 0
    num_glo_embeddings: 'int' = 1000
    scale_featurization: 'bool' = False
    grid_num_levels: 'int' = 10
    grid_level_interval: 'int' = 2
    grid_level_dim: 'int' = 4
    grid_base_resolution: 'int' = 16
    grid_disired_resolution: 'int' = 8192
    grid_log2_hashmap_size: 'int' = 21
    net_width_glo: 'int' = 128
    net_depth_glo: 'int' = 2

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        if self.use_reflections and not (self.enable_pred_normals or not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(direction, min_deg=0, max_deg=self.deg_view, append_identity=True)
            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        self.grid_num_levels = int(np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3, num_levels=self.grid_num_levels, level_dim=self.grid_level_dim, base_resolution=self.grid_base_resolution, desired_resolution=self.grid_disired_resolution, log2_hashmap_size=self.grid_log2_hashmap_size, gridtype='hash', align_corners=False)
        last_dim = self.encoder.output_dim
        if self.scale_featurization:
            last_dim += self.encoder.num_levels
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64), nn.ReLU(), nn.Linear(64, 1 if self.disable_rgb else self.bottleneck_width))
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)
        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)
            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)
            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0
            last_dim_rgb += dim_dir_enc
            if self.use_n_dot_v:
                last_dim_rgb += 1
            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f'lin_glo_{i}', nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f'lin_glo_{self.net_depth_glo - 1}', nn.Linear(last_dim_glo, self.bottleneck_width * 2))
            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f'lin_second_stage_{i}', lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, stds, rand=False, no_warp=False):
        """Helper function to output density."""
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            bound = 2
            means = means / bound
            stds = stds / bound
        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
        features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1), self.encoder.idx, torch.zeros(self.grid_num_levels, device=weights.device), self.grid_num_levels, reduce='mean')
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
            features = torch.cat([features, featurized_w], dim=-1)
        x = self.density_layer(features)
        raw_density = x[..., 0]
        if rand and self.density_noise > 0:
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, means.mean(dim=-2)

    def forward(self, rand, means, stds, viewdirs=None, imageplane=None, glo_vec=None, exposure=None, no_warp=False):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        if self.disable_density_normals:
            raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(outputs=raw_density, inputs=means, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            normals = -ref_utils.l2_normalize(raw_grad_density)
        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals
        density = F.softplus(raw_density + self.density_bias)
        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)
                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))
                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = F.softplus(raw_roughness + self.roughness_bias)
                if self.bottleneck_width > 0:
                    bottleneck = x
                    if rand and self.bottleneck_noise > 0:
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f'lin_glo_{i}')(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :], bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift
                    x = [bottleneck]
                else:
                    x = []
                if self.use_reflections:
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    dir_enc = torch.broadcast_to(dir_enc[..., None, :], bottleneck.shape[:-1] + (dir_enc.shape[-1],))
                x.append(dir_enc)
                if self.use_n_dot_v:
                    dotprod = torch.sum(normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)
                x = torch.cat(x, dim=-1)
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f'lin_second_stage_{i}')(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            rgb = torch.sigmoid(self.rgb_premultiplier * self.rgb_layer(x) + self.rgb_bias)
            if self.use_diffuse_color:
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        return dict(coord=means_contract, density=density, rgb=rgb, raw_grad_density=raw_grad_density, grad_pred=grad_pred, normals=normals, normals_pred=normals_pred, roughness=roughness)

