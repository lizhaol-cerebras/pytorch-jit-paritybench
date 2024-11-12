
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


import torch.nn.functional as F


import collections


import functools


import math


from typing import Callable


from typing import Optional


import torch.nn as nn


from typing import List


from typing import Union


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import time


import itertools


import random


from typing import Sequence


from torch.utils.data._utils.collate import collate


from torch.utils.data._utils.collate import default_collate_fn_map


from typing import Tuple


from torch import Tensor


from torch.utils.cpp_extension import _get_build_directory


from torch.utils.cpp_extension import load


from typing import Any


from typing import Mapping


import warnings


from typing import Dict


class MLP(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int'=None, net_depth: 'int'=8, net_width: 'int'=256, skip_layer: 'int'=4, hidden_init: 'Callable'=nn.init.xavier_uniform_, hidden_activation: 'Callable'=nn.ReLU(), output_enabled: 'bool'=True, output_init: 'Optional[Callable]'=nn.init.xavier_uniform_, output_activation: 'Optional[Callable]'=nn.Identity(), bias_enabled: 'bool'=True, bias_init: 'Callable'=nn.init.zeros_):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init
        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if self.skip_layer is not None and i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(in_features, self.output_dim, bias=bias_enabled)
        else:
            self.output_dim = in_features
        self.initialize()

    def initialize(self):

        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)
        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)
            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if self.skip_layer is not None and i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, net_depth=0, **kwargs)


class NerfMLP(nn.Module):

    def __init__(self, input_dim: 'int', condition_dim: 'int', net_depth: 'int'=8, net_width: 'int'=256, skip_layer: 'int'=4, net_depth_condition: 'int'=1, net_width_condition: 'int'=128):
        super().__init__()
        self.base = MLP(input_dim=input_dim, net_depth=net_depth, net_width=net_width, skip_layer=skip_layer, output_enabled=False)
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)
        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(input_dim=net_width + condition_dim, output_dim=3, net_depth=net_depth_condition, net_width=net_width_condition, skip_layer=None)
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view([num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: 'bool'=True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer('scales', torch.tensor([(2 ** i) for i in range(min_deg, max_deg)]))

    @property
    def latent_dim(self) ->int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(x[Ellipsis, None, :] * self.scales[:, None], list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim])
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):

    def __init__(self, net_depth: 'int'=8, net_width: 'int'=256, skip_layer: 'int'=4, net_depth_condition: 'int'=1, net_width_condition: 'int'=128) ->None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(input_dim=self.posi_encoder.latent_dim, condition_dim=self.view_encoder.latent_dim, net_depth=net_depth, net_width=net_width, skip_layer=skip_layer, net_depth_condition=net_depth_condition, net_width_condition=net_width_condition)

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class TNeRFRadianceField(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp = MLP(input_dim=self.posi_encoder.latent_dim + self.time_encoder.latent_dim, output_dim=3, net_depth=4, net_width=64, skip_layer=2, output_init=functools.partial(torch.nn.init.uniform_, b=0.0001))
        self.nerf = VanillaNeRFRadianceField()

    def query_opacity(self, x, timestamps, step_size):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        x = x + self.warp(torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1))
        return self.nerf.query_density(x)

    def forward(self, x, t, condition=None):
        x = x + self.warp(torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1))
        return self.nerf(x, condition=condition)


class NDRTNeRFRadianceField(nn.Module):
    """Invertble NN from https://arxiv.org/pdf/2206.15258.pdf"""

    def __init__(self) ->None:
        super().__init__()
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp_layers_1 = nn.ModuleList()
        self.time_layers_1 = nn.ModuleList()
        self.warp_layers_2 = nn.ModuleList()
        self.time_layers_2 = nn.ModuleList()
        self.posi_encoder_1 = SinusoidalEncoder(2, 0, 4, True)
        self.posi_encoder_2 = SinusoidalEncoder(1, 0, 4, True)
        for _ in range(3):
            self.warp_layers_1.append(MLP(input_dim=self.posi_encoder_1.latent_dim + 64, output_dim=1, net_depth=2, net_width=128, skip_layer=None, output_init=functools.partial(torch.nn.init.uniform_, b=0.0001)))
            self.warp_layers_2.append(MLP(input_dim=self.posi_encoder_2.latent_dim + 64, output_dim=1 + 2, net_depth=1, net_width=128, skip_layer=None, output_init=functools.partial(torch.nn.init.uniform_, b=0.0001)))
            self.time_layers_1.append(DenseLayer(input_dim=self.time_encoder.latent_dim, output_dim=64))
            self.time_layers_2.append(DenseLayer(input_dim=self.time_encoder.latent_dim, output_dim=64))
        self.nerf = VanillaNeRFRadianceField()

    def _warp(self, x, t_enc, i_layer):
        uv, w = x[:, :2], x[:, 2:]
        dw = self.warp_layers_1[i_layer](torch.cat([self.posi_encoder_1(uv), self.time_layers_1[i_layer](t_enc)], dim=-1))
        w = w + dw
        rt = self.warp_layers_2[i_layer](torch.cat([self.posi_encoder_2(w), self.time_layers_2[i_layer](t_enc)], dim=-1))
        r = self._euler2rot_2dinv(rt[:, :1])
        t = rt[:, 1:]
        uv = torch.bmm(r, (uv - t)[..., None]).squeeze(-1)
        return torch.cat([uv, w], dim=-1)

    def warp(self, x, t):
        t_enc = self.time_encoder(t)
        x = self._warp(x, t_enc, 0)
        x = x[..., [1, 2, 0]]
        x = self._warp(x, t_enc, 1)
        x = x[..., [2, 0, 1]]
        x = self._warp(x, t_enc, 2)
        return x

    def query_opacity(self, x, timestamps, step_size):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        x = self.warp(x, t)
        return self.nerf.query_density(x)

    def forward(self, x, t, condition=None):
        x = self.warp(x, t)
        return self.nerf(x, condition=condition)

    def _euler2rot_2dinv(self, euler_angle):
        theta = euler_angle.reshape(-1, 1, 1)
        rot = torch.cat((torch.cat((theta.cos(), -theta.sin()), 1), torch.cat((theta.sin(), theta.cos()), 1)), 2)
        return rot


def contract_to_unisphere(x: 'torch.Tensor', aabb: 'torch.Tensor', ord: 'Union[str, int]'=2, eps: 'float'=1e-06, derivative: 'bool'=False):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    if derivative:
        dev = (2 * mag - 1) / mag ** 2 + 2 * x ** 2 * (1 / mag ** 3 - (2 * mag - 1) / mag ** 4)
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5
        return x


class _TruncExp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(self, aabb: 'Union[torch.Tensor, List[float]]', num_dim: 'int'=3, use_viewdirs: 'bool'=True, density_activation: 'Callable'=lambda x: trunc_exp(x - 1), unbounded: 'bool'=False, base_resolution: 'int'=16, max_resolution: 'int'=4096, geo_feat_dim: 'int'=15, n_levels: 'int'=16, log2_hashmap_size: 'int'=19) ->None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)
        self.register_buffer('aabb', aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        per_level_scale = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)).tolist()
        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(n_input_dims=num_dim, encoding_config={'otype': 'Composite', 'nested': [{'n_dims_to_encode': 3, 'otype': 'SphericalHarmonics', 'degree': 4}]})
        self.mlp_base = tcnn.NetworkWithInputEncoding(n_input_dims=num_dim, n_output_dims=1 + self.geo_feat_dim, encoding_config={'otype': 'HashGrid', 'n_levels': n_levels, 'n_features_per_level': 2, 'log2_hashmap_size': log2_hashmap_size, 'base_resolution': base_resolution, 'per_level_scale': per_level_scale}, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 'n_hidden_layers': 1})
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(n_input_dims=(self.direction_encoding.n_output_dims if self.use_viewdirs else 0) + self.geo_feat_dim, n_output_dims=3, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 'n_hidden_layers': 2})

    def query_density(self, x, return_feat: 'bool'=False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = self.mlp_base(x.view(-1, self.num_dim)).view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
        density_before_activation, base_mlp_out = torch.split(x, [1, self.geo_feat_dim], dim=-1)
        density = self.density_activation(density_before_activation) * selector[..., None]
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: 'bool'=True):
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = self.mlp_head(h).reshape(list(embedding.shape[:-1]) + [3])
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(self, positions: 'torch.Tensor', directions: 'torch.Tensor'=None):
        if self.use_viewdirs and directions is not None:
            assert positions.shape == directions.shape, f'{positions.shape} v.s. {directions.shape}'
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density


class NGPDensityField(torch.nn.Module):
    """Instance-NGP Density Field used for resampling"""

    def __init__(self, aabb: 'Union[torch.Tensor, List[float]]', num_dim: 'int'=3, density_activation: 'Callable'=lambda x: trunc_exp(x - 1), unbounded: 'bool'=False, base_resolution: 'int'=16, max_resolution: 'int'=128, n_levels: 'int'=5, log2_hashmap_size: 'int'=17) ->None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer('aabb', aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        per_level_scale = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)).tolist()
        self.mlp_base = tcnn.NetworkWithInputEncoding(n_input_dims=num_dim, n_output_dims=1, encoding_config={'otype': 'HashGrid', 'n_levels': n_levels, 'n_features_per_level': 2, 'log2_hashmap_size': log2_hashmap_size, 'base_resolution': base_resolution, 'per_level_scale': per_level_scale}, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 'n_hidden_layers': 1})

    def forward(self, positions: 'torch.Tensor'):
        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        density_before_activation = self.mlp_base(positions.view(-1, self.num_dim)).view(list(positions.shape[:-1]) + [1])
        density = self.density_activation(density_before_activation) * selector[..., None]
        return density


class AbstractEstimator(nn.Module):
    """An abstract Transmittance Estimator class for Sampling."""

    def __init__(self) ->None:
        super().__init__()
        self.register_buffer('_dummy', torch.empty(0), persistent=False)

    @property
    def device(self) ->torch.device:
        return self._dummy.device

    def sampling(self, *args, **kwargs) ->Any:
        raise NotImplementedError

    def update_every_n_steps(self, *args, **kwargs) ->None:
        raise NotImplementedError


def _enlarge_aabb(aabb, factor: 'float') ->Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


_C = None


class _ExclusiveProd(torch.autograd.Function):
    """Exclusive Product on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_prod_forward(chunk_starts, chunk_cnts, inputs)
        if ctx.needs_input_grad[2]:
            ctx.save_for_backward(chunk_starts, chunk_cnts, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.exclusive_prod_backward(chunk_starts, chunk_cnts, inputs, outputs, grad_outputs)
        return None, None, grad_inputs


class _ExclusiveProdCUB(torch.autograd.Function):
    """Exclusive Product on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_prod_cub_forward(indices, inputs)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        indices, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.exclusive_prod_cub_backward(indices, inputs, outputs, grad_outputs)
        return None, grad_inputs


@torch.no_grad()
def pack_info(ray_indices: 'Tensor', n_rays: 'Optional[int]'=None) ->Tensor:
    """Pack `ray_indices` to `packed_info`. Useful for converting per sample data to per ray data.

    Note:
        this function is not differentiable to any inputs.

    Args:
        ray_indices: Ray indices of the samples. LongTensor with shape (n_sample).
        n_rays: Number of rays. If None, it is inferred from `ray_indices`. Default is None.

    Returns:
        A LongTensor of shape (n_rays, 2) that specifies the start and count
        of each chunk in the flattened input tensor, with in total n_rays chunks.

    Example:

    .. code-block:: python

        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2], device="cuda")
        >>> packed_info = pack_info(ray_indices, n_rays=3)
        >>> packed_info
        tensor([[0, 2], [2, 3], [5, 4]], device='cuda:0')

    """
    assert ray_indices.dim() == 1, 'ray_indices must be a 1D tensor with shape (n_samples).'
    if ray_indices.is_cuda:
        device = ray_indices.device
        dtype = ray_indices.dtype
        if n_rays is None:
            n_rays = ray_indices.max().item() + 1
        chunk_cnts = torch.zeros((n_rays,), device=device, dtype=dtype)
        chunk_cnts.index_add_(0, ray_indices, torch.ones_like(ray_indices))
        chunk_starts = chunk_cnts.cumsum(dim=0, dtype=dtype) - chunk_cnts
        packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    else:
        raise NotImplementedError('Only support cuda inputs.')
    return packed_info


def exclusive_prod(inputs: 'Tensor', packed_info: 'Optional[Tensor]'=None, indices: 'Optional[Tensor]'=None) ->Tensor:
    """Exclusive Product that supports flattened tensor.

    Similar to :func:`nerfacc.inclusive_prod`, but computes the exclusive product.

    Args:
        inputs: The tensor to be producted. Can be either a N-D tensor, or a flattened
            tensor with either `packed_info` or `indices` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the product is computed
            along the last dimension. Default is None.
        indices: A flattened tensor with the same shape as `inputs`.

    Returns:
        The exclusive product with the same shape as the input tensor.


    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> exclusive_prod(inputs, packed_info)
        tensor([1., 1., 1., 3., 12., 1., 6., 42., 336.], device='cuda:0')

    """
    if indices is not None and packed_info is not None:
        raise ValueError('Only one of `indices` and `packed_info` can be specified.')
    if indices is not None:
        assert indices.dim() == 1 and indices.shape == inputs.shape, 'indices must be 1-D with the same shape as inputs.'
        if _C.is_cub_available():
            outputs = _ExclusiveProdCUB.apply(indices, inputs)
        else:
            warnings.warn('Passing in `indices` without CUB available is slow. Considering passing in `packed_info` instead.')
            packed_info = pack_info(ray_indices=indices)
    if packed_info is not None:
        assert inputs.dim() == 1, 'inputs must be flattened.'
        assert packed_info.dim() == 2 and packed_info.shape[-1] == 2, 'packed_info must be 2-D with shape (B, 2).'
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
        outputs = _ExclusiveProd.apply(chunk_starts, chunk_cnts, inputs)
    if indices is None and packed_info is None:
        outputs = torch.cumprod(torch.cat([torch.ones_like(inputs[..., :1]), inputs[..., :-1]], dim=-1), dim=-1)
    return outputs


def _make_lazy_cuda_func(name: 'str') ->Callable:

    def call_cuda(*args, **kwargs):
        return getattr(_C, name)(*args, **kwargs)
    return call_cuda


is_cub_available = _make_lazy_cuda_func('is_cub_available')


def render_transmittance_from_alpha(alphas: 'Tensor', packed_info: 'Optional[Tensor]'=None, ray_indices: 'Optional[Tensor]'=None, n_rays: 'Optional[int]'=None, prefix_trans: 'Optional[Tensor]'=None) ->Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`.

    .. math::
        T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance with the same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
    """
    if not is_cub_available() and packed_info is None:
        packed_info = pack_info(ray_indices, n_rays)
        ray_indices = None
    trans = exclusive_prod(1 - alphas, packed_info=packed_info, indices=ray_indices)
    if prefix_trans is not None:
        trans = trans * prefix_trans
    return trans


@torch.no_grad()
def render_visibility_from_alpha(alphas: 'Tensor', packed_info: 'Optional[Tensor]'=None, ray_indices: 'Optional[Tensor]'=None, n_rays: 'Optional[int]'=None, early_stop_eps: 'float'=0.0001, alpha_thre: 'float'=0.0, prefix_trans: 'Optional[Tensor]'=None) ->Tensor:
    """Compute visibility from opacity :math:`\\alpha_i`.

    In this function, we first compute the transmittance from the sample opacity. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
        >>> visibility = render_visibility_from_alpha(
        >>>     alphas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans = render_transmittance_from_alpha(alphas, packed_info, ray_indices, n_rays, prefix_trans)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


class _ExclusiveSum(torch.autograd.Function):
    """Exclusive Sum on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs, normalize: 'bool'=False):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_sum(chunk_starts, chunk_cnts, inputs, normalize, False)
        if ctx.needs_input_grad[2]:
            ctx.normalize = normalize
            ctx.save_for_backward(chunk_starts, chunk_cnts)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts = ctx.saved_tensors
        normalize = ctx.normalize
        assert normalize == False, 'Only support backward for normalize==False.'
        grad_inputs = _C.exclusive_sum(chunk_starts, chunk_cnts, grad_outputs, normalize, True)
        return None, None, grad_inputs, None


class _ExclusiveSumCUB(torch.autograd.Function):
    """Exclusive Sum on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_sum_cub(indices, inputs, False)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        indices, = ctx.saved_tensors
        grad_inputs = _C.exclusive_sum_cub(indices, grad_outputs, True)
        return None, grad_inputs


def exclusive_sum(inputs: 'Tensor', packed_info: 'Optional[Tensor]'=None, indices: 'Optional[Tensor]'=None) ->Tensor:
    """Exclusive Sum that supports flattened tensor.

    Similar to :func:`nerfacc.inclusive_sum`, but computes the exclusive sum.

    Args:
        inputs: The tensor to be summed. Can be either a N-D tensor, or a flattened
            tensor with either `packed_info` or `indices` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the sum is computed
            along the last dimension. Default is None.
        indices: A flattened tensor with the same shape as `inputs`.

    Returns:
        The exclusive sum with the same shape as the input tensor.

    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> exclusive_sum(inputs, packed_info)
        tensor([ 0.,  1.,  0.,  3.,  7.,  0.,  6., 13., 21.], device='cuda:0')

    """
    if indices is not None and packed_info is not None:
        raise ValueError('Only one of `indices` and `packed_info` can be specified.')
    if indices is not None:
        assert indices.dim() == 1 and indices.shape == inputs.shape, 'indices must be 1-D with the same shape as inputs.'
        if _C.is_cub_available():
            outputs = _ExclusiveSumCUB.apply(indices, inputs)
        else:
            warnings.warn('Passing in `indices` without CUB available is slow. Considering passing in `packed_info` instead.')
            packed_info = pack_info(ray_indices=indices)
    if packed_info is not None:
        assert inputs.dim() == 1, 'inputs must be flattened.'
        assert packed_info.dim() == 2 and packed_info.shape[-1] == 2, 'packed_info must be 2-D with shape (B, 2).'
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
        outputs = _ExclusiveSum.apply(chunk_starts, chunk_cnts, inputs, False)
    if indices is None and packed_info is None:
        outputs = torch.cumsum(torch.cat([torch.zeros_like(inputs[..., :1]), inputs[..., :-1]], dim=-1), dim=-1)
    return outputs


def render_transmittance_from_density(t_starts: 'Tensor', t_ends: 'Tensor', sigmas: 'Tensor', packed_info: 'Optional[Tensor]'=None, ray_indices: 'Optional[Tensor]'=None, n_rays: 'Optional[int]'=None, prefix_trans: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`.

    .. math::
        T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\\delta_j)
    
    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with             shape (all_samples,) or (n_rays, n_samples).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with             shape (all_samples,) or (n_rays, n_samples).
        sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance and opacities, both with the same shape as `sigmas`.

    Examples:
    
    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

    """
    if not is_cub_available() and packed_info is None:
        packed_info = pack_info(ray_indices, n_rays)
        ray_indices = None
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt, packed_info=packed_info, indices=ray_indices))
    if prefix_trans is not None:
        trans = trans * prefix_trans
    return trans, alphas


@torch.no_grad()
def render_visibility_from_density(t_starts: 'Tensor', t_ends: 'Tensor', sigmas: 'Tensor', packed_info: 'Optional[Tensor]'=None, ray_indices: 'Optional[Tensor]'=None, n_rays: 'Optional[int]'=None, early_stop_eps: 'float'=0.0001, alpha_thre: 'float'=0.0, prefix_trans: 'Optional[Tensor]'=None) ->Tensor:
    """Compute visibility from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

    In this function, we first compute the transmittance and opacity from the sample density. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]
        >>> visibility = render_visibility_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans, alphas = render_transmittance_from_density(t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


class N3TreeEstimator(AbstractEstimator):
    """Use N3Tree to implement Occupancy Grid.

    This allows more flexible topologies than the cascaded grid. However, it is
    slower to create samples from the tree than the cascaded grid. By default,
    it has the same topology as the cascaded grid but `self.tree` can be
    modified to have different topologies.
    """

    def __init__(self, roi_aabb: 'Union[List[int], Tensor]', resolution: 'Union[int, List[int], Tensor]'=128, levels: 'int'=1, **kwargs) ->None:
        super().__init__()
        if 'contraction_type' in kwargs:
            raise ValueError('`contraction_type` is not supported anymore for nerfacc >= 0.4.0.')
        assert isinstance(resolution, int), 'N3Tree only supports uniform resolution!'
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, Tensor), f'Invalid type: {roi_aabb}!'
        assert roi_aabb.shape[0] == 6, f'Invalid shape: {roi_aabb}!'
        roi_aabb = roi_aabb.cpu()
        aabbs = torch.stack([_enlarge_aabb(roi_aabb, 2 ** i) for i in range(levels)], dim=0)
        self.register_buffer('aabbs', aabbs)
        center = (roi_aabb[:3] + roi_aabb[3:]) / 2.0
        radius = (roi_aabb[3:] - roi_aabb[:3]) / 2.0 * 2 ** (levels - 1)
        tree_depth = int(math.log2(resolution)) - 1
        self.tree = svox.N3Tree(N=2, data_dim=1, init_refine=tree_depth, depth_limit=20, radius=radius.tolist(), center=center.tolist())
        _aabbs = [_enlarge_aabb(roi_aabb, 2 ** i) for i in range(levels - 1)]
        for aabb in _aabbs[::-1]:
            leaf_c = self.tree.corners + self.tree.lengths * 0.5
            sel = ((leaf_c > aabb[:3]) & (leaf_c < aabb[3:])).all(dim=-1)
            self.tree[sel].refine()
        self.thresh = 0.0

    @torch.no_grad()
    def sampling(self, rays_o: 'Tensor', rays_d: 'Tensor', sigma_fn: 'Optional[Callable]'=None, alpha_fn: 'Optional[Callable]'=None, near_plane: 'float'=0.0, far_plane: 'float'=10000000000.0, t_min: 'Optional[Tensor]'=None, t_max: 'Optional[Tensor]'=None, render_step_size: 'float'=0.001, early_stop_eps: 'float'=0.0001, alpha_thre: 'float'=0.0, stratified: 'bool'=False, cone_angle: 'float'=0.0) ->Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If profided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If profided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """
        assert t_min is None and t_max is None, 'Do not supported per-ray min max. Please use near_plane and far_plane instead.'
        if stratified:
            near_plane += torch.rand(()).item() * render_step_size
        t_starts, t_ends, packed_info, ray_indices = svox.volume_sample(self.tree, thresh=self.thresh, rays=svox.Rays(rays_o.contiguous(), rays_d.contiguous(), rays_d.contiguous()), step_size=render_step_size, cone_angle=cone_angle, near_plane=near_plane, far_plane=far_plane)
        packed_info = packed_info.long()
        ray_indices = ray_indices.long()
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (sigma_fn is not None or alpha_fn is not None):
            alpha_thre = min(alpha_thre, self.thresh)
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert sigmas.shape == t_starts.shape, 'sigmas must have shape of (N,)! Got {}'.format(sigmas.shape)
                masks = render_visibility_from_density(t_starts=t_starts, t_ends=t_ends, sigmas=sigmas, ray_indices=ray_indices, n_rays=len(rays_o), early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert alphas.shape == t_starts.shape, 'alphas must have shape of (N,)! Got {}'.format(alphas.shape)
                masks = render_visibility_from_alpha(alphas=alphas, ray_indices=ray_indices, n_rays=len(rays_o), early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            ray_indices, t_starts, t_ends = ray_indices[masks], t_starts[masks], t_ends[masks]
        return ray_indices, t_starts, t_ends

    @torch.no_grad()
    def update_every_n_steps(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256, n: 'int'=16) ->None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError('You should only call this function only during training. Please call _update() directly if you want to update the field during inference.')
        if step % n == 0 and self.training:
            self._update(step=step, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre, ema_decay=ema_decay, warmup_steps=warmup_steps)

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: 'int') ->List[Tensor]:
        """Samples both n uniform and occupied cells."""
        uniform_indices = torch.randint(len(self.tree), (n,), device=self.device)
        occupied_indices = torch.nonzero(self.tree[:].values >= self.thresh)[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(len(occupied_indices), (n,), device=self.device)
            occupied_indices = occupied_indices[selector]
        indices = torch.cat([uniform_indices, occupied_indices], dim=0)
        return indices

    @torch.no_grad()
    def _update(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256) ->None:
        """Update the occ field in the EMA way."""
        if step < warmup_steps:
            x = self.tree.sample(1).squeeze(1)
            occ = occ_eval_fn(x).squeeze(-1)
            sel = *self.tree._all_leaves().T,
            self.tree.data.data[sel] = torch.maximum(self.tree.data.data[sel] * ema_decay, occ[:, None])
        else:
            N = len(self.tree) // 4
            indices = self._sample_uniform_and_occupied_cells(N)
            x = self.tree[indices].sample(1).squeeze(1)
            occ = occ_eval_fn(x).squeeze(-1)
            self.tree[indices] = torch.maximum(self.tree[indices].values * ema_decay, occ[:, None])
        self.thresh = min(occ_thre, self.tree[:].values.mean().item())


def _meshgrid3d(res: 'Tensor', device: 'Union[torch.device, str]'='cpu') ->Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(torch.meshgrid([torch.arange(res[0], dtype=torch.long), torch.arange(res[1], dtype=torch.long), torch.arange(res[2], dtype=torch.long)], indexing='ij'), dim=-1)


@torch.no_grad()
def ray_aabb_intersect(rays_o: 'Tensor', rays_d: 'Tensor', aabbs: 'Tensor', near_plane: 'float'=-float('inf'), far_plane: 'float'=float('inf'), miss_value: 'float'=float('inf')) ->Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_plane: Optional. Near plane. Default to -infinity.
        far_plane: Optional. Far plane. Default to infinity.
        miss_value: Optional. Value to use for tmin and tmax when there is no intersection.
            Default to infinity.

    Returns:
        A tuple of {Tensor, Tensor, BoolTensor}:

        - **t_mins**: (n_rays, m) tmin for each ray-AABB pair.
        - **t_maxs**: (n_rays, m) tmax for each ray-AABB pair.
        - **hits**: (n_rays, m) whether each ray-AABB pair intersects.
    """
    assert rays_o.ndim == 2 and rays_o.shape[-1] == 3
    assert rays_d.ndim == 2 and rays_d.shape[-1] == 3
    assert aabbs.ndim == 2 and aabbs.shape[-1] == 6
    t_mins, t_maxs, hits = _C.ray_aabb_intersect(rays_o.contiguous(), rays_d.contiguous(), aabbs.contiguous(), near_plane, far_plane, miss_value)
    return t_mins, t_maxs, hits


class OccGridEstimator(AbstractEstimator):
    """Occupancy grid transmittance estimator for spatial skipping.

    References: "Instant Neural Graphics Primitives."

    Args:
        roi_aabb: The axis-aligned bounding box of the region of interest. Useful for mapping
            the 3D space to the grid.
        resolution: The resolution of the grid. If an integer is given, the grid is assumed to
            be a cube. Otherwise, a list or a tensor of shape (3,) is expected. Default: 128.
        levels: The number of levels of the grid. Default: 1.
    """
    DIM: 'int' = 3

    def __init__(self, roi_aabb: 'Union[List[int], Tensor]', resolution: 'Union[int, List[int], Tensor]'=128, levels: 'int'=1, **kwargs) ->None:
        super().__init__()
        if 'contraction_type' in kwargs:
            raise ValueError('`contraction_type` is not supported anymore for nerfacc >= 0.4.0.')
        if isinstance(resolution, int):
            resolution = [resolution] * self.DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, Tensor), f'Invalid type: {resolution}!'
        assert resolution.shape[0] == self.DIM, f'Invalid shape: {resolution}!'
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, Tensor), f'Invalid type: {roi_aabb}!'
        assert roi_aabb.shape[0] == self.DIM * 2, f'Invalid shape: {roi_aabb}!'
        aabbs = torch.stack([_enlarge_aabb(roi_aabb, 2 ** i) for i in range(levels)], dim=0)
        self.cells_per_lvl = int(resolution.prod().item())
        self.levels = levels
        self.register_buffer('resolution', resolution)
        self.register_buffer('aabbs', aabbs)
        self.register_buffer('occs', torch.zeros(self.levels * self.cells_per_lvl))
        self.register_buffer('binaries', torch.zeros([levels] + resolution.tolist(), dtype=torch.bool))
        grid_coords = _meshgrid3d(resolution).reshape(self.cells_per_lvl, self.DIM)
        self.register_buffer('grid_coords', grid_coords, persistent=False)
        grid_indices = torch.arange(self.cells_per_lvl)
        self.register_buffer('grid_indices', grid_indices, persistent=False)

    @torch.no_grad()
    def sampling(self, rays_o: 'Tensor', rays_d: 'Tensor', sigma_fn: 'Optional[Callable]'=None, alpha_fn: 'Optional[Callable]'=None, near_plane: 'float'=0.0, far_plane: 'float'=10000000000.0, t_min: 'Optional[Tensor]'=None, t_max: 'Optional[Tensor]'=None, render_step_size: 'float'=0.001, early_stop_eps: 'float'=0.0001, alpha_thre: 'float'=0.0, stratified: 'bool'=False, cone_angle: 'float'=0.0) ->Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If provided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If provided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """
        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)
        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)
        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        intervals, samples, _ = traverse_grids(rays_o, rays_d, self.binaries, self.aabbs, near_planes=near_planes, far_planes=far_planes, step_size=render_step_size, cone_angle=cone_angle)
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices
        packed_info = samples.packed_info
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (sigma_fn is not None or alpha_fn is not None):
            alpha_thre = min(alpha_thre, self.occs.mean().item())
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert sigmas.shape == t_starts.shape, 'sigmas must have shape of (N,)! Got {}'.format(sigmas.shape)
                masks = render_visibility_from_density(t_starts=t_starts, t_ends=t_ends, sigmas=sigmas, packed_info=packed_info, early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert alphas.shape == t_starts.shape, 'alphas must have shape of (N,)! Got {}'.format(alphas.shape)
                masks = render_visibility_from_alpha(alphas=alphas, packed_info=packed_info, early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            ray_indices, t_starts, t_ends = ray_indices[masks], t_starts[masks], t_ends[masks]
        return ray_indices, t_starts, t_ends

    @torch.no_grad()
    def update_every_n_steps(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256, n: 'int'=16) ->None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError('You should only call this function only during training. Please call _update() directly if you want to update the field during inference.')
        if step % n == 0 and self.training:
            self._update(step=step, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre, ema_decay=ema_decay, warmup_steps=warmup_steps)

    @torch.no_grad()
    def mark_invisible_cells(self, K: 'Tensor', c2w: 'Tensor', width: 'int', height: 'int', near_plane: 'float'=0.0, chunk: 'int'=32 ** 3) ->None:
        """Mark the cells that aren't covered by the cameras with density -1.
        Should only be executed once before training starts.

        Args:
            K: Camera intrinsics of shape (N, 3, 3) or (1, 3, 3).
            c2w: Camera to world poses of shape (N, 3, 4) or (N, 4, 4).
            width: Image width in pixels
            height: Image height in pixels
            near_plane: Near plane distance
            chunk: The chunk size to split the cells (to avoid OOM)
        """
        assert K.dim() == 3 and K.shape[1:] == (3, 3)
        assert c2w.dim() == 3 and (c2w.shape[1:] == (3, 4) or c2w.shape[1:] == (4, 4))
        assert K.shape[0] == c2w.shape[0] or K.shape[0] == 1
        N_cams = c2w.shape[0]
        w2c_R = c2w[:, :3, :3].transpose(2, 1)
        w2c_T = -w2c_R @ c2w[:, :3, 3:]
        lvl_indices = self._get_all_cells()
        for lvl, indices in enumerate(lvl_indices):
            grid_coords = self.grid_coords[indices]
            for i in range(0, len(indices), chunk):
                x = grid_coords[i:i + chunk] / (self.resolution - 1)
                indices_chunk = indices[i:i + chunk]
                xyzs_w = (self.aabbs[lvl, :3] + x * (self.aabbs[lvl, 3:] - self.aabbs[lvl, :3])).T
                xyzs_c = w2c_R @ xyzs_w + w2c_T
                uvd = K @ xyzs_c
                uv = uvd[:, :2] / uvd[:, 2:]
                in_image = (uvd[:, 2] >= 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
                covered_by_cam = (uvd[:, 2] >= near_plane) & in_image
                count = covered_by_cam.sum(0) / N_cams
                too_near_to_cam = (uvd[:, 2] < near_plane) & in_image
                too_near_to_any_cam = too_near_to_cam.any(0)
                valid_mask = (count > 0) & ~too_near_to_any_cam
                cell_ids_base = lvl * self.cells_per_lvl
                self.occs[cell_ids_base + indices_chunk] = torch.where(valid_mask, 0.0, -1.0)

    @torch.no_grad()
    def _get_all_cells(self) ->List[Tensor]:
        """Returns all cells of the grid."""
        lvl_indices = []
        for lvl in range(self.levels):
            cell_ids = lvl * self.cells_per_lvl + self.grid_indices
            indices = self.grid_indices[self.occs[cell_ids] >= 0.0]
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: 'int') ->List[Tensor]:
        """Samples both n uniform and occupied cells."""
        lvl_indices = []
        for lvl in range(self.levels):
            uniform_indices = torch.randint(self.cells_per_lvl, (n,), device=self.device)
            cell_ids = lvl * self.cells_per_lvl + uniform_indices
            uniform_indices = uniform_indices[self.occs[cell_ids] >= 0.0]
            occupied_indices = torch.nonzero(self.binaries[lvl].flatten())[:, 0]
            if n < len(occupied_indices):
                selector = torch.randint(len(occupied_indices), (n,), device=self.device)
                occupied_indices = occupied_indices[selector]
            indices = torch.cat([uniform_indices, occupied_indices], dim=0)
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _update(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256) ->None:
        """Update the occ field in the EMA way."""
        if step < warmup_steps:
            lvl_indices = self._get_all_cells()
        else:
            N = self.cells_per_lvl // 4
            lvl_indices = self._sample_uniform_and_occupied_cells(N)
        for lvl, indices in enumerate(lvl_indices):
            grid_coords = self.grid_coords[indices]
            x = (grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)) / self.resolution
            x = self.aabbs[lvl, :3] + x * (self.aabbs[lvl, 3:] - self.aabbs[lvl, :3])
            occ = occ_eval_fn(x).squeeze(-1)
            cell_ids = lvl * self.cells_per_lvl + indices
            self.occs[cell_ids] = torch.maximum(self.occs[cell_ids] * ema_decay, occ)
        thre = torch.clamp(self.occs[self.occs >= 0].mean(), max=occ_thre)
        self.binaries = (self.occs > thre).view(self.binaries.shape)


def searchsorted(sorted_sequence: 'Union[RayIntervals, RaySamples]', values: 'Union[RayIntervals, RaySamples]') ->Tuple[Tensor, Tensor]:
    """Searchsorted that supports flattened tensor.

    This function returns {`ids_left`, `ids_right`} such that:

    `sorted_sequence.vals.gather(-1, ids_left) <= values.vals < sorted_sequence.vals.gather(-1, ids_right)`

    Note:
        When values is out of range of sorted_sequence, we return the
        corresponding ids as if the values is clipped to the range of
        sorted_sequence. See the example below.

    Args:
        sorted_sequence: A :class:`RayIntervals` or :class:`RaySamples` object. We assume
            the `sorted_sequence.vals` is acendingly sorted for each ray.
        values: A :class:`RayIntervals` or :class:`RaySamples` object.

    Returns:
        A tuple of LongTensor:

        - **ids_left**: A LongTensor with the same shape as `values.vals`.
        - **ids_right**: A LongTensor with the same shape as `values.vals`.

    Example:
        >>> sorted_sequence = RayIntervals(
        ...     vals=torch.tensor([0.0, 1.0, 0.0, 1.0, 2.0], device="cuda"),
        ...     packed_info=torch.tensor([[0, 2], [2, 3]], device="cuda"),
        ... )
        >>> values = RayIntervals(
        ...     vals=torch.tensor([0.5, 1.5, 2.5], device="cuda"),
        ...     packed_info=torch.tensor([[0, 1], [1, 2]], device="cuda"),
        ... )
        >>> ids_left, ids_right = searchsorted(sorted_sequence, values)
        >>> ids_left
        tensor([0, 3, 3], device='cuda:0')
        >>> ids_right
        tensor([1, 4, 4], device='cuda:0')
        >>> sorted_sequence.vals.gather(-1, ids_left)
        tensor([0., 1., 1.], device='cuda:0')
        >>> sorted_sequence.vals.gather(-1, ids_right)
        tensor([1., 2., 2.], device='cuda:0')
    """
    ids_left, ids_right = _C.searchsorted(values._to_cpp(), sorted_sequence._to_cpp())
    return ids_left, ids_right


def _pdf_loss(segments_query: 'RayIntervals', cdfs_query: 'torch.Tensor', segments_key: 'RayIntervals', cdfs_key: 'torch.Tensor', eps: 'float'=1e-07) ->torch.Tensor:
    ids_left, ids_right = searchsorted(segments_key, segments_query)
    if segments_query.vals.dim() > 1:
        w = cdfs_query[..., 1:] - cdfs_query[..., :-1]
        ids_left = ids_left[..., :-1]
        ids_right = ids_right[..., 1:]
    else:
        assert segments_query.is_left is not None
        assert segments_query.is_right is not None
        w = cdfs_query[segments_query.is_right] - cdfs_query[segments_query.is_left]
        ids_left = ids_left[segments_query.is_left]
        ids_right = ids_right[segments_query.is_right]
    w_outer = cdfs_key.gather(-1, ids_right) - cdfs_key.gather(-1, ids_left)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


def _transform_stot(transform_type: "Literal['uniform', 'lindisp']", s_vals: 'torch.Tensor', t_min: 'torch.Tensor', t_max: 'torch.Tensor') ->torch.Tensor:
    if transform_type == 'uniform':
        _contract_fn, _icontract_fn = lambda x: x, lambda x: x
    elif transform_type == 'lindisp':
        _contract_fn, _icontract_fn = lambda x: 1 / x, lambda x: 1 / x
    else:
        raise ValueError(f'Unknown transform_type: {transform_type}')
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)


class PropNetEstimator(AbstractEstimator):
    """Proposal network transmittance estimator.

    References: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields."

    Args:
        optimizer: The optimizer to use for the proposal networks.
        scheduler: The learning rate scheduler to use for the proposal networks.
    """

    def __init__(self, optimizer: 'Optional[torch.optim.Optimizer]'=None, scheduler: 'Optional[torch.optim.lr_scheduler._LRScheduler]'=None) ->None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prop_cache: 'List' = []

    @torch.no_grad()
    def sampling(self, prop_sigma_fns: 'List[Callable]', prop_samples: 'List[int]', num_samples: 'int', n_rays: 'int', near_plane: 'float', far_plane: 'float', sampling_type: "Literal['uniform', 'lindisp']"='lindisp', stratified: 'bool'=False, requires_grad: 'bool'=False) ->Tuple[Tensor, Tensor]:
        """Sampling with CDFs from proposal networks.

        Note:
            When `requires_grad` is `True`, the gradients are allowed to flow
            through the proposal networks, and the outputs of the proposal
            networks are cached to update them later when calling `update_every_n_steps()`

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), 'The number of proposal networks and the number of samples should be the same.'
        cdfs = torch.cat([torch.zeros((n_rays, 1), device=self.device), torch.ones((n_rays, 1), device=self.device)], dim=-1)
        intervals = RayIntervals(vals=cdfs)
        for level_fn, level_samples in zip(prop_sigma_fns, prop_samples):
            intervals, _ = importance_sampling(intervals, cdfs, level_samples, stratified)
            t_vals = _transform_stot(sampling_type, intervals.vals, near_plane, far_plane)
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]
            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas)
                cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)
                if requires_grad:
                    self.prop_cache.append((intervals, cdfs))
        intervals, _ = importance_sampling(intervals, cdfs, num_samples, stratified)
        t_vals = _transform_stot(sampling_type, intervals.vals, near_plane, far_plane)
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        if requires_grad:
            self.prop_cache.append((intervals, None))
        return t_starts, t_ends

    @torch.enable_grad()
    def compute_loss(self, trans: 'Tensor', loss_scaler: 'float'=1.0) ->Tensor:
        """Compute the loss for the proposal networks.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            loss_scaler: The loss scaler. Default to 1.0.

        Returns:
            The loss for the proposal networks.
        """
        if len(self.prop_cache) == 0:
            return torch.zeros((), device=self.device)
        intervals, _ = self.prop_cache.pop()
        cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)
        cdfs = cdfs.detach()
        loss = 0.0
        while self.prop_cache:
            prop_intervals, prop_cdfs = self.prop_cache.pop()
            loss += _pdf_loss(intervals, cdfs, prop_intervals, prop_cdfs).mean()
        return loss * loss_scaler

    @torch.enable_grad()
    def update_every_n_steps(self, trans: 'Tensor', requires_grad: 'bool'=False, loss_scaler: 'float'=1.0) ->float:
        """Update the estimator every n steps during training.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.
            loss_scaler: The loss scaler to use. Default to 1.0.

        Returns:
            The loss of the proposal networks for logging (a float scalar).
        """
        if requires_grad:
            return self._update(trans=trans, loss_scaler=loss_scaler)
        else:
            if self.scheduler is not None:
                self.scheduler.step()
            return 0.0

    @torch.enable_grad()
    def _update(self, trans: 'Tensor', loss_scaler: 'float'=1.0) ->float:
        assert len(self.prop_cache) > 0
        assert self.optimizer is not None, 'No optimizer is provided.'
        loss = self.compute_loss(trans, loss_scaler)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()


fVDB_ENABLED = True


@torch.no_grad()
def traverse_vdbs(rays_o: 'Tensor', rays_d: 'Tensor', grids: 'GridBatch', near_planes: 'Optional[Tensor]'=None, far_planes: 'Optional[Tensor]'=None, step_size: 'Optional[float]'=0.001, cone_angle: 'Optional[float]'=0.0):
    """Traverse the fVDB grids."""
    assert fVDB_ENABLED, 'Please install fVDB to use this function.'
    assert len(grids) == 1, 'Only support one grid for now.'
    if near_planes is None:
        near_planes = torch.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = torch.full_like(rays_o[:, 0], float('inf'))
    _, indices, intervals = grids.uniform_ray_samples(rays_o, rays_d, near_planes, far_planes, step_size, cone_angle, include_end_segments=False)
    t_starts, t_ends = torch.unbind(intervals.jdata, dim=-1)
    ray_indices = indices.jdata.long()
    return t_starts, t_ends, ray_indices


class VDBEstimator(AbstractEstimator):
    """Occupancy Estimator Using A VDB."""

    def __init__(self, init_grid: 'GridBatch', device='cuda:0') ->None:
        super().__init__()
        assert fVDB_ENABLED, 'Please install fVDB to use this class.'
        assert len(init_grid) == 1, 'Only support one grid for now.'
        self.grid = sparse_grid_from_ijk(init_grid.ijk, voxel_sizes=init_grid.voxel_sizes, origins=init_grid.origins, mutable=True)
        self.occs = torch.nn.Parameter(torch.zeros([self.grid.total_voxels], device=device), requires_grad=False)

    def state_dict(self):
        state_dict = self.state_dict()
        state_dict['grid'] = self.grid
        return state_dict

    def load_state_dict(self, state_dict: 'Mapping[str, Any]', strict: 'bool'=True):
        init_grid = state_dict['grid']
        self.grid = sparse_grid_from_ijk(init_grid.ijk, voxel_sizes=init_grid.voxel_sizes, origins=init_grid.origins, mutable=True)
        remaining_state_dict = {k: v for k, v in state_dict.items() if k not in ['grid']}
        super().load_state_dict(remaining_state_dict, strict=strict)

    def to(self, device: 'Union[str, torch.device]'):
        self.grid = self.grid
        self.occs = self.occs
        super()
        return self

    @torch.no_grad()
    def sampling(self, rays_o: 'Tensor', rays_d: 'Tensor', sigma_fn: 'Optional[Callable]'=None, alpha_fn: 'Optional[Callable]'=None, near_plane: 'float'=0.0, far_plane: 'float'=10000000000.0, t_min: 'Optional[Tensor]'=None, t_max: 'Optional[Tensor]'=None, render_step_size: 'float'=0.001, early_stop_eps: 'float'=0.0001, alpha_thre: 'float'=0.0, stratified: 'bool'=False, cone_angle: 'float'=0.0) ->Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If provided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If provided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """
        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)
        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)
        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        t_starts, t_ends, ray_indices = traverse_vdbs(rays_o, rays_d, self.grid, near_planes=near_planes, far_planes=far_planes, step_size=render_step_size, cone_angle=cone_angle)
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (sigma_fn is not None or alpha_fn is not None):
            alpha_thre = min(alpha_thre, self.occs.mean().item())
            n_rays = rays_o.shape[0]
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert sigmas.shape == t_starts.shape, 'sigmas must have shape of (N,)! Got {}'.format(sigmas.shape)
                masks = render_visibility_from_density(t_starts=t_starts, t_ends=t_ends, sigmas=sigmas, ray_indices=ray_indices, n_rays=n_rays, early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert alphas.shape == t_starts.shape, 'alphas must have shape of (N,)! Got {}'.format(alphas.shape)
                masks = render_visibility_from_alpha(alphas=alphas, ray_indices=ray_indices, n_rays=n_rays, early_stop_eps=early_stop_eps, alpha_thre=alpha_thre)
            ray_indices, t_starts, t_ends = ray_indices[masks], t_starts[masks], t_ends[masks]
        return ray_indices, t_starts, t_ends

    @torch.no_grad()
    def update_every_n_steps(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256, n: 'int'=16) ->None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError('You should only call this function only during training. Please call _update() directly if you want to update the field during inference.')
        if step % n == 0 and self.training:
            self._update(step=step, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre, ema_decay=ema_decay, warmup_steps=warmup_steps)

    @torch.no_grad()
    def _get_all_cells(self) ->List[Tensor]:
        """Returns all cells of the grid."""
        return self.grid.ijk.jdata

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self) ->List[Tensor]:
        """Samples both n uniform and occupied cells."""
        n = self.grid.total_voxels // 4
        uniform_selector = torch.randint(0, self.grid.total_voxels, (n,), device=self.device)
        uniform_ijks = self.grid.ijk.jdata[uniform_selector]
        occupied_ijks = self.grid.ijk_enabled.jdata
        if n < len(occupied_ijks):
            occupied_selector = torch.randint(0, len(occupied_ijks), (n,), device=self.device)
            occupied_ijks = occupied_ijks[occupied_selector]
        ijks = torch.cat([uniform_ijks, occupied_ijks], dim=0)
        return ijks

    @torch.no_grad()
    def _update(self, step: 'int', occ_eval_fn: 'Callable', occ_thre: 'float'=0.01, ema_decay: 'float'=0.95, warmup_steps: 'int'=256) ->None:
        """Update the occ field in the EMA way."""
        if step < warmup_steps:
            ijks = self._get_all_cells()
        else:
            ijks = self._sample_uniform_and_occupied_cells()
        grid_coords = ijks - 0.5 + torch.rand_like(ijks, dtype=torch.float32)
        x = self.grid.grid_to_world(grid_coords).jdata
        occ = occ_eval_fn(x).squeeze(-1)
        index = self.grid.ijk_to_index(ijks).jdata
        self.occs[index] = torch.maximum(self.occs[index] * ema_decay, occ)
        thre = torch.clamp(self.occs.mean(), max=occ_thre)
        active = self.occs[index] >= thre
        _ijks = ijks[active]
        if len(_ijks) > 0:
            self.grid.enable_ijk(_ijks)
        _ijks = ijks[~active]
        if len(_ijks) > 0:
            self.grid.disable_ijk(_ijks)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DenseLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinusoidalEncoder,
     lambda: ([], {'x_dim': 4, 'min_deg': 4, 'max_deg': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

