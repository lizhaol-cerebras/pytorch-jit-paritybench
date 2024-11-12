
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


from typing import *


import torch


import warnings


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import torch.distributed as dist


from torch.utils.data.sampler import SequentialSampler


from torch.utils.cpp_extension import load


import time


import torch.nn as nn


import torch.nn.functional as F


import functools


import scipy.signal


import torch.nn.init as init


import itertools


import numpy as onp


from random import random


import torch.autograd as autograd


from typing import Tuple


from typing import List


from typing import Optional


from typing import Union


from functools import reduce


from scipy.spatial.transform import Rotation


import math


from functools import partial


from typing import Any


from typing import Callable


render_utils_cuda = None


class Alphas2Weights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(alpha, weights, T, alphainv_last, i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


class Raw2Alpha(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1, 1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1, -1).expand(shape).flatten()
    return ray_id, step_id


ub360_utils_cuda = None


class DirectContractedVoxGO(nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None, mask_cache_world_size=None, fast_color_thres=0, bg_len=0.2, contracted_norm='inf', density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, **kwargs):
        super(DirectContractedVoxGO, self).__init__()
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectContractedVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1, -1, -1]) - bg_len)
        self.register_buffer('xyz_max', torch.Tensor([1, 1, 1]) + bg_len)
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.bg_len = bg_len
        self.contracted_norm = contracted_norm
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self._set_grid_resolution(num_voxels)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        None
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            None
            None
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        None
        None
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'num_voxels_base': self.num_voxels_base, 'alpha_init': self.alpha_init, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'contracted_norm': self.contracted_norm, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres
        new_p = self.mask_cache.mask.float().mean().item()
        None

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        None
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += ones.grid.grad > 1
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        None
        eps_time = time.time() - eps_time
        None

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        N_inner = int(2 / (2 + 2 * self.bg_len) * self.world_len / stepsize) + 1
        N_outer = N_inner
        b_inner = torch.linspace(0, 2, N_inner + 1)
        b_outer = 2 / torch.linspace(1, 1 / 128, N_outer + 1)
        t = torch.cat([(b_inner[1:] + b_inner[:-1]) * 0.5, (b_outer[1:] + b_outer[:-1]) * 0.5]).type_as(rays_o)
        ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        inner_mask = norm <= 1
        ray_pts = torch.where(inner_mask, ray_pts, ray_pts / norm * (1 + self.bg_len - self.bg_len / norm))
        return ray_pts, inner_mask.squeeze(-1), t

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres.keys():
            None
            self.fast_color_thres = self._fast_color_thres[global_step]
        ret_dict = {}
        N = len(rays_o)
        ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])
        mask = inner_mask.clone()
        dist_thres = (2 + 2 * self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:, 1:] - ray_pts[:, :-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[None].repeat(N, 1)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask.flatten()]
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(k0)
        else:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3], device=weights.device), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        wsum_mid = segment_coo(src=weights[inner_mask], index=ray_id[inner_mask], out=torch.zeros([N], device=weights.device), reduce='sum')
        s = 1 - 1 / (1 + t)
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'wsum_mid': wsum_mid, 'rgb_marched': rgb_marched, 'raw_density': density, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'step_id': step_id, 'n_max': n_max, 't': t, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([N], device=weights.device), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


class DirectMPIGO(torch.nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, mpi_depth=0, mask_cache_path=None, mask_cache_thres=0.001, mask_cache_world_size=None, fast_color_thres=0, density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=0, **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self._set_grid_resolution(num_voxels, mpi_depth)
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.act_shift = grid.DenseGrid(channels=1, world_size=[1, 1, mpi_depth], xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1.0 / mpi_depth - 1e-06)
            p = [1 - g[0]]
            for i in range(1, len(g)):
                p.append((1 - g[:i + 1].sum()) / (1 - g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1 / self.voxel_size_ratio) - 1))
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2 + self.k0_dim
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
        None
        None
        None
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(path=mask_cache_path, mask_cache_thres=mask_cache_thres)
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2])), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256.0 / mpi_depth
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'mpi_depth': self.mpi_depth, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_path': self.mask_cache_path, 'mask_cache_thres': self.mask_cache_thres, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            dens = self.density.get_dense_grid() + self.act_shift.grid
            self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres
        new_p = self.mask_cache.mask.float().mean().item()
        None

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        None
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += ones.grid.grad > 1
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        None
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        None

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        assert near == 0 and far == 1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth - 1) / stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1, 1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1, -1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples

    def forward(self, rays_o, rays_d, viewdirs, is_train, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
        ray_id = ray_id
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        vox_emb = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(vox_emb)
        else:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3], device=ray_id.device), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        s = (step_id + 0.5) / N_samples
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'rgb_marched': rgb_marched, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'n_max': N_samples, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([N]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


class DirectVoxGO(torch.nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels=0, num_voxels_base=0, alpha_init=None, mask_cache_path=None, mask_cache_thres=0.001, mask_cache_world_size=None, fast_color_thres=0, density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False, rgbnet_depth=3, rgbnet_width=128, viewbase_pe=4, **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        None
        self._set_grid_resolution(num_voxels)
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct, 'rgbnet_full_implicit': rgbnet_full_implicit, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet = None
        else:
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim - 3
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            None
            None
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(path=mask_cache_path, mask_cache_thres=mask_cache_thres)
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2])), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        None
        None
        None
        None

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels': self.num_voxels, 'num_voxels_base': self.num_voxels_base, 'alpha_init': self.alpha_init, 'voxel_size_ratio': self.voxel_size_ratio, 'mask_cache_path': self.mask_cache_path, 'mask_cache_thres': self.mask_cache_thres, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, **self.rgbnet_kwargs}

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
        nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1) for co in cam_o]).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        None
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        None
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        None

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2])), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= cache_grid_alpha > self.fast_color_thres

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        None
        far = 1000000000.0
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr, rays_d_tr):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = torch.from_numpy(rays_o_[::downrate, ::downrate]).flatten(0, -2).split(10000)
                rays_d_ = torch.from_numpy(rays_d_[::downrate, ::downrate]).flatten(0, -2).split(10000)
            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-06), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += ones.grid.grad > 1
        eps_time = time.time() - eps_time
        None
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Check whether the rays hit the solved coarse geometry or not"""
        far = 1000000000.0
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1000000000.0
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        ray_pts, ray_id, step_id = self.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(k0)
        else:
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'rgb_marched': rgb_marched, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * step_id, index=ray_id, out=torch.zeros([N]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


total_variation_cuda = None


class DenseGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        if isinstance(xyz_min, np.ndarray):
            xyz_min, xyz_max = torch.from_numpy(xyz_min), torch.from_numpy(xyz_max)
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        """Add gradients by total variation loss in-place"""
        total_variation_cuda.total_variation_add_grad(self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    xy_feat = F.grid_sample(xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
    feat = torch.cat([xy_feat * z_feat, xz_feat * y_feat, yz_feat * x_feat], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat


def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    xy_feat = F.grid_sample(xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


class TensoRFGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R + R + Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        """
        xyz: global coordinates to query
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, -1, 3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[..., [0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(self.xy_plane, self.xz_plane, self.yz_plane, self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape, self.channels)
        else:
            out = compute_tensorf_val(self.xy_plane, self.xz_plane, self.yz_plane, self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X, Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X, Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y, Z], mode='bilinear', align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X, 1], mode='bilinear', align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y, 1], mode='bilinear', align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z, 1], mode='bilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        """Add gradients by total variation loss in-place"""
        loss = wx * F.smooth_l1_loss(self.xy_plane[:, :, 1:], self.xy_plane[:, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.xy_plane[:, :, :, 1:], self.xy_plane[:, :, :, :-1], reduction='sum') + wx * F.smooth_l1_loss(self.xz_plane[:, :, 1:], self.xz_plane[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.xz_plane[:, :, :, 1:], self.xz_plane[:, :, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.yz_plane[:, :, 1:], self.yz_plane[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.yz_plane[:, :, :, 1:], self.yz_plane[:, :, :, :-1], reduction='sum') + wx * F.smooth_l1_loss(self.x_vec[:, :, 1:], self.x_vec[:, :, :-1], reduction='sum') + wy * F.smooth_l1_loss(self.y_vec[:, :, 1:], self.y_vec[:, :, :-1], reduction='sum') + wz * F.smooth_l1_loss(self.z_vec[:, :, 1:], self.z_vec[:, :, :-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0, :, :, 0]), torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0, :, :, 0]), torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0, :, :, 0])])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0, :, :, 0]) + torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0, :, :, 0]) + torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0, :, :, 0])
            grid = grid[None, None]
        return grid

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config['n_comp']}"


class MaskGrid(nn.Module):

    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = xyz_min
            xyz_max = xyz_max
        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        """Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'


BASIS_TYPE_3D_TEXTURE = 4


BASIS_TYPE_MLP = 255


BASIS_TYPE_SH = 1


def _get_c_extension():
    from warnings import warn
    try:
        if not hasattr(_C, 'sample_grid'):
            _C = None
    except:
        _C = None
    return _C


_C = _get_c_extension()


class SparseGrid(nn.Module):

    def __init__(self, reso: 'Union[int, List[int], Tuple[int, int, int]]'=128, radius: 'Union[float, List[float]]'=1.0, center: 'Union[float, List[float]]'=[0.0, 0.0, 0.0], basis_type: 'int'=BASIS_TYPE_SH, basis_dim: 'int'=9, use_z_order: 'bool'=False, use_sphere_bound: 'bool'=False, mlp_posenc_size: 'int'=0, mlp_width: 'int'=16, background_nlayers: 'int'=0, background_reso: 'int'=256, device: 'Union[torch.device, str]'='cpu'):
        super().__init__()
        self.basis_type = basis_type
        if basis_type == BASIS_TYPE_SH:
            assert utils.isqrt(basis_dim) is not None, 'basis_dim (SH) must be a square number'
        assert basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS, f'basis_dim 1-{utils.MAX_SH_BASIS} supported'
        self.basis_dim = basis_dim
        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width
        self.background_nlayers = background_nlayers
        assert background_nlayers == 0 or background_nlayers > 1, 'Please use at least 2 MSI layers (trilerp limitation)'
        self.background_reso = background_reso
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert len(reso) == 3, 'reso must be an integer or indexable object of 3 ints'
        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
            use_z_order = False
        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device='cpu')
        if isinstance(center, torch.Tensor):
            center = center
        else:
            center = torch.tensor(center, dtype=torch.float32, device='cpu')
        self.radius: 'torch.Tensor' = radius
        self.center: 'torch.Tensor' = center
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius
        n3: 'int' = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(reso[0], device=device, dtype=torch.int32).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)
        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(roffset, points, rscaling)
            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + 3 ** 0.5 / gsz.max()
            self.capacity: 'int' = mask.sum()
            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1
            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3
        self.register_parameter('density_data', nn.Parameter(torch.zeros(self.capacity, 1, dtype=torch.float32, device=device), requires_grad=True))
        self.density_data.grad = torch.zeros_like(self.density_data)
        self.register_parameter('sh_data', nn.Parameter(torch.zeros(self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device), requires_grad=True))
        self.sh_data.grad = torch.zeros_like(self.sh_data)
        self.register_parameter('basis_data', nn.Parameter(torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device), requires_grad=False))
        self.background_links: 'Optional[torch.Tensor]'
        self.background_data: 'Optional[torch.Tensor]'
        if self.use_background:
            background_capacity = self.background_reso ** 2 * 2
            background_links = torch.arange(background_capacity, dtype=torch.int32, device=device).reshape(self.background_reso * 2, self.background_reso)
            self.register_buffer('background_links', background_links)
            self.register_parameter('background_data', nn.Parameter(torch.zeros(background_capacity, self.background_nlayers, 4, dtype=torch.float32, device=device), requires_grad=True))
            self.background_data.grad = torch.zeros_like(self.background_data)
        else:
            self.register_parameter('background_data', nn.Parameter(torch.empty(0, 0, 0, dtype=torch.float32, device=device), requires_grad=False))
        self.register_buffer('links', init_links.view(reso))
        self.links: 'torch.Tensor'
        self.opt = dataclass.RenderOptions()
        self.sparse_grad_indexer: 'Optional[torch.Tensor]' = None
        self.sparse_sh_grad_indexer: 'Optional[torch.Tensor]' = None
        self.sparse_background_indexer: 'Optional[torch.Tensor]' = None
        self.density_rms: 'Optional[torch.Tensor]' = None
        self.sh_rms: 'Optional[torch.Tensor]' = None
        self.background_rms: 'Optional[torch.Tensor]' = None
        self.basis_rms: 'Optional[torch.Tensor]' = None
        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def use_background(self):
        return self.background_nlayers > 0

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros((links.size(0), 1), device=links.device, dtype=torch.float32)
        results_sh = torch.zeros((links.size(0), self.sh_data.size(1)), device=links.device, dtype=torch.float32)
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        return results_sigma, results_sh

    def sample(self, points: 'torch.Tensor', use_kernel: 'bool'=True, grid_coords: 'bool'=False, want_colors: 'bool'=True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
            more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
            else returns density and a dummy tensor to be ignored
            (much faster)

        :return: (density, color)
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            return autograd._SampleGridAutogradFunction.apply(self.density_data, self.sh_data, self._to_cpp(grid_coords=grid_coords), points, want_colors)
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]
            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])
            return samples_sigma, samples_rgb

    def forward(self, points: 'torch.Tensor', use_kernel: 'bool'=True):
        return self.sample(points, use_kernel=use_kernel)

    def volume_render(self, rays: 'dataclass.Rays', use_kernel: 'bool'=True, randomize: 'bool'=False, return_raylen: 'bool'=False):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        basis_data = None
        return autograd._VolumeRenderFunction.apply(self.density_data, self.sh_data, basis_data, self.background_data if self.use_background else None, self._to_cpp(replace_basis_data=basis_data), rays._to_cpp(), self.opt._to_cpp(randomize=randomize), self.opt.backend)

    def volume_render_fused(self, rays: 'dataclass.Rays', rgb_gt: 'torch.Tensor', randomize: 'bool'=False, beta_loss: 'float'=0.0, sparsity_loss: 'float'=0.0):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
            (fused into the backward pass).
            This is average voer the rays in the batch.
            Beta loss also from neural volumes:
            [Lombardi et al., ToG 2019]
        :return: (N, 3), predicted RGB
        """
        grad_density, grad_sh, grad_basis, grad_bg = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt, dtype=torch.float32)
        basis_data: 'Optional[torch.Tensor]' = None
        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),), dtype=torch.bool, device=self.density_data.device)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        if self.basis_type != BASIS_TYPE_SH:
            grad_holder.grad_basis_out = grad_basis
        grad_holder.mask_out = self.sparse_grad_indexer
        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]), dtype=torch.bool, device=self.background_data.device)
            grad_holder.mask_background_out = self.sparse_background_indexer
        cu_fn = _C.__dict__[f'volume_render_{self.opt.backend}_fused']
        cu_fn(self._to_cpp(replace_basis_data=basis_data), rays._to_cpp(), self.opt._to_cpp(randomize=randomize), rgb_gt, beta_loss, sparsity_loss, rgb_out, grad_holder)
        if self.basis_type == BASIS_TYPE_MLP:
            basis_data.backward(grad_basis)
        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out

    def volume_render_depth(self, rays: 'dataclass.Rays', sigma_thresh):
        """
        Volumetric depth rendering for rays

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: finds the first point where sigma strictly exceeds sigma_thresh

        :return: (N,)
        """
        assert not sigma_thresh is None
        return _C.volume_render_sigma_thresh(self._to_cpp(), rays._to_cpp(), self.opt._to_cpp(), sigma_thresh)

    def resample(self, reso: 'Union[int, List[int]]', sigma_thresh: 'float'=5.0, weight_thresh: 'float'=0.01, dilate: 'int'=2, cameras: 'Optional[List[dataclass.Camera]]'=None, use_z_order: 'bool'=False, accelerate: 'bool'=True, weight_render_stop_thresh: 'float'=0.2, max_elements: 'int'=0):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
            (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
            to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
            0.0 = no thresholding, 1.0 = hides everything.
            Useful for force-cutting off junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
            upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert len(reso) == 3, 'reso must be an integer or indexable object of 3 ints'
            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                use_z_order = False
            self.capacity: 'int' = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [(0.5 * curr_reso[i] / reso[i]) for i in range(3)]
            X = torch.linspace(reso_facts[0] - 0.5, curr_reso[0] - reso_facts[0] - 0.5, reso[0], dtype=dtype)
            Y = torch.linspace(reso_facts[1] - 0.5, curr_reso[1] - reso_facts[1] - 0.5, reso[1], dtype=dtype)
            Z = torch.linspace(reso_facts[2] - 0.5, curr_reso[2] - reso_facts[2] - 0.5, reso[2], dtype=dtype)
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points
            use_weight_thresh = cameras is not None
            batch_size = 720720
            all_sample_vals_density = []
            None
            for i in range(0, len(points), batch_size):
                sample_vals_density, _ = self.sample(points[i:i + batch_size], grid_coords=True, want_colors=False)
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None
            sample_vals_density = torch.cat(all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = self._offset * gsz - 0.5
                scaling = self._scaling * gsz
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                None
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(sample_vals_density, cam._to_cpp(), 0.5, weight_render_stop_thresh, False, offset, scaling, max_wt_grid)
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() and max_elements < torch.count_nonzero(sample_vals_mask):
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1), k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    None
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() and max_elements < torch.count_nonzero(sample_vals_mask):
                    sigma_thresh_bounded = torch.topk(sample_vals_density.view(-1), k=max_elements, sorted=False).values.min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    None
                    sample_vals_mask = sample_vals_density >= sigma_thresh
                if self.opt.last_sample_opaque:
                    sample_vals_mask[:, :, -1] = 1
            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()
            points = points[sample_vals_mask]
            None
            all_sample_vals_sh = []
            for i in range(0, len(points), batch_size):
                _, sample_vals_sh = self.sample(points[i:i + batch_size], grid_coords=True, want_colors=True)
                all_sample_vals_sh.append(sample_vals_sh)
            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh
            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full((sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32)
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = torch.cumsum(sample_vals_mask, dim=-1).int() - 1
                init_links[~sample_vals_mask] = -1
            self.capacity = cnz
            None
            del sample_vals_mask
            None
            None
            None
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1))
            self.sh_data = nn.Parameter(sample_vals_sh)
            self.links = init_links.view(reso)
            if accelerate and self.links.is_cuda:
                self.accelerate()

    def sparsify_background(self, sigma_thresh: 'float'=1.0, dilate: 'int'=1):
        device = self.background_links.device
        sigma_mask = torch.zeros(list(self.background_links.shape) + [self.background_nlayers], dtype=torch.bool, device=device).view(-1, self.background_nlayers)
        nonempty_mask = self.background_links.view(-1) >= 0
        data_mask = self.background_data[..., -1] >= sigma_thresh
        sigma_mask[nonempty_mask] = data_mask
        sigma_mask = sigma_mask.view(list(self.background_links.shape) + [self.background_nlayers])
        for _ in range(int(dilate)):
            sigma_mask = _C.dilate(sigma_mask)
        sigma_mask = sigma_mask.any(-1) & nonempty_mask.view(self.background_links.shape)
        self.background_links[~sigma_mask] = -1
        retain_vals = self.background_links[sigma_mask]
        self.background_links[sigma_mask] = torch.arange(retain_vals.size(0), dtype=torch.int32, device=device)
        self.background_data = nn.Parameter(self.background_data.data[retain_vals.long()])

    def resize(self, basis_dim: 'int'):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert utils.isqrt(basis_dim) is not None, 'basis_dim (SH) must be a square number'
        assert basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS, f'basis_dim 1-{utils.MAX_SH_BASIS} supported'
        old_basis_dim = self.basis_dim
        self.basis_dim = basis_dim
        device = self.sh_data.device
        old_data = self.sh_data.data.cpu()
        shrinking = basis_dim < old_basis_dim
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_basis_dim
            arr = torch.arange(basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        else:
            shift = basis_dim
            arr = torch.arange(old_basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        del self.sh_data
        new_data = torch.zeros((old_data.size(0), 3 * basis_dim + 1), device='cpu')
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data
        self.sh_data = nn.Parameter(new_data)
        self.sh_rms = None

    def accelerate(self):
        """
        Accelerate
        """
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(offset, points, scaling)

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(roffset, points, rscaling)

    def tv(self, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0)):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
        mean over voxels)
        """
        assert not logalpha, 'No longer supported'
        return autograd._TotalVariationFunction.apply(self.density_data, self.links, 0, 1, logalpha, logalpha_delta, False, ndc_coeffs)

    def tv_color(self, start_dim: 'int'=0, end_dim: 'Optional[int]'=None, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0)):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
        Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
        Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
        mean over voxels)
        """
        assert not logalpha, 'No longer supported'
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return autograd._TotalVariationFunction.apply(self.sh_data, self.links, start_dim, end_dim, logalpha, logalpha_delta, True, ndc_coeffs)

    def inplace_tv_grad(self, grad: 'torch.Tensor', scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), contiguous: 'bool'=True):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert not logalpha, 'No longer supported'
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(self.links, self.density_data, rand_cells, self._get_sparse_grad_indexer(), 0, 1, scaling, logalpha, logalpha_delta, False, self.opt.last_sample_opaque, ndc_coeffs[0], ndc_coeffs[1], grad)
        else:
            _C.tv_grad(self.links, self.density_data, 0, 1, scaling, logalpha, logalpha_delta, False, ndc_coeffs[0], ndc_coeffs[1], grad)
            self.sparse_grad_indexer: 'Optional[torch.Tensor]' = None

    def inplace_tv_color_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), contiguous: 'bool'=True):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
            Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
            Default None = all dimensions until the end.
        """
        assert not logalpha, 'No longer supported'
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
                _C.tv_grad_sparse(self.links, self.sh_data, rand_cells, indexer, start_dim, end_dim, scaling, logalpha, logalpha_delta, True, False, ndc_coeffs[0], ndc_coeffs[1], grad)
        else:
            _C.tv_grad(self.links, self.sh_data, start_dim, end_dim, scaling, logalpha, logalpha_delta, True, ndc_coeffs[0], ndc_coeffs[1], grad)
            self.sparse_sh_grad_indexer = None

    def inplace_tv_lumisphere_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), dir_factor: 'float'=1.0, dir_perturb_radians: 'float'=0.05):
        assert self.basis_type != BASIS_TYPE_MLP, 'MLP not supported'
        rand_cells = self._get_rand_cells(sparse_frac)
        grad_holder = _C.GridOutputGrads()
        indexer = self._get_sparse_sh_grad_indexer()
        assert indexer is not None
        grad_holder.mask_out = indexer
        grad_holder.grad_sh_out = grad
        batch_size = rand_cells.size(0)
        dirs = torch.randn(3, device=rand_cells.device)
        dirs /= torch.norm(dirs)
        sh_mult = utils.eval_sh_bases(self.basis_dim, dirs[None])
        sh_mult = sh_mult[0]
        if dir_factor > 0.0:
            axis = torch.randn((batch_size, 3))
            axis /= torch.norm(axis, dim=-1, keepdim=True)
            axis *= dir_perturb_radians
            R = Rotation.from_rotvec(axis.numpy()).as_matrix()
            R = torch.from_numpy(R).float()
            dirs_perturb = (R * dirs.unsqueeze(-2)).sum(-1)
        else:
            dirs_perturb = dirs
        sh_mult_u = utils.eval_sh_bases(self.basis_dim, dirs_perturb[None])
        sh_mult_u = sh_mult_u[0]
        _C.lumisphere_tv_grad_sparse(self._to_cpp(), rand_cells, sh_mult, sh_mult_u, scaling, ndc_coeffs[0], ndc_coeffs[1], dir_factor, grad_holder)

    def inplace_l2_color_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0):
        """
        Add gradient of L2 regularization for color
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
            Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
            Default None = all dimensions until the end.
        """
        with torch.no_grad():
            if end_dim is None:
                end_dim = self.sh_data.size(1)
            end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
            start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
            if self.sparse_sh_grad_indexer is None:
                scale = scaling / self.sh_data.size(0)
                grad[:, start_dim:end_dim] += scale * self.sh_data[:, start_dim:end_dim]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
                nz: 'int' = torch.count_nonzero(indexer).item() if indexer.dtype == torch.bool else indexer.size(0)
                scale = scaling / nz
                grad[indexer, start_dim:end_dim] += scale * self.sh_data[indexer, start_dim:end_dim]

    def inplace_tv_background_grad(self, grad: 'torch.Tensor', scaling: 'float'=1.0, scaling_density: 'Optional[float]'=None, sparse_frac: 'float'=0.01, contiguous: 'bool'=False):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'
        """
        rand_cells_bg = self._get_rand_cells_background(sparse_frac, contiguous)
        indexer = self._get_sparse_background_grad_indexer()
        if scaling_density is None:
            scaling_density = scaling
        _C.msi_tv_grad_sparse(self.background_links, self.background_data, rand_cells_bg, indexer, scaling, scaling_density, grad)

    def optim_density_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == 'rmsprop':
            if self.density_rms is None or self.density_rms.shape != self.density_data.shape:
                del self.density_rms
                self.density_rms = torch.zeros_like(self.density_data.data)
            _C.rmsprop_step(self.density_data.data, self.density_rms, self.density_data.grad, indexer, beta, lr, epsilon, -1000000000.0, lr)
        elif optim == 'sgd':
            _C.sgd_step(self.density_data.data, self.density_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_sh_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if optim == 'rmsprop':
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data)
            _C.rmsprop_step(self.sh_data.data, self.sh_rms, self.sh_data.grad, indexer, beta, lr, epsilon, -1000000000.0, lr)
        elif optim == 'sgd':
            _C.sgd_step(self.sh_data.data, self.sh_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_background_step(self, lr_sigma: 'float', lr_color: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        indexer = self._maybe_convert_sparse_grad_indexer(bg=True)
        n_chnl = self.background_data.size(-1)
        if optim == 'rmsprop':
            if self.background_rms is None or self.background_rms.shape != self.background_data.shape:
                del self.background_rms
                self.background_rms = torch.zeros_like(self.background_data.data)
            _C.rmsprop_step(self.background_data.data.view(-1, n_chnl), self.background_rms.view(-1, n_chnl), self.background_data.grad.view(-1, n_chnl), indexer, beta, lr_color, epsilon, -1000000000.0, lr_sigma)
        elif optim == 'sgd':
            _C.sgd_step(self.background_data.data.view(-1, n_chnl), self.background_data.grad.view(-1, n_chnl), indexer, lr_color, lr_sigma)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_basis_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != self.basis_data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(self.basis_data.data)
            self.basis_rms.mul_(beta).addcmul_(self.basis_data.grad, self.basis_data.grad, value=1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            self.basis_data.data.addcdiv_(self.basis_data.grad, denom, value=-lr)
        elif optim == 'sgd':
            self.basis_data.grad.mul_(lr)
            self.basis_data.data -= self.basis_data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        self.basis_data.grad.zero_()

    @property
    def basis_type_name(self):
        if self.basis_type == BASIS_TYPE_SH:
            return 'SH'
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            return '3D_TEXTURE'
        elif self.basis_type == BASIS_TYPE_MLP:
            return 'MLP'
        return 'UNKNOWN'

    def __repr__(self):
        return f'svox2.SparseGrid(basis_type={self.basis_type_name}, ' + f'basis_dim={self.basis_dim}, ' + f'reso={list(self.links.shape)}, ' + f'capacity:{self.sh_data.size(0)})'

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])

    def _to_cpp(self, grid_coords: 'bool'=False, replace_basis_data: 'Optional[torch.Tensor]'=None):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz
        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        if replace_basis_data:
            gspec.basis_data = replace_basis_data
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            gspec.basis_data = self.basis_data
        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device='cpu', dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in ['density_data', 'sh_data', 'basis_data', 'background_data']:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if not hasattr(param, 'grad') or param.grad is None or param.grad.shape != param.data.shape:
                    if hasattr(param, 'grad'):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_background_grad_indexer(self):
        indexer = self.sparse_background_indexer
        if indexer is None:
            indexer = torch.empty((0, 0, 0, 0), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False, bg=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        indexer = self.sparse_sh_grad_indexer if sh else self.sparse_grad_indexer
        if bg:
            indexer = self.sparse_background_indexer
            if indexer is not None:
                indexer = indexer.view(-1)
        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        if indexer.dtype == torch.bool and torch.count_nonzero(indexer).item() < indexer.size(0) // 8:
            indexer = torch.nonzero(indexer.flatten(), as_tuple=False).flatten()
        return indexer

    def _get_rand_cells(self, sparse_frac: 'float', force: 'bool'=False, contiguous: 'bool'=True):
        if sparse_frac < 1.0 or force:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == torch.bool, 'please call sparse loss after rendering and before gradient updates'
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=self.links.device)
                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device)
        return None

    def _get_rand_cells_background(self, sparse_frac: 'float', contiguous: 'bool'=True):
        assert self.use_background, 'Can only use sparse background loss if using background'
        assert self.sparse_background_indexer is None or self.sparse_background_indexer.dtype == torch.bool, 'please call sparse loss after rendering and before gradient updates'
        grid_size = self.background_links.size(0) * self.background_links.size(1) * self.background_data.size(1)
        sparse_num = max(int(sparse_frac * grid_size), 1)
        if contiguous:
            start = np.random.randint(0, grid_size)
            arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=self.links.device)
            if start > grid_size - sparse_num:
                arr[grid_size - sparse_num - start:] -= grid_size
            return arr
        else:
            return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device)

