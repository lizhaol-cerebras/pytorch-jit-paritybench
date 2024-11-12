
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


import torch.nn.functional as F


import time


import functools


import numpy as np


import torch.nn as nn


import enum


from typing import List


from typing import Mapping


from typing import Optional


from typing import Text


from typing import Tuple


from typing import Union


from torch import Tensor


import scipy


from torch.utils import cpp_extension


from scipy.spatial.transform import Rotation as R


from copy import deepcopy


from typing import Sequence


import matplotlib


import matplotlib.pylab as pylab


import matplotlib.pyplot as plt


from matplotlib.lines import Line2D


from scipy.special import jv


from scipy.ndimage import gaussian_filter1d


import copy


import math


import scipy.signal


import torch.optim


from collections import defaultdict


from torch.utils.data import DataLoader


from torch import nn


from torch.optim import SGD


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision import transforms


from torch.utils.data import Dataset


import random


import tensorflow as tf


import torchvision


import torchvision.transforms as T


class NeRFPosEmbedding(nn.Module):

    def __init__(self, num_freqs: 'int', logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(NeRFPosEmbedding, self).__init__()
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(out, -1)


class FourierGrid(nn.Module):

    def __init__(self, grid_len=1000, band_num=10, data_point_num=100):
        super(FourierGrid, self).__init__()
        self.grid_len = grid_len
        self.data_point_num = data_point_num
        self.interval_num = self.grid_len - 1
        self.band_num = band_num
        axis_coord = np.array([(0 + i * 1 / grid_len) for i in range(grid_len)])
        self.ms_x, self.ms_t = np.meshgrid(axis_coord, axis_coord)
        x_coord = np.ravel(self.ms_x).reshape(-1, 1)
        t_coord = np.ravel(self.ms_t).reshape(-1, 1)
        self.x_coord = torch.tensor(x_coord).float()
        self.t_coord = torch.tensor(t_coord).float()
        axis_index = np.array([i for i in range(grid_len)])
        ms_x, ms_t = np.meshgrid(axis_index, axis_index)
        x_ind = np.ravel(ms_x).reshape(-1, 1)
        t_ind = np.ravel(ms_t).reshape(-1, 1)
        self.x_ind = torch.tensor(x_ind).long()
        self.t_ind = torch.tensor(t_ind).long()
        self.voxel = nn.Parameter(torch.rand(grid_len * (self.band_num + 1)), requires_grad=True)

    def gamma_x_i(self, x, i):
        if i % 2 == 0:
            raw_fourier = np.sin((2 ^ i // 2) * np.pi * x)
        else:
            raw_fourier = np.cos((2 ^ i // 2) * np.pi * x)
        fourier = (raw_fourier + 1) / 2
        return fourier

    def forward(self):
        jacobian_y_w = np.zeros((self.data_point_num, self.grid_len * self.band_num))
        for idx in range(self.data_point_num):
            real_x = idx / self.data_point_num
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(real_x, jdx)
                left_grid = int(fourier // (1 / self.grid_len))
                right_grid = right_grid = left_grid + 1
                if left_grid > 0:
                    jacobian_y_w[idx][self.grid_len * jdx + left_grid] = abs(fourier - right_grid * 1 / self.grid_len) * self.grid_len
                if right_grid < self.grid_len:
                    jacobian_y_w[idx][self.grid_len * jdx + right_grid] = abs(fourier - left_grid * 1 / self.grid_len) * self.grid_len
        jacobian_y_w_transpose = np.transpose(jacobian_y_w)
        result_matrix = np.matmul(jacobian_y_w, jacobian_y_w_transpose)
        return result_matrix

    def one_d_regress(self, x_train, x_test, y_train, y_test_gt):
        train_loss = 0
        for idx, one_x in enumerate(x_train):
            y_pred = 0
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(one_x, jdx)
                left_grid = int(fourier * self.interval_num)
                right_grid = left_grid + 1
                left_value = self.voxel[self.grid_len * jdx + left_grid]
                right_value = self.voxel[self.grid_len * jdx + right_grid]
                left_weight = abs(fourier - right_grid * 1 / self.interval_num) * self.interval_num
                right_weight = abs(fourier - left_grid * 1 / self.interval_num) * self.interval_num
                assert abs(left_weight + right_weight - 1) < 0.0001
                y_pred += left_value * left_weight + right_value * right_weight
            y_pred /= self.band_num
            y_pred = torch.sigmoid(y_pred)
            train_loss += torch.nn.functional.mse_loss(y_pred, torch.tensor(y_train[idx]).float())
        y_test = []
        test_loss = []
        for idx, one_x in enumerate(x_test):
            y_pred = 0
            for jdx in range(self.band_num):
                fourier = self.gamma_x_i(one_x, jdx)
                left_grid = int(fourier * self.interval_num)
                right_grid = left_grid + 1
                left_value = self.voxel[self.grid_len * jdx + left_grid]
                right_value = self.voxel[self.grid_len * jdx + right_grid]
                left_weight = abs(fourier - right_grid * 1 / self.interval_num) * self.interval_num
                right_weight = abs(fourier - left_grid * 1 / self.interval_num) * self.interval_num
                y_pred += left_value * left_weight + right_value * right_weight
            y_pred /= self.band_num
            y_pred = torch.sigmoid(y_pred)
            y_test.append(y_pred.item())
            test_loss.append((y_pred.item() - y_test_gt[idx]) ** 2)
        test_loss = np.mean(test_loss)
        return train_loss, test_loss, y_test


class MaskGrid(nn.Module):

    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density_grid = st['model_state_dict']['density.grid']
            if density_grid.shape[1] > 1:
                density_grid = density_grid[0][0].unsqueeze(0).unsqueeze(0)
            density = F.max_pool3d(density_grid, kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)
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


class FourierMSELoss(nn.Module):

    def __init__(self, num_freqs=7, logscale=True):
        super(FourierMSELoss, self).__init__()

    def forward(self, pred, gt):
        fft_dim = -1
        pred_fft = torch.fft.fft(pred, dim=fft_dim)
        gt_fft = torch.fft.fft(gt, dim=fft_dim)
        pred_real, pred_imag = pred_fft.real, pred_fft.imag
        gt_real, gt_imag = gt_fft.real, gt_fft.imag
        real_loss = F.mse_loss(pred_real, gt_real)
        return real_loss


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


class DenseGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
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


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:, :3].T
    rays_o = c2w[:, 3].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]
    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)
    return rays_o, rays_d, viewdirs


class FourierGridModel(nn.Module):

    def __init__(self, xyz_min, xyz_max, num_voxels_density=0, num_voxels_base_density=0, num_voxels_rgb=0, num_voxels_base_rgb=0, num_voxels_viewdir=0, alpha_init=None, mask_cache_world_size=None, fast_color_thres=0, bg_len=0.2, contracted_norm='inf', density_type='DenseGrid', k0_type='DenseGrid', density_config={}, k0_config={}, rgbnet_dim=0, rgbnet_depth=3, rgbnet_width=128, fourier_freq_num=5, viewbase_pe=4, img_emb_dim=-1, verbose=False, **kwargs):
        super(FourierGridModel, self).__init__()
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
        self.verbose = verbose
        self.fourier_freq_num = fourier_freq_num
        self.num_voxels_base_density = num_voxels_base_density
        self.voxel_size_base_density = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base_density).pow(1 / 3)
        self.num_voxels_base_rgb = num_voxels_base_rgb
        self.voxel_size_base_rgb = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base_rgb).pow(1 / 3)
        self.num_voxels_viewdir = num_voxels_viewdir
        self.voxel_size_viewdir = ((torch.Tensor([1, 1, 1]) - torch.Tensor([-1, -1, -1])).prod() / self.num_voxels_viewdir).pow(1 / 3)
        self._set_grid_resolution(num_voxels_density, num_voxels_rgb)
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        if self.verbose:
            None
        self.density_type = density_type
        self.density_config = density_config
        self.world_size = self.world_size_density
        self.density = FourierGrid_grid.create_grid(density_type, channels=1, world_size=self.world_size_density, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True, fourier_freq_num=self.fourier_freq_num, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.img_embed_dim = img_emb_dim
        if 'sample_num' not in kwargs:
            self.sample_num = -1
        else:
            self.sample_num = kwargs['sample_num']
        if img_emb_dim > 0 and self.sample_num > 0:
            self.img_embeddings = nn.Embedding(num_embeddings=self.sample_num, embedding_dim=self.img_embed_dim)
        else:
            self.img_embeddings = None
            self.img_embed_dim = 0
        pos_emb = False
        if pos_emb and self.sample_num > 0:
            self.pos_emb = torch.zeros((self.sample_num, 3), requires_grad=True)
        else:
            self.pos_emb = None
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            self.k0 = FourierGrid_grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size_rgb, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=False, fourier_freq_num=self.fourier_freq_num, config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = FourierGrid_grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size_rgb, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True, fourier_freq_num=self.fourier_freq_num, config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(viewbase_pe)]))
            dim0 = 3 + 3 * viewbase_pe * 2
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True), *[nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True)) for _ in range(rgbnet_depth - 2)], nn.Linear(rgbnet_width, 3))
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            if self.verbose:
                None
                None
        use_view_grid = num_voxels_viewdir > 0
        if use_view_grid:
            self.vd = FourierGrid_grid.create_grid(k0_type, channels=3, world_size=self.world_size_viewdir, xyz_min=torch.Tensor([-1, -1, -1]), xyz_max=torch.Tensor([1, 1, 1]), fourier_freq_num=self.fourier_freq_num, use_nerf_pos=False)
        else:
            self.vd = None
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size_density
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = FourierGrid_grid.MaskGrid(path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    @torch.no_grad()
    def FourierGrid_get_training_rays(self, rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
        if self.pos_emb is not None:
            train_poses[:, :3, 3] = train_poses[:, :3, 3] + self.pos_emb
        assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
        eps_time = time.time()
        DEVICE = rgb_tr_ori[0].device
        N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
        rgb_tr = torch.zeros([N, 3], device=DEVICE)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        indexs_tr = torch.zeros_like(rgb_tr)
        imsz = []
        top = 0
        cur_idx = 0
        for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
            assert img.shape[:2] == (H, W)
            rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
            n = H * W
            rgb_tr[top:top + n].copy_(img.flatten(0, 1))
            rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1))
            rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1))
            viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1))
            indexs_tr[top:top + n].copy_(torch.tensor(cur_idx).long())
            cur_idx += 1
            imsz.append(n)
            top += n
        assert top == N
        eps_time = time.time() - eps_time
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_tr, imsz

    def gather_training_rays(self, data_dict, images, cfg, i_train, cfg_train, poses, HW, Ks, render_kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i] for i in i_train]
        else:
            rgb_tr_ori = images[i_train]
        indexs_train = None
        FourierGrid_datasets = ['waymo', 'mega', 'nerfpp']
        if cfg.data.dataset_type in FourierGrid_datasets or cfg.model == 'FourierGrid':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz = self.FourierGrid_get_training_rays(rgb_tr_ori=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        elif cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(rgb_tr_ori=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, model=self, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(rgb_tr_ori=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(rgb_tr=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda : next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, indexs_train, imsz, batch_index_sampler

    def _set_grid_resolution(self, num_voxels_density, num_voxels_rgb):
        self.num_voxels_density = num_voxels_density
        self.num_voxels_rgb = num_voxels_rgb
        self.voxel_size_density = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_density).pow(1 / 3)
        self.voxel_size_rgb = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_rgb).pow(1 / 3)
        self.voxel_size_viewdir = ((torch.Tensor([1, 1, 1]) - torch.Tensor([-1, -1, -1])).prod() / self.num_voxels_viewdir).pow(1 / 3)
        self.world_size_density = ((self.xyz_max - self.xyz_min) / self.voxel_size_density).long()
        self.world_size_rgb = ((self.xyz_max - self.xyz_min) / self.voxel_size_rgb).long()
        self.world_size_viewdir = (torch.Tensor([1, 1, 1]) - torch.Tensor([-1, -1, -1]) / self.voxel_size_viewdir).long()
        self.world_len_density = self.world_size_density[0].item()
        self.world_len_rgb = self.world_size_rgb[0].item()
        self.world_len_viewdir = self.world_size_viewdir[0].item()
        self.voxel_size_ratio_density = self.voxel_size_density / self.voxel_size_base_density
        self.voxel_size_ratio_rgb = self.voxel_size_rgb / self.voxel_size_base_rgb

    def get_kwargs(self):
        return {'xyz_min': self.xyz_min.cpu().numpy(), 'xyz_max': self.xyz_max.cpu().numpy(), 'num_voxels_density': self.num_voxels_density, 'num_voxels_rgb': self.num_voxels_rgb, 'num_voxels_viewdir': self.num_voxels_viewdir, 'fourier_freq_num': self.fourier_freq_num, 'num_voxels_base_density': self.num_voxels_base_density, 'num_voxels_base_rgb': self.num_voxels_base_rgb, 'alpha_init': self.alpha_init, 'voxel_size_ratio_density': self.voxel_size_ratio_density, 'voxel_size_ratio_rgb': self.voxel_size_ratio_rgb, 'mask_cache_world_size': list(self.mask_cache.mask.shape), 'fast_color_thres': self.fast_color_thres, 'contracted_norm': self.contracted_norm, 'density_type': self.density_type, 'k0_type': self.k0_type, 'density_config': self.density_config, 'k0_config': self.k0_config, 'sample_num': self.sample_num, **self.rgbnet_kwargs}

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        ind_norm = ((cam_o - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        pos_embed = self.density.nerf_pos(ind_norm).squeeze()
        self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.world_size_density[0]), torch.linspace(-1, 1, self.world_size_density[1]), torch.linspace(-1, 1, self.world_size_density[2])), -1)
        for i in range(self.density.pos_embed_output_dim):
            cur_pos_embed = pos_embed[:, 3 * i:3 * (i + 1)].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1) for co in cur_pos_embed.split(10)]).amin(0)
            self.density.grid[0][i][nearest_dist <= near_clip] = -100

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        None
        far = 1000000000.0
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size_density.cpu()) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = DenseGrid(1, self.world_size_density, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].flatten(0, -2).split(10000)
            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-06), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size_density * rng
                interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += ones.grid.grad > 1
        eps_time = time.time() - eps_time
        None
        return count

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels_density, num_voxels_rgb):
        if self.verbose:
            None
        self._set_grid_resolution(num_voxels_density, num_voxels_rgb)
        self.density.scale_volume_grid(self.world_size_density)
        self.k0.scale_volume_grid(self.world_size_rgb)
        if np.prod(self.world_size_density.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size_density[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size_density[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size_density[2])), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
            self.mask_cache = FourierGrid_grid.MaskGrid(path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres), xyz_min=self.xyz_min, xyz_max=self.xyz_max)
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
        if self.verbose:
            None

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        if self.verbose:
            None
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = FourierGrid_grid.FourierGrid(1, self.world_size_density, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, indexs, inner_mask, t, rays_d_e = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += ones.grid.grad > 1
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        if self.verbose:
            None
        eps_time = time.time() - eps_time
        if self.verbose:
            None

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size_density.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size_rgb.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio_density
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        """Check whether the rays hit the solved coarse geometry or not"""
        far = 1000000000.0
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size_density
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        """Sample query points on rays: central sampling.
        Ori_rays_o needs to be properly scaled!
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
        N_inner = int(2 / (2 + 2 * self.bg_len) * self.world_len_density / stepsize) + 1
        N_outer = N_inner
        t_boundary = 1.5
        b_inner = torch.linspace(0, t_boundary, N_inner + 1)
        b_outer = t_boundary / torch.linspace(1, 1 / 128, N_outer + 1)
        t = torch.cat([(b_inner[1:] + b_inner[:-1]) * 0.5, (b_outer[1:] + b_outer[:-1]) * 0.5])
        ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        seperate_boundary = 1.0
        B = 1 + self.bg_len
        order = 1
        A = B * seperate_boundary ** order - seperate_boundary ** (order + 1)
        ray_pts = torch.where(norm <= seperate_boundary, ray_pts, ray_pts / norm * (B - A / norm ** order))
        indexs = None
        rays_d_extend = None
        inner_mask = norm <= seperate_boundary
        return ray_pts, indexs, inner_mask.squeeze(-1), t, rays_d_extend

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        """Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only support point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            if self.verbose:
                None
            self.fast_color_thres = self._fast_color_thres[global_step]
        ret_dict = {}
        num_rays = len(rays_o)
        ray_pts, ray_indexs, inner_mask, t, rays_d_e = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio_density
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])
        mask = inner_mask.clone()
        t = t[None].repeat(num_rays, 1)
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            if rays_d_e is not None:
                rays_d_e = rays_d_e[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask.flatten()]
            step_id = step_id[mask.flatten()]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, num_rays)
        if self.fast_color_thres > 0:
            mask = weights > self.fast_color_thres
            ray_pts = ray_pts[mask]
            if rays_d_e is not None:
                rays_d_e = rays_d_e[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]
        else:
            ray_pts = ray_pts.reshape(-1, ray_pts.shape[-1])
            weights = weights.reshape(-1)
            inner_mask = inner_mask.reshape(-1)
        k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            rgb = torch.sigmoid(k0)
        elif self.vd is not None:
            viewdirs_color = self.vd(viewdirs)[ray_id]
            rgb_logit = k0 + viewdirs_color
            rgb = torch.sigmoid(rgb_logit)
        else:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([num_rays, 3]), reduce='sum')
        if render_kwargs.get('rand_bkgd', False):
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        s = 1 - 1 / (1 + t)
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'rgb_marched': rgb_marched, 'raw_density': density, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'step_id': step_id, 'n_max': n_max, 't': t, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([num_rays]), reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict

    def export_geometry_for_visualize(self, save_path):
        with torch.no_grad():
            dense_grid = self.density.get_dense_grid()
            alpha = self.activate_density(dense_grid).squeeze().cpu().numpy()
            color_grid = self.k0.get_dense_grid()
            rgb = torch.sigmoid(color_grid).squeeze().permute(1, 2, 3, 0).cpu().numpy()
            np.savez_compressed(save_path, alpha=alpha, rgb=rgb)
            None


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
        t = torch.cat([(b_inner[1:] + b_inner[:-1]) * 0.5, (b_outer[1:] + b_outer[:-1]) * 0.5])
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
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only support point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
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
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched)
        else:
            rgb_marched += alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        wsum_mid = segment_coo(src=weights[inner_mask], index=ray_id[inner_mask], out=torch.zeros([N]), reduce='sum')
        s = 1 - 1 / (1 + t)
        ret_dict.update({'alphainv_last': alphainv_last, 'weights': weights, 'wsum_mid': wsum_mid, 'rgb_marched': rgb_marched, 'raw_density': density, 'raw_alpha': alpha, 'raw_rgb': rgb, 'ray_id': ray_id, 'step_id': step_id, 'n_max': n_max, 't': t, 's': s})
        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(src=weights * s, index=ray_id, out=torch.zeros([N]), reduce='sum')
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

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
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
        rgb_marched = segment_coo(src=weights.unsqueeze(-1) * rgb, index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
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
        self.fourier_freq_num = 0
        self.use_fourier_grid = self.fourier_freq_num > 1
        if not self.use_fourier_grid:
            self.density = grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.density_config)
        else:
            self.density = FourierGrid_grid.create_grid(density_type, channels=1, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True, fourier_freq_num=self.fourier_freq_num, config=self.density_config)
        self.rgbnet_kwargs = {'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct, 'rgbnet_full_implicit': rgbnet_full_implicit, 'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width, 'viewbase_pe': viewbase_pe}
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            self.k0_dim = 3
            if not self.use_fourier_grid:
                self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            else:
                self.k0 = FourierGrid_grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True, fourier_freq_num=self.fourier_freq_num, config=self.k0_config)
            self.rgbnet = None
        else:
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            if not self.use_fourier_grid:
                self.k0 = grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, config=self.k0_config)
            else:
                self.k0 = FourierGrid_grid.create_grid(k0_type, channels=self.k0_dim, world_size=self.world_size, xyz_min=self.xyz_min, xyz_max=self.xyz_max, use_nerf_pos=True, fourier_freq_num=self.fourier_freq_num, config=self.k0_config)
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
        if not self.use_fourier_grid:
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]), torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]), torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2])), -1)
            nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1) for co in cam_o.split(100)]).amin(0)
            self.density.grid[nearest_dist[None, None] <= near_clip] = -100
        else:
            ind_norm = ((cam_o - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            pos_embed = self.density.nerf_pos(ind_norm).squeeze()
            self_grid_xyz = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.world_size[0]), torch.linspace(-1, 1, self.world_size[1]), torch.linspace(-1, 1, self.world_size[2])), -1)
            for i in range(self.density.pos_embed_output_dim):
                cur_pos_embed = pos_embed[:, 3 * i:3 * (i + 1)].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                nearest_dist = torch.stack([(self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1) for co in cur_pos_embed.split(10)]).amin(0)
                self.density.grid[0][i][nearest_dist <= near_clip] = -100

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
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].flatten(0, -2).split(10000)
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
        if self.mask_cache is not None and not self.use_fourier_grid:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0 and not self.use_fourier_grid:
            mask = alpha > self.fast_color_thres
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0 and not self.use_fourier_grid:
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


class VoxelGrid(nn.Module):

    def __init__(self, grid_len=1000, data_point_num=100):
        super(VoxelGrid, self).__init__()
        self.grid_len = grid_len
        self.data_point_num = data_point_num
        self.interval_num = grid_len - 1
        axis_coord = np.array([(0 + i * 1 / grid_len) for i in range(grid_len)])
        self.ms_x, self.ms_t = np.meshgrid(axis_coord, axis_coord)
        x_coord = np.ravel(self.ms_x).reshape(-1, 1)
        t_coord = np.ravel(self.ms_t).reshape(-1, 1)
        self.x_coord = torch.tensor(x_coord).float()
        self.t_coord = torch.tensor(t_coord).float()
        axis_index = np.array([i for i in range(grid_len)])
        ms_x, ms_t = np.meshgrid(axis_index, axis_index)
        x_ind = np.ravel(ms_x).reshape(-1, 1)
        t_ind = np.ravel(ms_t).reshape(-1, 1)
        self.x_ind = torch.tensor(x_ind).long()
        self.t_ind = torch.tensor(t_ind).long()
        self.voxel = nn.Parameter(torch.rand(grid_len), requires_grad=True)

    def forward(self):
        jacobian_y_w = np.zeros((self.data_point_num, self.grid_len))
        for idx in range(self.data_point_num):
            real_x = idx / self.data_point_num
            left_grid = int(real_x // (1 / self.grid_len))
            right_grid = left_grid + 1
            if left_grid >= 0:
                jacobian_y_w[idx][left_grid] = abs(real_x - right_grid * 1 / self.grid_len) * self.grid_len
            if right_grid < self.grid_len:
                jacobian_y_w[idx][right_grid] = abs(real_x - left_grid * 1 / self.grid_len) * self.grid_len
        jacobian_y_w_transpose = np.transpose(jacobian_y_w)
        result_matrix = np.matmul(jacobian_y_w, jacobian_y_w_transpose)
        return result_matrix

    def one_d_regress(self, x_train, x_test, y_train, y_test_gt):
        train_loss = 0
        for idx, one_x in enumerate(x_train):
            left_grid = int(one_x // (1 / self.interval_num))
            right_grid = left_grid + 1
            left_value = self.voxel[left_grid]
            right_value = self.voxel[right_grid]
            left_weight = abs(one_x - right_grid * 1 / self.interval_num) * self.interval_num
            right_weight = abs(one_x - left_grid * 1 / self.interval_num) * self.interval_num
            y_pred = left_value * left_weight + right_value * right_weight
            y_pred = torch.sigmoid(y_pred)
            train_loss += torch.nn.functional.mse_loss(y_pred, torch.tensor(y_train[idx]).float())
        y_test = []
        test_loss = []
        for idx, one_x in enumerate(x_test):
            left_grid = int(one_x // (1 / self.interval_num))
            right_grid = left_grid + 1
            left_value = self.voxel[left_grid]
            right_value = self.voxel[right_grid]
            left_weight = abs(one_x - right_grid * 1 / self.interval_num) * self.interval_num
            right_weight = abs(one_x - left_grid * 1 / self.interval_num) * self.interval_num
            y_pred = left_value * left_weight + right_value * right_weight
            y_pred = torch.sigmoid(y_pred)
            y_test.append(y_pred.item())
            test_loss.append((y_pred.item() - y_test_gt[idx]) ** 2)
        test_loss = np.mean(test_loss)
        return train_loss, test_loss, y_test


class BlockNeRFLoss(nn.Module):

    def __init__(self, lambda_mu=0.01, Visi_loss=0.01):
        super(BlockNeRFLoss, self).__init__()
        self.lambda_mu = lambda_mu
        self.Visi_loss = Visi_loss

    def forward(self, inputs, targets):
        loss = {}
        loss['rgb_coarse'] = self.lambda_mu * ((inputs['rgb_coarse'] - targets[..., :3]) ** 2).mean()
        loss['rgb_fine'] = ((inputs['rgb_fine'] - targets[..., :3]) ** 2).mean()
        loss['transmittance_coarse'] = self.lambda_mu * self.Visi_loss * ((inputs['transmittance_coarse_real'].detach() - inputs['transmittance_coarse_vis'].squeeze()) ** 2).mean()
        loss['transmittance_fine'] = self.Visi_loss * ((inputs['transmittance_fine_real'].detach() - inputs['transmittance_fine_vis'].squeeze()) ** 2).mean()
        return loss


class InterPosEmbedding(nn.Module):

    def __init__(self, N_freqs=10):
        super(InterPosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        self.freq_band_1 = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        self.freq_band_2 = self.freq_band_1 ** 2

    def forward(self, mu, diagE):
        sin_out = []
        sin_cos = []
        for freq in self.freq_band_1:
            for func in self.funcs:
                sin_cos.append(func(freq * mu))
            sin_out.append(sin_cos)
            sin_cos = []
        diag_out = []
        for freq in self.freq_band_2:
            diag_out.append(freq * diagE)
        out = []
        for sc_γ, diag_Eγ in zip(sin_out, diag_out):
            for sin_cos in sc_γ:
                out.append(sin_cos * torch.exp(-0.5 * diag_Eγ))
        return torch.cat(out, -1)


class PosEmbedding(nn.Module):

    def __init__(self, N_freqs):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)

    def forward(self, x):
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


class Block_NeRF(nn.Module):

    def __init__(self, D=8, W=256, skips=[4], in_channel_xyz=60, in_channel_dir=24, in_channel_exposure=8, in_channel_appearance=32, add_apperance=True, add_exposure=True):
        super(Block_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channel_xyz = in_channel_xyz
        self.in_channel_dir = in_channel_dir
        self.in_channel_exposure = in_channel_exposure
        self.in_channel_appearance = in_channel_appearance
        self.add_appearance = add_apperance
        self.add_exposure = add_exposure
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channel_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channel_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i + 1}', layer)
        self.xyz_encoding_final = nn.Linear(W, W)
        input_channel = W + in_channel_dir
        if add_apperance:
            input_channel += in_channel_appearance
        if add_exposure:
            input_channel += in_channel_exposure
        self.dir_encoding = nn.Sequential(nn.Linear(input_channel, W // 2), nn.ReLU(True), nn.Linear(W // 2, W // 2), nn.ReLU(True), nn.Linear(W // 2, W // 2), nn.ReLU(True))
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        if sigma_only:
            input_xyz = x
        else:
            input_xyz, input_dir, input_exp, input_appear = torch.split(x, [self.in_channel_xyz, self.in_channel_dir, self.in_channel_exposure, self.in_channel_appearance], dim=-1)
        xyz = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz = torch.cat([xyz, input_xyz], dim=-1)
            xyz = getattr(self, f'xyz_encoding_{i + 1}')(xyz)
        static_sigma = self.static_sigma(xyz)
        if sigma_only:
            return static_sigma
        xyz_feature = self.xyz_encoding_final(xyz)
        input_xyz_feature = torch.cat([xyz_feature, input_dir], dim=-1)
        if self.add_exposure:
            input_xyz_feature = torch.cat([input_xyz_feature, input_exp], dim=-1)
        if self.add_appearance:
            input_xyz_feature = torch.cat([input_xyz_feature, input_appear], dim=-1)
        dir_encoding = self.dir_encoding(input_xyz_feature)
        static_rgb = self.static_rgb(dir_encoding)
        static_rgb_sigma = torch.cat([static_rgb, static_sigma], dim=-1)
        return static_rgb_sigma


class Visibility(nn.Module):

    def __init__(self, in_channel_xyz=60, in_channel_dir=24, W=128):
        super(Visibility, self).__init__()
        self.in_channel_xyz = in_channel_xyz
        self.in_channel_dir = in_channel_dir
        self.vis_encoding = nn.Sequential(nn.Linear(in_channel_xyz + in_channel_dir, W), nn.ReLU(True), nn.Linear(W, W), nn.ReLU(True), nn.Linear(W, W), nn.ReLU(True), nn.Linear(W, W), nn.ReLU(True))
        self.visibility = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

    def forward(self, x):
        vis_encode = self.vis_encoding(x)
        visibility = self.visibility(vis_encode)
        return visibility


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FourierGrid,
     lambda: ([], {}),
     lambda: ([], {})),
    (FourierMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (InterPosEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (NeRFPosEmbedding,
     lambda: ([], {'num_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PosEmbedding,
     lambda: ([], {'N_freqs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VoxelGrid,
     lambda: ([], {}),
     lambda: ([], {})),
]

