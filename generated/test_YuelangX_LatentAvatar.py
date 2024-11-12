
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


from torch.nn import DataParallel


import torchvision as tv


from torch.utils.data import Dataset


import numpy as np


import random


import torch.nn.functional as F


import math


from torch import nn


from torch.nn import functional as F


import torch.nn as nn


from torch import einsum


from math import floor


from math import log2


from functools import partial


import time


import copy


class Encoder(nn.Module):

    def __init__(self, latent_dim, dims):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.dims = dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(dims[i + 1]), nn.LeakyReLU()))
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(dims[-1] * 8 * 8, latent_dim)

    def encode(self, input):
        z = self.encoder(input)
        z = torch.flatten(z, start_dim=1)
        z = self.fc(z)
        mean = torch.mean(z, 1, keepdim=True)
        var = torch.var(z, 1, keepdim=True)
        z = (z - mean) / var
        return z

    def forward(self, input):
        return self.encode(input)


class Conv2DMod(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-08, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape
        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x


def exists(val):
    return val is not None


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class GeneratorBlock(nn.Module):

    def __init__(self, latent_dim, input_channels, filters, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)
        self.activation = leaky_relu()

    def forward(self, x, istyle):
        if exists(self.upsample):
            x = self.upsample(x)
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x)
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)
        return x


class Generator(nn.Module):

    def __init__(self, image_size, latent_dim, network_capacity=48, transparent=False, fmap_max=256):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        filters = [(network_capacity * 2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])
        self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != self.num_layers - 1
            block = GeneratorBlock(latent_dim, in_chan, out_chan, upsample=not_first)
            self.blocks.append(block)

    def forward(self, style):
        avg_style = style[:, :, None, None]
        x = self.to_initial_block(avg_style)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x, style)
        return x


class MLP(nn.Module):

    def __init__(self, dims, last_op=None):
        super(MLP, self).__init__()
        self.dims = dims
        self.last_op = last_op
        if len(dims) < 5:
            self.skip = None
        else:
            self.skip = int(len(dims) / 2)
        if self.skip:
            layers = []
            for i in range(self.skip - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            self.layers1 = nn.Sequential(*layers)
            layers = []
            layers.append(nn.Linear(dims[self.skip] + dims[0], dims[self.skip + 1]))
            layers.append(nn.LeakyReLU())
            for i in range(self.skip + 1, len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.layers2 = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(0, len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.skip:
            y = self.layers1(x)
            y = torch.cat([y, x], dim=1)
            y = self.layers2(y)
        else:
            y = self.layers(x)
        if self.last_op:
            y = self.last_op(y)
        return y


class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {'include_input': True, 'input_dims': 1, 'max_freq_log2': multires - 1, 'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class HeadModule(nn.Module):

    def __init__(self, opt):
        super(HeadModule, self).__init__()
        self.generator = Generator(opt.triplane_res, opt.exp_dim_3d, opt.triplane_dim * 3 // 2)
        self.density_mlp = MLP(opt.density_mlp, last_op=None)
        self.color_mlp = MLP(opt.color_mlp, last_op=None)
        self.pos_embedding, _ = get_embedder(opt.pos_freq)
        self.view_embedding, _ = get_embedder(opt.view_freq)
        self.noise = opt.noise
        self.bbox = opt.bbox

    def forward(self, data):
        B, C, N = data['query_pts'].shape
        query_pts = data['query_pts']
        query_viewdirs = data['query_viewdirs']
        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, 3:, None]
            S = data['scale'][:, :, None]
            query_pts = torch.bmm(R.permute(0, 2, 1), query_pts - T) / S
            query_viewdirs = torch.bmm(R.permute(0, 2, 1), query_viewdirs)
        query_viewdirs_embedding = self.view_embedding(rearrange(query_viewdirs, 'b c n -> (b n) c'))
        triplanes = self.generate(data)
        plane_dim = triplanes.shape[1] // 3
        plane_x = triplanes[:, plane_dim * 0:plane_dim * 1, :, :]
        plane_y = triplanes[:, plane_dim * 1:plane_dim * 2, :, :]
        plane_z = triplanes[:, plane_dim * 2:plane_dim * 3, :, :]
        u = (query_pts[:, 0:1] - 0.5 * (self.bbox[0][0] + self.bbox[0][1])) / (0.5 * (self.bbox[0][1] - self.bbox[0][0]))
        v = (query_pts[:, 1:2] - 0.5 * (self.bbox[1][0] + self.bbox[1][1])) / (0.5 * (self.bbox[1][1] - self.bbox[1][0]))
        w = (query_pts[:, 2:3] - 0.5 * (self.bbox[2][0] + self.bbox[2][1])) / (0.5 * (self.bbox[2][1] - self.bbox[2][0]))
        vw = rearrange(torch.cat([v, w], dim=1), 'b (t c) n -> b n t c', t=1)
        uw = rearrange(torch.cat([u, w], dim=1), 'b (t c) n -> b n t c', t=1)
        uv = rearrange(torch.cat([u, v], dim=1), 'b (t c) n -> b n t c', t=1)
        feature_x = torch.nn.functional.grid_sample(plane_x, vw, align_corners=True, mode='bilinear')
        feature_y = torch.nn.functional.grid_sample(plane_y, uw, align_corners=True, mode='bilinear')
        feature_z = torch.nn.functional.grid_sample(plane_z, uv, align_corners=True, mode='bilinear')
        feature_x = rearrange(feature_x, 'b c n t -> b c (n t)')
        feature_y = rearrange(feature_y, 'b c n t -> b c (n t)')
        feature_z = rearrange(feature_z, 'b c n t -> b c (n t)')
        feature = feature_x + feature_y + feature_z
        feature = rearrange(feature, 'b c n -> (b n) c')
        density = rearrange(self.density_mlp(feature), '(b n) c -> b c n', b=B)
        if self.training:
            density = density + torch.randn_like(density) * self.noise
        color_input = torch.cat([feature, query_viewdirs_embedding], 1)
        color = rearrange(self.color_mlp(color_input), '(b n) c -> b c n', b=B)
        data['density'] = density
        data['color'] = color
        return data

    def generate(self, data):
        code = data['exp_code_3d']
        triplanes = self.generator(code)
        return triplanes


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        y = self.act_fn(self.conv(x))
        return y


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class Upsampler(nn.Module):

    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(Upsampler, self).__init__()
        dims = [(network_capacity * s) for s in [4, 8, 4, 2, 1]]
        self.inc = DoubleConv(input_dim, dims[0])
        self.down1 = Down(dims[0], dims[1])
        self.up1 = Up(dims[0] + dims[1], dims[2])
        self.up2 = Up(dims[2], dims[3])
        self.up3 = Up(dims[3], dims[4])
        self.outc = OutConv(dims[4], output_dim)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        y = self.up1(x2, x1)
        y = self.up2(y)
        y = self.up3(y)
        y = self.outc(y)
        return y


class AvatarModule(nn.Module):

    def __init__(self, opt):
        super(AvatarModule, self).__init__()
        self.exp_dim_2d = opt.exp_dim_2d
        self.encoder = Encoder(opt.exp_dim_2d, opt.encoder_dims)
        self.mapping_mlp = MLP(opt.mapping_dims)
        self.headmodule = HeadModule(opt.headmodule)
        self.upsampler = Upsampler(opt.headmodule.color_mlp[-1], 3, opt.upsampler_capacity)

    def encode(self, input):
        return self.encoder(input - 0.5)

    def mapping(self, exp_code_2d):
        return self.mapping_mlp(exp_code_2d)

    def head(self, data):
        return self.headmodule(data)

    def upsample(self, feature_map):
        return self.upsampler(feature_map)

    def forward(self, func, data):
        if func == 'encode':
            return self.encode(data)
        elif func == 'mapping':
            return self.mapping(data)
        elif func == 'head':
            return self.head(data)
        elif func == 'upsample':
            return self.upsample(data)


class NeuralCameraModule(nn.Module):

    def __init__(self, avatarmodule, opt):
        super(NeuralCameraModule, self).__init__()
        self.avatarmodule = avatarmodule
        self.model_bbox = opt.model_bbox
        self.image_size = opt.image_size
        self.N_samples = opt.N_samples
        self.near_far = opt.near_far

    @staticmethod
    def gen_part_rays(extrinsic, intrinsic, resolution, image_size):
        rays_o_list = []
        rays_d_list = []
        rot = extrinsic[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, extrinsic[:, :3, 3:])
        c2w = torch.cat((rot, trans.reshape(-1, 3, 1)), dim=2)
        for b in range(intrinsic.shape[0]):
            fx, fy, cx, cy = intrinsic[b, 0, 0], intrinsic[b, 1, 1], intrinsic[b, 0, 2], intrinsic[b, 1, 2]
            res_w = resolution[b, 0].int().item()
            res_h = resolution[b, 1].int().item()
            W = image_size[b, 0].int().item()
            H = image_size[b, 1].int().item()
            i, j = torch.meshgrid(torch.linspace(0.5, W - 0.5, res_w, device=c2w.device), torch.linspace(0.5, H - 0.5, res_h, device=c2w.device))
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
            rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[b, :3, :3], -1)
            rays_o = c2w[b, :3, -1].expand(rays_d.shape)
            rays_o_list.append(rays_o.unsqueeze(0))
            rays_d_list.append(rays_d.unsqueeze(0))
        rays_o_list = torch.cat(rays_o_list, dim=0)
        rays_d_list = torch.cat(rays_d_list, dim=0)
        return rearrange(rays_o_list, 'b h w c -> b c h w'), rearrange(rays_d_list, 'b h w c -> b c h w')

    @staticmethod
    def coords_select(image, coords):
        select_rays = []
        for i in range(image.shape[0]):
            select_rays.append(image[i, :, coords[i, :, 1], coords[i, :, 0]].unsqueeze(0))
        select_rays = torch.cat(select_rays, dim=0)
        return select_rays

    @staticmethod
    def gen_near_far_fixed(near, far, samples, batch_size, device):
        nf = torch.zeros((batch_size, 2, samples), device=device)
        nf[:, 0, :] = near
        nf[:, 1, :] = far
        return nf

    def gen_near_far(self, rays_o, rays_d, R, T, S):
        """calculate intersections with 3d bounding box for batch"""
        B = rays_o.shape[0]
        rays_o_can = torch.bmm(R.permute(0, 2, 1), rays_o - T) / S
        rays_d_can = torch.bmm(R.permute(0, 2, 1), rays_d) / S
        bbox = torch.tensor(self.model_bbox, dtype=rays_o.dtype, device=rays_o.device)
        mask_in_box_batch = []
        near_batch = []
        far_batch = []
        for b in range(B):
            norm_d = torch.linalg.norm(rays_d_can[b], axis=0, keepdims=True)
            viewdir = rays_d_can[b] / norm_d
            viewdir[(viewdir < 1e-05) & (viewdir > -1e-10)] = 1e-05
            viewdir[(viewdir > -1e-05) & (viewdir < 1e-10)] = -1e-05
            tmin = (bbox[:, :1] - rays_o_can[b, :, :1]) / viewdir
            tmax = (bbox[:, 1:2] - rays_o_can[b, :, :1]) / viewdir
            t1 = torch.minimum(tmin, tmax)
            t2 = torch.maximum(tmin, tmax)
            near = torch.max(t1, 0)[0]
            far = torch.min(t2, 0)[0]
            mask_in_box = near < far
            mask_in_box_batch.append(mask_in_box)
            near_batch.append(near / norm_d[0])
            far_batch.append(far / norm_d[0])
        mask_in_box_batch = torch.stack(mask_in_box_batch)
        near_batch = torch.stack(near_batch)
        far_batch = torch.stack(far_batch)
        return near_batch, far_batch, mask_in_box_batch

    @staticmethod
    def sample_pdf(density, z_vals, rays_d, N_importance):
        """sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """
        bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        _, _, _, weights = NeuralCameraModule.integrate(density, z_vals, rays_d)
        weights = weights[..., 1:-1] + 1e-05
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], dtype=weights.dtype, device=weights.device)
        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)
        matched_shape = inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-05, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        sample_z = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        sample_z, _ = torch.sort(sample_z, dim=-1)
        return sample_z

    @staticmethod
    def integrate(density, z_vals, rays_d, color=None, method='nerf'):
        """Transforms module's predictions to semantically meaningful values.
        Args:
            density: [num_rays, num_samples along ray, 4]. Prediction from module.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            acc_map: [num_rays]. Sum of weights along each ray.
            depth_map: [num_rays]. Estimated distance to object.
        """
        dists = (z_vals[..., 1:] - z_vals[..., :-1]) * 100.0
        dists = torch.cat([dists, torch.ones(1, device=density.device).expand(dists[..., :1].shape) * 10000000000.0], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        if method == 'nerf':
            alpha = 1 - torch.exp(-F.relu(density[..., 0]) * dists)
        elif method == 'unisurf':
            alpha = density[..., 0]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=density.device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
        acc_map = torch.sum(weights, -1)
        depth_map = torch.sum(weights * z_vals, -1)
        if color == None:
            return None, acc_map, depth_map, weights
        rgb_map = torch.sum(weights[..., None] * color, -2)
        return rgb_map, acc_map, depth_map, weights

    def render_rays(self, data, N_samples=64):
        B, C, N = data['rays_o'].shape
        rays_o = rearrange(data['rays_o'], 'b c n -> (b n) c')
        rays_d = rearrange(data['rays_d'], 'b c n -> (b n) c')
        N_rays = rays_o.shape[0]
        rays_nf = rearrange(data['rays_nf'], 'b c n -> (b n) c')
        near, far = rays_nf[..., :1], rays_nf[..., 1:]
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=rays_o.device).unsqueeze(0)
        z_vals = near * (1 - t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, N_samples])
        query_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B)
        query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_samples, 1), '(b n) s c -> b c (n s)', b=B)
        data['query_pts'] = query_pts
        data['query_viewdirs'] = query_viewdirs
        data = self.avatarmodule('head', data)
        density = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
        color = rearrange(data['color'], 'b c (n s) -> (b n) s c', n=N)
        density = torch.cat([density, torch.ones([density.shape[0], 1, density.shape[2]], device=density.device) * 100000000.0], 1)
        color = torch.cat([color, torch.ones([color.shape[0], 1, color.shape[2]], device=color.device)], 1)
        z_vals = torch.cat([z_vals, torch.ones([z_vals.shape[0], 1], device=z_vals.device) * 100000000.0], 1)
        render_image, render_mask, _, _ = NeuralCameraModule.integrate(density, z_vals, rays_d, color=color, method='nerf')
        render_image = rearrange(render_image, '(b n) c -> b c n', b=B)
        render_mask = rearrange(render_mask, '(b n c) -> b c n', b=B, c=1)
        data.update({'render_image': render_image, 'render_mask': render_mask})
        return data

    def forward(self, data, resolution):
        B = data['exp_code_3d'].shape[0]
        H = W = resolution // 4
        device = data['exp_code_3d'].device
        rays_o_grid, rays_d_grid = self.gen_part_rays(data['extrinsic'], data['intrinsic'], torch.FloatTensor([[H, W]]).repeat(B, 1), torch.FloatTensor([[self.image_size, self.image_size]]).repeat(B, 1))
        rays_o = rearrange(rays_o_grid, 'b c h w -> b c (h w)')
        rays_d = rearrange(rays_d_grid, 'b c h w -> b c (h w)')
        rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
        R = so3_exponential_map(data['pose'][:, :3])
        T = data['pose'][:, 3:, None]
        S = data['scale'][:, :, None]
        rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
        for b in range(B):
            rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
            rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]
        render_data = {'exp_code_3d': data['exp_code_3d'], 'pose': data['pose'], 'scale': data['scale'], 'rays_o': rays_o, 'rays_d': rays_d, 'rays_nf': rays_nf}
        render_data = self.render_rays(render_data, N_samples=self.N_samples)
        render_feature = rearrange(render_data['render_image'], 'b c (h w) -> b c h w', h=H)
        render_mask = rearrange(render_data['render_mask'], 'b c (h w) -> b c h w', h=H)
        render_image = self.avatarmodule('upsample', render_feature)
        data['render_feature'] = render_feature
        data['render_image'] = render_image
        data['render_mask'] = render_mask
        return data


class Decoder(nn.Module):

    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.dims = dims
        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Sequential(nn.Conv2d(dims[i], 4 * dims[i + 1], kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(4 * dims[i + 1]), nn.LeakyReLU(), nn.PixelShuffle(2), nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(dims[i + 1]), nn.LeakyReLU(0.2), nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(dims[i + 1]), nn.LeakyReLU(0.2)))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.Conv2d(dims[-2], dims[-1], kernel_size=1, padding=0), nn.Sigmoid())

    def decode(self, input):
        result = self.decoder(input)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        return self.decode(input)


class NeckDecoder(nn.Module):

    def __init__(self, latent_dim, dims):
        super(NeckDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.dims = dims
        self.fc = nn.Linear(latent_dim, dims[0] * 8 * 8)
        self.upsampler = nn.Sequential(nn.Conv2d(dims[0], 4 * dims[1], kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(4 * dims[1]), nn.LeakyReLU(), nn.PixelShuffle(2))

    def decode(self, z):
        result = self.fc(z)
        result = result.view(-1, self.dims[0], 8, 8)
        result = self.upsampler(result)
        return result

    def forward(self, input):
        return self.decode(input)


class YVAEModule(nn.Module):

    def __init__(self, opt):
        super(YVAEModule, self).__init__()
        self.exp_dim_2d = opt.exp_dim_2d
        self.domains = opt.domains
        self.shared_encoder = Encoder(opt.exp_dim_2d, opt.encoder_dims)
        self.shared_neckdecoder = NeckDecoder(opt.exp_dim_2d, opt.neck_dims)
        self.mapping_mlp = MLP(opt.mapping_dims)
        self.decoder_dict = {}
        for domain in opt.domains:
            self.decoder_dict[domain] = Decoder(opt.decoder_dims)
        self.decoder_dict = nn.ModuleDict(self.decoder_dict)

    def encode(self, input):
        exp_code_2d = self.shared_encoder(input - 0.5)
        return exp_code_2d

    def decode(self, exp_code_2d, domain):
        result = self.shared_neckdecoder(exp_code_2d)
        result = self.decoder_dict[domain](result)
        return result

    def mapping(self, exp_code_2d):
        return self.mapping_mlp(exp_code_2d)

    def forward(self, func, data):
        if func == 'encode':
            return self.encode(data)
        elif func == 'decode':
            return self.decode(data[0], data[1])
        elif func == 'mapping':
            return self.mapping(data)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Decoder,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Down,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Generator,
     lambda: ([], {'image_size': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (GeneratorBlock,
     lambda: ([], {'latent_dim': 4, 'input_channels': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})),
    (MLP,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Up,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsampler,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

