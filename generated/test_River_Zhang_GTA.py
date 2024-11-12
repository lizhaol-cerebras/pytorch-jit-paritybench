
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


from torch import nn


import logging


import warnings


import math


from typing import NewType


import torch.nn as nn


import torch.nn.functional as F


import matplotlib.pyplot as plt


from torchvision.utils import make_grid


from torch.utils.data import DataLoader


import random


import torchvision.transforms as transforms


import torchvision


from scipy.spatial import cKDTree


from collections import namedtuple


from torch.nn import functional as F


import functools


from torch.autograd import Function


from torchvision import models


from torch.autograd import grad


from functools import partial


from typing import Union


from typing import Tuple


from typing import List


from typing import Optional


import torch.sparse as sp


from torch.nn import init


from torch.hub import load_state_dict_from_url


from torch.nn.modules.utils import _pair


from collections import OrderedDict


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import BasicBlock


from torch.nn.parameter import Parameter


import torch.optim as optim


from scipy.ndimage import morphology


import torchvision.models.resnet as resnet


import scipy


import scipy.misc


from torchvision.models import detection


from torchvision import transforms


from typing import Dict


class cleanShader(torch.nn.Module):

    def __init__(self, device='cpu', cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of TexturedSoftPhongShader'
            raise ValueError(msg)
        blend_params = kwargs.get('blend_params', self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)
        return images


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    This class implements methods for rasterizing a batch of heterogenous Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {'image_size': image_size, 'blur_radius': 0.0, 'faces_per_pixel': 1, 'bin_size': None, 'max_faces_per_bin': None, 'perspective_correct': False}
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(meshes_screen, image_size=raster_settings.image_size, blur_radius=raster_settings.blur_radius, faces_per_pixel=raster_settings.faces_per_pixel, bin_size=raster_settings.bin_size, max_faces_per_bin=raster_settings.max_faces_per_bin, perspective_correct=raster_settings.perspective_correct)
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SmoothConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size for smooth_conv must be odd: {3, 5, ...}'
        self.padding = (kernel_size - 1) // 2
        weight = torch.ones((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=torch.float32) / kernel_size ** 3
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv3d(input, self.weight, padding=self.padding)


def create_grid3D(min, max, steps):
    if type(min) is int:
        min = min, min, min
    if type(max) is int:
        max = max, max, max
    if type(steps) is int:
        steps = steps, steps, steps
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).long()
    gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH, gridD])
    coords = coords.view(3, -1).t()
    return coords


def plot_mask3D(mask=None, title='', point_coords=None, figsize=1500, point_marker_size=8, interactive=True):
    """
    Simple plotting tool to show intermediate mask predictions and points 
    where PointRend is applied.

    Args:
    mask (Tensor): mask prediction of shape DxHxW
    title (str): title for the plot
    point_coords ((Tensor, Tensor, Tensor)): x and y and z point coordinates
    figsize (int): size of the figure to plot
    point_marker_size (int): marker size for points
    """
    vp = vtkplotter.Plotter(title=title, size=(figsize, figsize))
    vis_list = []
    if mask is not None:
        mask = mask.detach().numpy()
        mask = mask.transpose(2, 1, 0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.5, gradient_direction='ascent')
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        vis_list.append(mesh)
    if point_coords is not None:
        point_coords = torch.stack(point_coords, 1).numpy()
        pc = vtkplotter.Points(point_coords, r=point_marker_size, c='red')
        vis_list.append(pc)
    vp.show(*vis_list, bg='white', axes=1, interactive=interactive, azimuth=30, elevation=30)


class Seg3dLossless(nn.Module):

    def __init__(self, query_func, b_min, b_max, resolutions, channels=1, balance_value=0.5, align_corners=False, visualize=False, debug=False, use_cuda_impl=False, faster=False, use_shadow=False, **kwargs):
        """
        align_corners: same with how you process gt. (grid_sample / interpolate) 
        """
        super().__init__()
        self.query_func = query_func
        self.register_buffer('b_min', torch.tensor(b_min).float().unsqueeze(1))
        self.register_buffer('b_max', torch.tensor(b_max).float().unsqueeze(1))
        if type(resolutions[0]) is int:
            resolutions = torch.tensor([(res, res, res) for res in resolutions])
        else:
            resolutions = torch.tensor(resolutions)
        self.register_buffer('resolutions', resolutions)
        self.batchsize = self.b_min.size(0)
        assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels
        assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize
        self.debug = debug
        self.use_cuda_impl = use_cuda_impl
        self.faster = faster
        self.use_shadow = use_shadow
        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, f'resolution {resolution} need to be odd becuase of align_corner.'
        init_coords = create_grid3D(0, resolutions[-1] - 1, steps=resolutions[0])
        init_coords = init_coords.unsqueeze(0).repeat(self.batchsize, 1, 1)
        self.register_buffer('init_coords', init_coords)
        calculated = torch.zeros((self.resolutions[-1][2], self.resolutions[-1][1], self.resolutions[-1][0]), dtype=torch.bool)
        self.register_buffer('calculated', calculated)
        gird8_offsets = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])])).int().view(3, -1).t()
        self.register_buffer('gird8_offsets', gird8_offsets)
        self.smooth_conv3x3 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=3)
        self.smooth_conv5x5 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=5)
        self.smooth_conv7x7 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=7)
        self.smooth_conv9x9 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=9)

    def batch_eval(self, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
        coords = coords.detach()
        if self.align_corners:
            coords2D = coords.float() / (self.resolutions[-1] - 1)
        else:
            step = 1.0 / self.resolutions[-1].float()
            coords2D = coords.float() / self.resolutions[-1] + step / 2
        coords2D = coords2D * (self.b_max - self.b_min) + self.b_min
        occupancys = self.query_func(**kwargs, points=coords2D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys)
        assert len(occupancys.size()) == 3, 'query_func should return a occupancy with shape of [bz, C, N]'
        return occupancys

    def forward(self, **kwargs):
        if self.faster:
            return self._forward_faster(**kwargs)
        else:
            return self._forward(**kwargs)

    def _forward_faster(self, **kwargs):
        """
        In faster mode, we make following changes to exchange accuracy for speed:
        1. no conflict checking: 4.88 fps -> 6.56 fps
        2. smooth_conv9x9 ~ smooth_conv3x3 for different resolution
        3. last step no examine
        """
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]
        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone()
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)
                if (occupancys > 0.5).sum() == 0:
                    return None
                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)
                with torch.no_grad():
                    coords_accum = coords / stride
            elif torch.equal(resolution, self.resolutions[-1]):
                with torch.no_grad():
                    valid = F.interpolate((occupancys > self.balance_value).float(), size=(D, H, W), mode='trilinear', align_corners=True)
                occupancys = F.interpolate(occupancys.float(), size=(D, H, W), mode='trilinear', align_corners=True)
                is_boundary = valid == 0.5
            else:
                coords_accum *= 2
                with torch.no_grad():
                    valid = F.interpolate((occupancys > self.balance_value).float(), size=(D, H, W), mode='trilinear', align_corners=True)
                occupancys = F.interpolate(occupancys.float(), size=(D, H, W), mode='trilinear', align_corners=True)
                is_boundary = (valid > 0.0) & (valid < 1.0)
                with torch.no_grad():
                    if torch.equal(resolution, self.resolutions[1]):
                        is_boundary = (self.smooth_conv9x9(is_boundary.float()) > 0)[0, 0]
                    elif torch.equal(resolution, self.resolutions[2]):
                        is_boundary = (self.smooth_conv7x7(is_boundary.float()) > 0)[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                    coords_accum = coords_accum.long()
                    is_boundary[coords_accum[0, :, 2], coords_accum[0, :, 1], coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero(as_tuple=False).unsqueeze(0)
                    point_indices = point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W + point_coords[:, :, 0]
                    R, C, D, H, W = occupancys.shape
                    coords = point_coords * stride
                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = occupancys.reshape(R, C, D * H * W).scatter_(2, point_indices, occupancys_topk).view(R, C, D, H, W)
                with torch.no_grad():
                    voxels = coords / stride
                    coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)
        return occupancys[0, 0]

    def _forward(self, **kwargs):
        """
        output occupancy field would be:
        (bz, C, res, res)
        """
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]
        calculated = self.calculated.clone()
        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)
            if self.visualize:
                this_stage_coords = []
            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone()
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)
                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)
                with torch.no_grad():
                    coords_accum = coords / stride
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True
            else:
                coords_accum *= 2
                with torch.no_grad():
                    valid = F.interpolate((occupancys > self.balance_value).float(), size=(D, H, W), mode='trilinear', align_corners=True)
                occupancys = F.interpolate(occupancys.float(), size=(D, H, W), mode='trilinear', align_corners=True)
                is_boundary = (valid > 0.0) & (valid < 1.0)
                with torch.no_grad():
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        depth_res = resolution[2].item()
                        depth_index = torch.linspace(0, depth_res - 1, steps=depth_res).type_as(occupancys.device)
                        depth_index_max = torch.max((occupancys > self.balance_value) * (depth_index + 1), dim=-1, keepdim=True)[0] - 1
                        shadow = depth_index < depth_index_max
                        is_boundary[shadow] = False
                        is_boundary = is_boundary[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]
                    is_boundary[coords_accum[0, :, 2], coords_accum[0, :, 1], coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero(as_tuple=False).unsqueeze(0)
                    point_indices = point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W + point_coords[:, :, 0]
                    R, C, D, H, W = occupancys.shape
                    occupancys_interp = torch.gather(occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))
                    coords = point_coords * stride
                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                if self.visualize:
                    this_stage_coords.append(coords)
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = occupancys.reshape(R, C, D * H * W).scatter_(2, point_indices, occupancys_topk).view(R, C, D, H, W)
                with torch.no_grad():
                    conflicts = ((occupancys_interp - self.balance_value) * (occupancys_topk - self.balance_value) < 0)[0, 0]
                    if self.visualize:
                        self.plot(occupancys, coords, final_D, final_H, final_W)
                    voxels = coords / stride
                    coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True
                while conflicts.sum() > 0:
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        break
                    with torch.no_grad():
                        conflicts_coords = coords[0, conflicts, :]
                        if self.debug:
                            self.plot(occupancys, conflicts_coords.unsqueeze(0), final_D, final_H, final_W, title='conflicts')
                        conflicts_boundary = (conflicts_coords.int() + self.gird8_offsets.unsqueeze(1) * stride.int()).reshape(-1, 3).long().unique(dim=0)
                        conflicts_boundary[:, 0] = conflicts_boundary[:, 0].clamp(0, calculated.size(2) - 1)
                        conflicts_boundary[:, 1] = conflicts_boundary[:, 1].clamp(0, calculated.size(1) - 1)
                        conflicts_boundary[:, 2] = conflicts_boundary[:, 2].clamp(0, calculated.size(0) - 1)
                        coords = conflicts_boundary[calculated[conflicts_boundary[:, 2], conflicts_boundary[:, 1], conflicts_boundary[:, 0]] == False]
                        if self.debug:
                            self.plot(occupancys, coords.unsqueeze(0), final_D, final_H, final_W, title='coords')
                        coords = coords.unsqueeze(0)
                        point_coords = coords / stride
                        point_indices = point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W + point_coords[:, :, 0]
                        R, C, D, H, W = occupancys.shape
                        occupancys_interp = torch.gather(occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1))
                        coords = point_coords * stride
                    if coords.size(1) == 0:
                        break
                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    if self.visualize:
                        this_stage_coords.append(coords)
                    with torch.no_grad():
                        conflicts = ((occupancys_interp - self.balance_value) * (occupancys_topk - self.balance_value) < 0)[0, 0]
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    occupancys = occupancys.reshape(R, C, D * H * W).scatter_(2, point_indices, occupancys_topk).view(R, C, D, H, W)
                    with torch.no_grad():
                        voxels = coords / stride
                        coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)
                        calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True
                if self.visualize:
                    this_stage_coords = torch.cat(this_stage_coords, dim=1)
                    self.plot(occupancys, this_stage_coords, final_D, final_H, final_W)
        return occupancys[0, 0]

    def plot(self, occupancys, coords, final_D, final_H, final_W, title='', **kwargs):
        final = F.interpolate(occupancys.float(), size=(final_D, final_H, final_W), mode='trilinear', align_corners=True)
        x = coords[0, :, 0]
        y = coords[0, :, 1]
        z = coords[0, :, 2]
        plot_mask3D(final[0, 0], title, (x, y, z), **kwargs)

    def find_vertices(self, sdf, direction='front'):
        """
            - direction: "front" | "back" | "left" | "right"
        """
        resolution = sdf.size(2)
        if direction == 'front':
            pass
        elif direction == 'left':
            sdf = sdf.permute(2, 1, 0)
        elif direction == 'back':
            inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
            sdf = sdf[inv_idx, :, :]
        elif direction == 'right':
            inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
            sdf = sdf[:, :, inv_idx]
            sdf = sdf.permute(2, 1, 0)
        inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
        sdf = sdf[inv_idx, :, :]
        sdf_all = sdf.permute(2, 1, 0)
        grad_v = (sdf_all > 0.5) * torch.linspace(resolution, 1, steps=resolution)
        grad_c = torch.ones_like(sdf_all) * torch.linspace(0, resolution - 1, steps=resolution)
        max_v, max_c = grad_v.max(dim=2)
        shadow = grad_c > max_c.view(resolution, resolution, 1)
        keep = (sdf_all > 0.5) & ~shadow
        p1 = keep.nonzero(as_tuple=False).t()
        p2 = p1.clone()
        p2[2, :] = (p2[2, :] - 2).clamp(0, resolution)
        p3 = p1.clone()
        p3[1, :] = (p3[1, :] - 2).clamp(0, resolution)
        p4 = p1.clone()
        p4[0, :] = (p4[0, :] - 2).clamp(0, resolution)
        v1 = sdf_all[p1[0, :], p1[1, :], p1[2, :]]
        v2 = sdf_all[p2[0, :], p2[1, :], p2[2, :]]
        v3 = sdf_all[p3[0, :], p3[1, :], p3[2, :]]
        v4 = sdf_all[p4[0, :], p4[1, :], p4[2, :]]
        X = p1[0, :].long()
        Y = p1[1, :].long()
        Z = p2[2, :].float() * (0.5 - v1) / (v2 - v1) + p1[2, :].float() * (v2 - 0.5) / (v2 - v1)
        Z = Z.clamp(0, resolution)
        norm_z = v2 - v1
        norm_y = v3 - v1
        norm_x = v4 - v1
        norm = torch.stack([norm_x, norm_y, norm_z], dim=1)
        norm = norm / torch.norm(norm, p=2, dim=1, keepdim=True)
        return X, Y, Z, norm

    def render_normal(self, resolution, X, Y, Z, norm):
        image = torch.ones((1, 3, resolution, resolution), dtype=torch.float32)
        color = (norm + 1) / 2.0
        color = color.clamp(0, 1)
        image[0, :, Y, X] = color.t()
        return image

    def display(self, sdf):
        X, Y, Z, norm = self.find_vertices(sdf, direction='front')
        image1 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction='left')
        image2 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction='right')
        image3 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction='back')
        image4 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        image = torch.cat([image1, image2, image3, image4], axis=3)
        image = image.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0
        return np.uint8(image)

    def export_mesh(self, occupancys):
        final = occupancys[1:, 1:, 1:].contiguous()
        if final.shape[0] > 256:
            occu_arr = final.detach().cpu().numpy()
            vertices, triangles = mcubes.marching_cubes(occu_arr, self.balance_value)
            verts = torch.as_tensor(vertices[:, [2, 1, 0]])
            faces = torch.as_tensor(triangles.astype(np.long), dtype=torch.long)[:, [0, 2, 1]]
        else:
            torch.cuda.empty_cache()
            vertices, triangles = voxelgrids_to_trianglemeshes(final.unsqueeze(0))
            verts = vertices[0][:, [2, 1, 0]].cpu()
            faces = triangles[0][:, [0, 2, 1]].cpu()
        return verts, faces


class SmoothConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size for smooth_conv must be odd: {3, 5, ...}'
        self.padding = (kernel_size - 1) // 2
        weight = torch.ones((in_channels, out_channels, kernel_size, kernel_size), dtype=torch.float32) / kernel_size ** 2
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=self.padding)


class GMoF(torch.nn.Module):

    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        dist = torch.div(residual, residual + self.rho ** 2)
        return self.rho ** 2 * dist


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, bias=False, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes * groups, out_planes * groups, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes * groups, planes * groups, kernel_size=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes * groups, planes * groups, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes * groups, planes * self.expansion * groups, kernel_size=1, bias=False, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion * groups, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        return x2


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def transform_mat(R: 'Tensor', t: 'Tensor') ->Tensor:
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats: 'Tensor', joints: 'Tensor', parents: 'Tensor', dtype=torch.float32) ->Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def batch_rodrigues(rot_vecs: 'Tensor', epsilon: 'float'=1e-08) ->Tensor:
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-08, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def blend_shapes(betas: 'Tensor', shape_disps: 'Tensor') ->Tensor:
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor: 'Tensor', vertices: 'Tensor') ->Tensor:
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(betas: 'Tensor', pose: 'Tensor', v_template: 'Tensor', shapedirs: 'Tensor', posedirs: 'Tensor', J_regressor: 'Tensor', parents: 'Tensor', lbs_weights: 'Tensor', pose2rot: 'bool'=True, return_transformation: 'bool'=False) ->Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    if return_transformation:
        return verts, J_transformed, A, T
    return verts, J_transformed


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array: 'Union[Array, Tensor]', dtype=torch.float32) ->Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'jaw', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_thumb', 'right_thumb', 'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe']
    LEAF_NAMES = ['head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe']
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, model_path, h36m_jregressor, gender='neutral', dtype=torch.float32, num_joints=29):
        """ SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        """
        super(SMPL_layer, self).__init__()
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9
        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))
        self.gender = gender
        self.dtype = dtype
        self.faces = self.smpl_data.f
        """ Register Buffer """
        self.register_buffer('faces_tensor', to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))
        self.register_buffer('shapedirs', to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        self.register_buffer('J_regressor_h36m', to_tensor(to_np(h36m_jregressor), dtype=dtype))
        self.num_joints = num_joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:self.NUM_JOINTS + 1] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map', self._parents_to_children(parents))
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1
        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')
        return children

    def forward(self, pose_axis_angle, betas, global_orient, transl=None, return_verts=True):
        """ Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        """
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle
        pose2rot = True
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.J_regressor_h36m, self.parents, self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)
        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
        output = ModelOutput(vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output

    def hybrik(self, pose_skeleton, betas, phis, global_orient, transl=None, return_verts=True, leaf_thetas=None):
        """ Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        """
        batch_size = pose_skeleton.shape[0]
        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)
        vertices, new_joints, rot_mats, joints_from_verts = hybrik(betas, global_orient, pose_skeleton, phis, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map, self.lbs_weights, dtype=self.dtype, train=self.training, leaf_thetas=leaf_thetas)
        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)
        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        output = ModelOutput(vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output


def norm_heatmap(norm_type, heatmap):
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


def update_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


class HybrIKBaseSMPLCam(nn.Module):

    def __init__(self, cfg_file, smpl_path, data_path, norm_layer=nn.BatchNorm2d):
        super(HybrIKBaseSMPLCam, self).__init__()
        cfg = update_config(cfg_file)['MODEL']
        self.deconv_dim = cfg['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = cfg['NUM_JOINTS']
        self.norm_type = cfg['POST']['NORM_TYPE']
        self.depth_dim = cfg['EXTRA']['DEPTH_DIM']
        self.height_dim = cfg['HEATMAP_SIZE'][0]
        self.width_dim = cfg['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        backbone = ResNet
        self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}")
        import torchvision.models as tm
        if cfg['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif cfg['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif cfg['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif cfg['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        h36m_jregressor = np.load(os.path.join(data_path, 'J_regressor_h36m.npy'))
        self.smpl = SMPL_layer(smpl_path, h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.leaf_pairs = (0, 1), (3, 4)
        self.root_idx_smpl = 0
        init_shape = np.load(os.path.join(data_path, 'h36m_mean_beta.npy'))
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        init_cam = torch.tensor([0.9, 0, 0])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)
        self.deccam = nn.Linear(1024, 3)
        self.focal_length = cfg['FOCAL_LENGTH']
        self.input_size = 256.0

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]
        if shift:
            pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]
        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)
        return pred_jts

    def flip_xyz_coord(self, pred_jts, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]
        pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]
        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)
        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def forward(self, x, flip_item=None, flip_output=False, gt_uvd=None, gt_uvd_weight=None, **kwargs):
        batch_size = x.shape[0]
        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)
        out = out.reshape((out.shape[0], self.num_joints, -1))
        maxvals, _ = torch.max(out, dim=2, keepdim=True)
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape
        heatmaps = out / out.sum(dim=2, keepdim=True)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x0 = heatmaps.sum((2, 3))
        hm_y0 = heatmaps.sum((2, 4))
        hm_z0 = heatmaps.sum((3, 4))
        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
        hm_x = hm_x0 * range_tensor
        hm_y = hm_y0 * range_tensor
        hm_z = hm_z0 * range_tensor
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)
        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        xc = x0
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-09)
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
        pred_xyz_jts_29_meter = pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * 2.2 + camDepth) - camTrans
        pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / 2.2
        camera_root = pred_xyz_jts_29[:, [0]] * 2.2
        camera_root[:, :, :2] += camTrans
        camera_root[:, :, [2]] += camDepth
        if not self.training:
            pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        if flip_item is not None:
            assert flip_output is not None
            pred_xyz_jts_29_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item
        if flip_output:
            pred_xyz_jts_29 = self.flip_xyz_coord(pred_xyz_jts_29, flatten=False)
        if flip_output and flip_item is not None:
            pred_xyz_jts_29 = (pred_xyz_jts_29 + pred_xyz_jts_29_orig.reshape(batch_size, 29, 3)) / 2
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        if flip_output:
            pred_phi = self.flip_phi(pred_phi)
        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * 2.2, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / 2
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24, 3, 3)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72) / 2
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        transl = pred_xyz_jts_29[:, 0, :] * 2.2 - pred_xyz_jts_17[:, 0, :] * 2.2
        transl[:, :2] += camTrans[:, 0]
        transl[:, 2] += camDepth[:, 0, 0]
        new_cam = torch.zeros_like(transl)
        new_cam[:, 1:] = transl[:, :2]
        new_cam[:, 0] = self.focal_length / (self.input_size * transl[:, 2] + 1e-09)
        output = dict(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1), pred_xyz_jts_29=pred_xyz_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17_flat, pred_vertices=pred_vertices, maxvals=maxvals, cam_scale=camScale[:, 0], cam_trans=camTrans[:, 0], cam_root=camera_root, pred_camera=new_cam, transl=transl)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, opt):
        super(ConvBlock, self).__init__()
        [k, s, d, p] = opt.conv3x3
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), k, s, d, p)
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), k, s, d, p)
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), k, s, d, p)
        if opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(self.bn4, nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, opt):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.opt = opt
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, self.opt))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, self.opt))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, self.opt))
        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, self.opt))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):

    def __init__(self, opt, num_modules, in_dim):
        super(HGFilter, self).__init__()
        self.num_modules = num_modules
        self.opt = opt
        [k, s, d, p] = self.opt.conv1
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=k, stride=s, dilation=d, padding=p)
        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt)
        else:
            raise NameError('Unknown Fan Filter setting!')
        self.conv3 = ConvBlock(128, 128, self.opt)
        self.conv4 = ConvBlock(128, 256, self.opt)
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, opt.hourglass_dim, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs


class FuseHGFilter(nn.Module):

    def __init__(self, opt, num_modules, in_dim):
        super(FuseHGFilter, self).__init__()
        self.num_modules = num_modules
        self.opt = opt
        [k, s, d, p] = self.opt.conv1
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=k, stride=s, dilation=d, padding=p)
        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = ConvBlock(64, 128, self.opt)
        self.down_conv2 = nn.Conv2d(128, 96, kernel_size=3, stride=2, padding=1)
        dim = 96 + 32
        self.conv3 = ConvBlock(dim, dim, self.opt)
        self.conv4 = ConvBlock(dim, 256, self.opt)
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
            hourglass_dim = 256
            self.add_module('l' + str(hg_module), nn.Conv2d(256, hourglass_dim, kernel_size=1, stride=1, padding=0))
            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(hourglass_dim, 256, kernel_size=1, stride=1, padding=0))
        self.up_conv = nn.ConvTranspose2d(hourglass_dim, 64, kernel_size=2, stride=2)

    def forward(self, x, plane):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        x = self.conv2(x)
        x = self.down_conv2(x)
        x = torch.cat([x, plane], 1)
        x = self.conv3(x)
        x = self.conv4(x)
        previous = x
        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        out = self.up_conv(outputs[-1])
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class IRB(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)


class PoolingAttention(nn.Module):

    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([(t * t) for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for pool_ratio, l in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop, ksize=3)

    def forward(self, x, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, f'img_size {img_size} should be divided by patch_size {patch_size}.'
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class PyramidPoolingTransformer(nn.Module):

    def __init__(self, img_size=512, patch_size=2, in_chans=3, num_classes=1000, embed_dims=[64, 256, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=[2, 2, 9, 3]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, kernel_size=7, in_chans=in_chans, embed_dim=embed_dims[0], overlap=True)
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], overlap=True)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], overlap=True)
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3], overlap=True)
        self.d_convs1 = nn.ModuleList([nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pool_ratios[0]])
        self.d_convs2 = nn.ModuleList([nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pool_ratios[1]])
        self.d_convs3 = nn.ModuleList([nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pool_ratios[2]])
        self.d_convs4 = nn.ModuleList([nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pool_ratios[3]])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        ksize = 3
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[0]) for i in range(depths[0])])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[1]) for i in range(depths[1])])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[2]) for i in range(depths[2])])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[3]) for i in range(depths[3])])
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embed1(x)
        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, (H, W) = self.patch_embed2(x)
        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x

    def forward_features_for_fpn(self, x):
        outs = []
        B = x.shape[0]
        x, (H, W) = self.patch_embed1(x)
        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        x, (H, W) = self.patch_embed2(x)
        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        x, (H, W) = self.patch_embed3(x)
        for idx, blk in enumerate(self.block3):
            x = blk(x, H, W, self.d_convs3)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        x, (H, W) = self.patch_embed4(x)
        for idx, blk in enumerate(self.block4):
            x = blk(x, H, W, self.d_convs4)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def forward_for_fpn(self, x):
        return self.forward_features_for_fpn(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, last=True)]
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        if opt.use_tanh:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class PreNorm(nn.Module):

    def __init__(self, dim: 'int', fn: 'nn.Module') ->None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: 'torch.FloatTensor', **kwargs) ->torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int') ->None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim))

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim: 'int', heads: 'int'=8, dim_head: 'int'=64) ->None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):

    def __init__(self, dim: 'int', heads: 'int'=8, dim_head: 'int'=64) ->None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.multi_head_attention = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head))

    def forward(self, x: 'torch.FloatTensor', q_x: 'torch.FloatTensor') ->torch.FloatTensor:
        q_in = self.multi_head_attention(q_x) + q_x
        q_in = self.norm(q_in)
        q = rearrange(self.to_q(q_in), 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), q_in


class Transformer(nn.Module):

    def __init__(self, dim: 'int', depth: 'int', heads: 'int', dim_head: 'int', mlp_dim: 'int') ->None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)), PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class CrossTransformer(nn.Module):

    def __init__(self, dim: 'int', depth: 'int', heads: 'int', dim_head: 'int', mlp_dim: 'int') ->None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([CrossAttention(dim, heads=heads, dim_head=dim_head), PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.FloatTensor', q_x: 'torch.FloatTensor') ->torch.FloatTensor:
        encoder_output = x
        for attn, ff in self.layers:
            x, q_in = attn(encoder_output, q_x)
            x = x + q_in
            x = ff(x) + x
            q_x = x
        return self.norm(q_x)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)


class ViTEncoder(nn.Module):

    def __init__(self, image_size: 'Union[Tuple[int, int], int]', patch_size: 'Union[Tuple[int, int], int]', dim: 'int', depth: 'int', heads: 'int', mlp_dim: 'int', channels: 'int'=3, dim_head: 'int'=64) ->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        self.num_patches = image_height // patch_height * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size), Rearrange('b c h w -> b (h w) c'))
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.apply(init_weights)

    def forward(self, img: 'torch.FloatTensor') ->torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)
        return x


class ViTDecoder(nn.Module):

    def __init__(self, image_size: 'Union[Tuple[int, int], int]', patch_size: 'Union[Tuple[int, int], int]', dim: 'int', depth: 'int', heads: 'int', mlp_dim: 'int', channels: 'int'=32, dim_head: 'int'=64) ->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        self.num_patches = image_height // patch_height * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=image_height // patch_height), nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4))
        self.to_origin_pixiel = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=image_height // patch_height))
        self.apply(init_weights)

    def forward(self, token: 'torch.FloatTensor') ->torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        self.seq = self.to_origin_pixiel(x)
        x = self.to_pixel(x)
        return x

    def get_last_layer(self) ->nn.Parameter:
        return self.to_pixel[-1].weight


class CrossAttDecoder(nn.Module):

    def __init__(self, image_size: 'Union[Tuple[int, int], int]', patch_size: 'Union[Tuple[int, int], int]', dim: 'int', depth: 'int', heads: 'int', mlp_dim: 'int', channels: 'int'=64, dim_head: 'int'=64) ->None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        self.num_patches = image_height // patch_height * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim)
        self.query = nn.Parameter(torch.randn(1024, dim), requires_grad=True)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=image_height // patch_height), nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4))
        self.to_origin_pixiel = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=image_height // patch_height))
        self.apply(init_weights)

    def forward(self, token: 'torch.FloatTensor') ->torch.FloatTensor:
        batch_size = token.shape[0]
        query = self.query.repeat(batch_size, 1, 1) + self.de_pos_embedding
        x = token + self.de_pos_embedding
        x = self.transformer(x, query)
        self.seq = self.to_origin_pixiel(x)
        x = self.to_pixel(x)
        return x

    def get_last_layer(self) ->nn.Parameter:
        return self.to_pixel[-1].weight


class BaseQuantizer(nn.Module):

    def __init__(self, embed_dim: 'int', n_embed: 'int', straight_through: 'bool'=True, use_norm: 'bool'=True, use_residual: 'bool'=False, num_quantizers: 'Optional[int]'=None) ->None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.use_residual = use_residual
        self.num_quantizers = num_quantizers
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()

    def quantize(self, z: 'torch.FloatTensor') ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass

    def forward(self, z: 'torch.FloatTensor') ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()
            losses = []
            encoding_indices = []
            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)
                encoding_indices.append(indices)
                losses.append(loss)
            losses, encoding_indices = map(partial(torch.stack, dim=-1), (losses, encoding_indices))
            loss = losses.mean()
        if self.straight_through:
            z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):

    def __init__(self, embed_dim: 'int', n_embed: 'int', beta: 'float'=0.25, use_norm: 'bool'=True, use_residual: 'bool'=False, num_quantizers: 'Optional[int]'=None, **kwargs) ->None:
        super().__init__(embed_dim, n_embed, True, use_norm, use_residual, num_quantizers)
        self.beta = beta

    def quantize(self, z: 'torch.FloatTensor') ->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + torch.sum(embedding_norm ** 2, dim=1) - 2 * torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm) ** 2) + torch.mean((z_qnorm - z_norm.detach()) ** 2)
        return z_qnorm, loss, encoding_indices


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = F.normalize(x)
        return x


class LocalAffine(nn.Module):

    def __init__(self, num_points, batch_size=1, edges=None):
        """
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        """
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1))
        self.b = nn.Parameter(torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batch_size, num_points, 1, 1))
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        """
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix, 
        """
        if self.edges is None:
            raise Exception('edges cannot be none when calculate stiff')
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2) ** 2
        w_rigid = (torch.linalg.det(self.A) - 1.0) ** 2
        return w_diff, w_rigid

    def forward(self, x, return_stiff=False):
        """
            x should have shape of B * N * 3
        """
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness, rigid = self.stiffness()
            return out_x, stiffness, rigid
        else:
            return out_x


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VoxelizationFunction(Function):
    """
    Definition of differentiable voxelization function
    Currently implemented only for cuda Tensors
    """

    @staticmethod
    def forward(ctx, smpl_vertices, smpl_face_center, smpl_face_normal, smpl_vertex_code, smpl_face_code, smpl_tetrahedrons, volume_res, sigma, smooth_kernel_size):
        """
        forward pass
        Output format: (batch_size, z_dims, y_dims, x_dims, channel_num) 
        """
        assert smpl_vertices.size()[1] == smpl_vertex_code.size()[1]
        assert smpl_face_center.size()[1] == smpl_face_normal.size()[1]
        assert smpl_face_center.size()[1] == smpl_face_code.size()[1]
        ctx.batch_size = smpl_vertices.size()[0]
        ctx.volume_res = volume_res
        ctx.sigma = sigma
        ctx.smooth_kernel_size = smooth_kernel_size
        ctx.smpl_vertex_num = smpl_vertices.size()[1]
        ctx.device = smpl_vertices.device
        smpl_vertices = smpl_vertices.contiguous()
        smpl_face_center = smpl_face_center.contiguous()
        smpl_face_normal = smpl_face_normal.contiguous()
        smpl_vertex_code = smpl_vertex_code.contiguous()
        smpl_face_code = smpl_face_code.contiguous()
        smpl_tetrahedrons = smpl_tetrahedrons.contiguous()
        occ_volume = torch.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res).fill_(0.0)
        semantic_volume = torch.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res, 3).fill_(0.0)
        weight_sum_volume = torch.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res).fill_(0.001)
        occ_volume, semantic_volume, weight_sum_volume = voxelize_cuda.forward_semantic_voxelization(smpl_vertices, smpl_vertex_code, smpl_tetrahedrons, occ_volume, semantic_volume, weight_sum_volume, sigma)
        return semantic_volume


class Voxelization(nn.Module):
    """
    Wrapper around the autograd function VoxelizationFunction
    """

    def __init__(self, smpl_vertex_code, smpl_face_code, smpl_face_indices, smpl_tetraderon_indices, volume_res, sigma, smooth_kernel_size, batch_size, device):
        super(Voxelization, self).__init__()
        assert len(smpl_face_indices.shape) == 2
        assert len(smpl_tetraderon_indices.shape) == 2
        assert smpl_face_indices.shape[1] == 3
        assert smpl_tetraderon_indices.shape[1] == 4
        self.volume_res = volume_res
        self.sigma = sigma
        self.smooth_kernel_size = smooth_kernel_size
        self.batch_size = batch_size
        self.device = device
        self.smpl_vertex_code = smpl_vertex_code
        self.smpl_face_code = smpl_face_code
        self.smpl_face_indices = smpl_face_indices
        self.smpl_tetraderon_indices = smpl_tetraderon_indices

    def update_param(self, batch_size, smpl_tetra):
        self.batch_size = batch_size
        self.smpl_tetraderon_indices = smpl_tetra
        smpl_vertex_code_batch = np.tile(self.smpl_vertex_code, (self.batch_size, 1, 1))
        smpl_face_code_batch = np.tile(self.smpl_face_code, (self.batch_size, 1, 1))
        smpl_face_indices_batch = np.tile(self.smpl_face_indices, (self.batch_size, 1, 1))
        smpl_tetraderon_indices_batch = np.tile(self.smpl_tetraderon_indices, (self.batch_size, 1, 1))
        smpl_vertex_code_batch = torch.from_numpy(smpl_vertex_code_batch).contiguous()
        smpl_face_code_batch = torch.from_numpy(smpl_face_code_batch).contiguous()
        smpl_face_indices_batch = torch.from_numpy(smpl_face_indices_batch).contiguous()
        smpl_tetraderon_indices_batch = torch.from_numpy(smpl_tetraderon_indices_batch).contiguous()
        self.register_buffer('smpl_vertex_code_batch', smpl_vertex_code_batch)
        self.register_buffer('smpl_face_code_batch', smpl_face_code_batch)
        self.register_buffer('smpl_face_indices_batch', smpl_face_indices_batch)
        self.register_buffer('smpl_tetraderon_indices_batch', smpl_tetraderon_indices_batch)

    def forward(self, smpl_vertices):
        """
        Generate semantic volumes from SMPL vertices
        """
        assert smpl_vertices.size()[0] == self.batch_size
        self.check_input(smpl_vertices)
        smpl_faces = self.vertices_to_faces(smpl_vertices)
        smpl_tetrahedrons = self.vertices_to_tetrahedrons(smpl_vertices)
        smpl_face_center = self.calc_face_centers(smpl_faces)
        smpl_face_normal = self.calc_face_normals(smpl_faces)
        smpl_surface_vertex_num = self.smpl_vertex_code_batch.size()[1]
        smpl_vertices_surface = smpl_vertices[:, :smpl_surface_vertex_num, :]
        vol = VoxelizationFunction.apply(smpl_vertices_surface, smpl_face_center, smpl_face_normal, self.smpl_vertex_code_batch, self.smpl_face_code_batch, smpl_tetrahedrons, self.volume_res, self.sigma, self.smooth_kernel_size)
        return vol.permute((0, 4, 1, 2, 3))

    def vertices_to_faces(self, vertices):
        assert vertices.ndimension() == 3
        bs, nv = vertices.shape[:2]
        device = vertices.device
        face = self.smpl_face_indices_batch + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[face.long()]

    def vertices_to_tetrahedrons(self, vertices):
        assert vertices.ndimension() == 3
        bs, nv = vertices.shape[:2]
        device = vertices.device
        tets = self.smpl_tetraderon_indices_batch + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[tets.long()]

    def calc_face_centers(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_centers = (face_verts[:, :, 0, :] + face_verts[:, :, 1, :] + face_verts[:, :, 2, :]) / 3.0
        face_centers = face_centers.reshape((bs, nf, 3))
        return face_centers

    def calc_face_normals(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_verts = face_verts.reshape((bs * nf, 3, 3))
        v10 = face_verts[:, 0] - face_verts[:, 1]
        v12 = face_verts[:, 2] - face_verts[:, 1]
        normals = F.normalize(torch.cross(v10, v12), eps=1e-05)
        normals = normals.reshape((bs, nf, 3))
        return normals

    def check_input(self, x):
        if x.device == 'cpu':
            raise TypeError('Voxelization module supports only cuda tensors')
        if x.type() != 'torch.cuda.FloatTensor':
            raise TypeError('Voxelization module supports only float32 tensors')


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i]), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3), nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


logger = logging.getLogger(__name__)


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.final_layer = nn.Conv2d(in_channels=pre_stage_channels[0], out_channels=cfg['MODEL']['NUM_JOINTS'], kernel_size=extra['FINAL_CONV_KERNEL'], stride=1, padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        self.upsample_stage_2 = self._make_upsample_layer(1, num_channel=self.stage2_cfg['NUM_CHANNELS'][-1])
        self.upsample_stage_3 = self._make_upsample_layer(2, num_channel=self.stage3_cfg['NUM_CHANNELS'][-1])
        self.upsample_stage_4 = self._make_upsample_layer(3, num_channel=self.stage4_cfg['NUM_CHANNELS'][-1])

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _make_upsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernel_size, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_channel, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x1 = self.upsample_stage_2(x[1])
        x2 = self.upsample_stage_3(x[2])
        x3 = self.upsample_stage_4(x[3])
        x = torch.cat([x[0], x1, x2, x3], 1)
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


class HRNet(nn.Module):

    def __init__(self, arch, pretrained=True):
        super(HRNet, self).__init__()
        self.m = timm.create_model(arch, pretrained=pretrained)

    def forward(self, x):
        return self.m.forward_features(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), norm_layer(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CoAttention(nn.Module):

    def __init__(self, n_channel, final_conv='simple'):
        super(CoAttention, self).__init__()
        self.linear_e = nn.Linear(n_channel, n_channel, bias=False)
        self.channel = n_channel
        self.gate = nn.Conv2d(n_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.softmax = nn.Sigmoid()
        if final_conv.startswith('double'):
            kernel_size = int(final_conv[-1])
            conv = conv1x1 if kernel_size == 1 else conv3x3
            self.final_conv_1 = nn.Sequential(conv(n_channel * 2, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True), conv(n_channel, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True))
            self.final_conv_2 = nn.Sequential(conv(n_channel * 2, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True), conv(n_channel, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True))
        elif final_conv.startswith('single'):
            kernel_size = int(final_conv[-1])
            conv = conv1x1 if kernel_size == 1 else conv3x3
            self.final_conv_1 = nn.Sequential(conv(n_channel * 2, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True))
            self.final_conv_2 = nn.Sequential(conv(n_channel * 2, n_channel), nn.BatchNorm2d(n_channel), nn.ReLU(inplace=True))
        elif final_conv == 'simple':
            self.final_conv_1 = conv1x1(n_channel * 2, n_channel)
            self.final_conv_2 = conv1x1(n_channel * 2, n_channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_1, input_2):
        """
        input_1: [N, C, H, W]
        input_2: [N, C, H, W]
        """
        b, c, h, w = input_1.shape
        exemplar, query = input_1, input_2
        exemplar_flat = exemplar.reshape(-1, c, h * w)
        query_flat = query.reshape(-1, c, h * w)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1)
        exemplar_att = torch.bmm(query_flat, B)
        input1_att = exemplar_att.reshape(-1, c, h, w)
        input2_att = query_att.reshape(-1, c, h, w)
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1 = self.final_conv_1(input1_att)
        input2 = self.final_conv_2(input2_att)
        return input1, input2


class KeypointAttention(nn.Module):

    def __init__(self, use_conv=False, in_channels=(256, 64), out_channels=(256, 64), act='softmax', use_scale=False):
        super(KeypointAttention, self).__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.use_scale = use_scale
        if use_conv:
            self.conv1x1_pose = nn.Conv1d(in_channels[0], out_channels[0], kernel_size=1)
            self.conv1x1_shape_cam = nn.Conv1d(in_channels[1], out_channels[1], kernel_size=1)

    def forward(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape
        if self.use_scale:
            scale = 1.0 / np.sqrt(height * width)
            heatmaps = heatmaps * scale
        if self.act == 'softmax':
            normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        elif self.act == 'sigmoid':
            normalized_heatmap = torch.sigmoid(heatmaps.reshape(batch_size, num_joints, -1))
        features = features.reshape(batch_size, -1, height * width)
        attended_features = torch.matmul(normalized_heatmap, features.transpose(2, 1))
        attended_features = attended_features.transpose(2, 1)
        if self.use_conv:
            if attended_features.shape[1] == self.in_channels[0]:
                attended_features = self.conv1x1_pose(attended_features)
            else:
                attended_features = self.conv1x1_shape_cam(attended_features)
        return attended_features


class LocallyConnected2d(nn.Module):

    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size[0], output_size[1]), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


def get_coord_maps(size=56):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)
    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)
    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)
    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)
    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)
    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)
    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)
    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out


def get_heatmap_preds(batch_heatmaps, normalize_keypoints=True):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = torch.floor(preds[:, :, 1] / width)
    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()
    preds *= pred_mask
    if normalize_keypoints:
        preds[:, :, 0] = preds[:, :, 0] / (width - 1) * 2 - 1
        preds[:, :, 1] = preds[:, :, 1] / (height - 1) * 2 - 1
    return preds, maxvals


def get_smpl_neighbor_triplets():
    return [[0, 1, 2], [1, 4, 0], [2, 0, 5], [3, 0, 6], [4, 7, 1], [5, 2, 8], [6, 3, 9], [7, 10, 4], [8, 5, 11], [9, 13, 14], [10, 7, 4], [11, 8, 5], [12, 9, 15], [13, 16, 9], [14, 9, 17], [15, 9, 12], [16, 18, 13], [17, 14, 19], [18, 20, 16], [19, 17, 21], [20, 22, 18], [21, 19, 23], [22, 20, 18], [23, 19, 21]]


def interpolate(feat, uv):
    """

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    if uv.shape[-1] != 2:
        uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    if int(torch.__version__.split('.')[1]) < 4:
        samples = torch.nn.functional.grid_sample(feat, uv)
    else:
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples[:, :, :, 0]


def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)


def softargmax2d(heatmaps, temperature=None, normalize_keypoints=True):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, height, width = heatmaps.shape
    x = torch.arange(0, width, device=device, dtype=dtype).reshape(1, 1, 1, width).expand(batch_size, -1, height, -1)
    y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(heatmaps.reshape(batch_size, num_channels, -1), temperature=temperature.reshape(1, -1, 1), dim=-1)
    keypoints = (normalized_heatmap.reshape(batch_size, -1, 1, height * width) * points.reshape(batch_size, 1, 2, -1)).sum(dim=-1)
    if normalize_keypoints:
        keypoints[:, :, 0] = keypoints[:, :, 0] / (width - 1) * 2 - 1
        keypoints[:, :, 1] = keypoints[:, :, 1] / (height - 1) * 2 - 1
    return keypoints, normalized_heatmap.reshape(batch_size, -1, height, width)


class PareHead(nn.Module):

    def __init__(self, num_joints, num_input_features, softmax_temp=1.0, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), num_camera_params=3, num_features_smpl=64, final_conv_kernel=1, iterative_regression=False, iter_residual=False, num_iterations=3, shape_input_type='feats', pose_input_type='feats', pose_mlp_num_layers=1, shape_mlp_num_layers=1, pose_mlp_hidden_size=256, shape_mlp_hidden_size=256, use_keypoint_features_for_smpl_regression=False, use_heatmaps='', use_keypoint_attention=False, use_postconv_keypoint_attention=False, keypoint_attention_act='softmax', use_scale_keypoint_attention=False, use_branch_nonlocal=None, use_final_nonlocal=None, backbone='resnet', use_hmr_regression=False, use_coattention=False, num_coattention_iter=1, coattention_conv='simple', use_upsampling=False, use_soft_attention=False, num_branch_iteration=0, branch_deeper=False, use_resnet_conv_hrnet=False, use_position_encodings=None, use_mean_camshape=False, use_mean_pose=False, init_xavier=False):
        super(PareHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = False
        self.use_heatmaps = use_heatmaps
        self.num_iterations = num_iterations
        self.use_final_nonlocal = use_final_nonlocal
        self.use_branch_nonlocal = use_branch_nonlocal
        self.use_hmr_regression = use_hmr_regression
        self.use_coattention = use_coattention
        self.num_coattention_iter = num_coattention_iter
        self.coattention_conv = coattention_conv
        self.use_soft_attention = use_soft_attention
        self.num_branch_iteration = num_branch_iteration
        self.iter_residual = iter_residual
        self.iterative_regression = iterative_regression
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size
        self.use_keypoint_attention = use_keypoint_attention
        self.use_keypoint_features_for_smpl_regression = use_keypoint_features_for_smpl_regression
        self.use_position_encodings = use_position_encodings
        self.use_mean_camshape = use_mean_camshape
        self.use_mean_pose = use_mean_pose
        self.num_input_features = num_input_features
        if use_soft_attention:
            self.use_keypoint_features_for_smpl_regression = True
            self.use_hmr_regression = True
            self.use_coattention = False
            logger.warning('Coattention cannot be used together with soft attention')
            logger.warning('Overriding use_coattention=False')
        if use_coattention:
            self.use_keypoint_features_for_smpl_regression = False
            logger.warning('"use_keypoint_features_for_smpl_regression" cannot be used together with co-attention')
            logger.warning('Overriding "use_keypoint_features_for_smpl_regression"=False')
        if use_hmr_regression:
            self.iterative_regression = False
            logger.warning('iterative_regression cannot be used together with hmr regression')
        if self.use_heatmaps in ['part_segm', 'attention']:
            logger.info('"Keypoint Attention" should be activated to be able to use part segmentation')
            logger.info('Overriding use_keypoint_attention')
            self.use_keypoint_attention = True
        assert num_iterations > 0, '"num_iterations" should be greater than 0.'
        if use_position_encodings:
            assert backbone.startswith('hrnet'), 'backbone should be hrnet to use position encodings'
            self.register_buffer('pos_enc', get_coord_maps(size=56))
            num_input_features += 2
            self.num_input_features = num_input_features
        if backbone.startswith('hrnet'):
            if use_resnet_conv_hrnet:
                logger.info('Using resnet block for keypoint and smpl conv layers...')
                self.keypoint_deconv_layers = self._make_res_conv_layers(input_channels=self.num_input_features, num_channels=num_deconv_filters[-1], num_basic_blocks=num_deconv_layers)
                self.num_input_features = num_input_features
                self.smpl_deconv_layers = self._make_res_conv_layers(input_channels=self.num_input_features, num_channels=num_deconv_filters[-1], num_basic_blocks=num_deconv_layers)
            else:
                self.keypoint_deconv_layers = self._make_conv_layer(num_deconv_layers, num_deconv_filters, (3,) * num_deconv_layers)
                self.num_input_features = num_input_features
                self.smpl_deconv_layers = self._make_conv_layer(num_deconv_layers, num_deconv_filters, (3,) * num_deconv_layers)
        else:
            conv_fn = self._make_upsample_layer if use_upsampling else self._make_deconv_layer
            if use_upsampling:
                logger.info('Upsampling is active to increase spatial dimension')
                logger.info(f'Upsampling conv kernels: {num_deconv_kernels}')
            self.keypoint_deconv_layers = conv_fn(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
            self.num_input_features = num_input_features
            self.smpl_deconv_layers = conv_fn(num_deconv_layers, num_deconv_filters, num_deconv_kernels)
        pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        shape_mlp_inp_dim = num_joints * smpl_final_dim
        if self.use_soft_attention:
            logger.info('Soft attention (Stefan & Otmar 3DV) is active')
            self.keypoint_final_layer = nn.Sequential(conv3x3(num_deconv_filters[-1], 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True), conv1x1(256, num_joints + 1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints))
            soft_att_feature_size = smpl_final_dim
            self.smpl_final_layer = nn.Sequential(conv3x3(num_deconv_filters[-1], 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True), conv1x1(256, soft_att_feature_size))
        else:
            self.keypoint_final_layer = nn.Conv2d(in_channels=num_deconv_filters[-1], out_channels=num_joints + 1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints, kernel_size=final_conv_kernel, stride=1, padding=1 if final_conv_kernel == 3 else 0)
            self.smpl_final_layer = nn.Conv2d(in_channels=num_deconv_filters[-1], out_channels=smpl_final_dim, kernel_size=final_conv_kernel, stride=1, padding=1 if final_conv_kernel == 3 else 0)
        self.register_buffer('temperature', torch.tensor(softmax_temp))
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        if self.iterative_regression:
            input_type_dim = {'feats': 0, 'neighbor_pose_feats': 2 * 256, 'all_pose': 24 * 6, 'self_pose': 6, 'neighbor_pose': 2 * 6, 'shape': 10, 'cam': num_camera_params}
            assert 'feats' in shape_input_type, '"feats" should be the default value'
            assert 'feats' in pose_input_type, '"feats" should be the default value'
            self.shape_input_type = shape_input_type.split('.')
            self.pose_input_type = pose_input_type.split('.')
            pose_mlp_inp_dim = pose_mlp_inp_dim + sum([input_type_dim[x] for x in self.pose_input_type])
            shape_mlp_inp_dim = shape_mlp_inp_dim + sum([input_type_dim[x] for x in self.shape_input_type])
            logger.debug(f'Shape MLP takes "{self.shape_input_type}" as input, input dim: {shape_mlp_inp_dim}')
            logger.debug(f'Pose MLP takes "{self.pose_input_type}" as input, input dim: {pose_mlp_inp_dim}')
        self.pose_mlp_inp_dim = pose_mlp_inp_dim
        self.shape_mlp_inp_dim = shape_mlp_inp_dim
        if self.use_hmr_regression:
            logger.info(f'HMR regression is active...')
            self.fc1 = nn.Linear(num_joints * smpl_final_dim + num_joints * 6 + 10 + num_camera_params, 1024)
            self.drop1 = nn.Dropout()
            self.fc2 = nn.Linear(1024, 1024)
            self.drop2 = nn.Dropout()
            self.decpose = nn.Linear(1024, num_joints * 6)
            self.decshape = nn.Linear(1024, 10)
            self.deccam = nn.Linear(1024, num_camera_params)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        else:
            self.shape_mlp = self._get_shape_mlp(output_size=10)
            self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)
            self.pose_mlp = self._get_pose_mlp(num_joints=num_joints, output_size=6)
            if init_xavier:
                nn.init.xavier_uniform_(self.shape_mlp.weight, gain=0.01)
                nn.init.xavier_uniform_(self.cam_mlp.weight, gain=0.01)
                nn.init.xavier_uniform_(self.pose_mlp.weight, gain=0.01)
        if self.use_branch_nonlocal:
            logger.info(f'Branch nonlocal is active, type {self.use_branch_nonlocal}')
            self.branch_2d_nonlocal = eval(self.use_branch_nonlocal).NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True)
            self.branch_3d_nonlocal = eval(self.use_branch_nonlocal).NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True)
        if self.use_final_nonlocal:
            logger.info(f'Final nonlocal is active, type {self.use_final_nonlocal}')
            self.final_pose_nonlocal = eval(self.use_final_nonlocal).NONLocalBlock1D(in_channels=self.pose_mlp_inp_dim, sub_sample=False, bn_layer=True)
            self.final_shape_nonlocal = eval(self.use_final_nonlocal).NONLocalBlock1D(in_channels=num_features_smpl, sub_sample=False, bn_layer=True)
        if self.use_keypoint_attention:
            logger.info('Keypoint attention is active')
            self.keypoint_attention = KeypointAttention(use_conv=use_postconv_keypoint_attention, in_channels=(self.pose_mlp_inp_dim, smpl_final_dim), out_channels=(self.pose_mlp_inp_dim, smpl_final_dim), act=keypoint_attention_act, use_scale=use_scale_keypoint_attention)
        if self.use_coattention:
            logger.info(f'Coattention is active, final conv type {self.coattention_conv}')
            self.coattention = CoAttention(n_channel=num_deconv_filters[-1], final_conv=self.coattention_conv)
        if self.num_branch_iteration > 0:
            logger.info(f'Branch iteration is active')
            if branch_deeper:
                self.branch_iter_2d_nonlocal = nn.Sequential(conv3x3(num_deconv_filters[-1], 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True), dot_product.NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True))
                self.branch_iter_3d_nonlocal = nn.Sequential(conv3x3(num_deconv_filters[-1], 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True), dot_product.NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True))
            else:
                self.branch_iter_2d_nonlocal = dot_product.NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True)
                self.branch_iter_3d_nonlocal = dot_product.NONLocalBlock2D(in_channels=num_deconv_filters[-1], sub_sample=False, bn_layer=True)

    def _get_shape_mlp(self, output_size):
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)
        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(nn.Linear(self.shape_mlp_inp_dim, self.shape_mlp_hidden_size))
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(nn.Linear(self.shape_mlp_hidden_size, output_size))
            else:
                module_list.append(nn.Linear(self.shape_mlp_hidden_size, self.shape_mlp_hidden_size))
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        if self.pose_mlp_num_layers == 1:
            return LocallyConnected2d(in_channels=self.pose_mlp_inp_dim, out_channels=output_size, output_size=[num_joints, 1], kernel_size=1, stride=1)
        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(LocallyConnected2d(in_channels=self.pose_mlp_inp_dim, out_channels=self.pose_mlp_hidden_size, output_size=[num_joints, 1], kernel_size=1, stride=1))
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(LocallyConnected2d(in_channels=self.pose_mlp_hidden_size, out_channels=output_size, output_size=[num_joints, 1], kernel_size=1, stride=1))
            else:
                module_list.append(LocallyConnected2d(in_channels=self.pose_mlp_hidden_size, out_channels=self.pose_mlp_hidden_size, output_size=[num_joints, 1], kernel_size=1, stride=1))
        return nn.Sequential(*module_list)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(nn.Conv2d(in_channels=self.num_input_features, out_channels=planes, kernel_size=kernel, stride=1, padding=padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes
        return nn.Sequential(*layers)

    def _make_res_conv_layers(self, input_channels, num_channels=64, num_heads=1, num_basic_blocks=2):
        head_layers = []
        head_layers.append(nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        for i in range(num_heads):
            layers = []
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))
        return nn.Sequential(*head_layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.num_input_features, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes
        return nn.Sequential(*layers)

    def _make_upsample_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_layers is different len(num_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_layers is different len(num_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(in_channels=self.num_input_features, out_channels=planes, kernel_size=kernel, stride=1, padding=padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes
        return nn.Sequential(*layers)

    def _prepare_pose_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        batch_size, num_joints = pred_pose.shape[0], pred_pose.shape[2]
        joint_triplets = get_smpl_neighbor_triplets()
        inp_list = []
        for inp_type in self.pose_input_type:
            if inp_type == 'feats':
                inp_list.append(feats)
            if inp_type == 'neighbor_pose_feats':
                n_pose_feat = []
                for jt in joint_triplets:
                    n_pose_feat.append(feats[:, :, jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2))
                n_pose_feat = torch.cat(n_pose_feat, 2)
                inp_list.append(n_pose_feat)
            if inp_type == 'self_pose':
                inp_list.append(pred_pose)
            if inp_type == 'all_pose':
                all_pose = pred_pose.reshape(batch_size, -1, 1)[..., None].repeat(1, 1, num_joints, 1)
                inp_list.append(all_pose)
            if inp_type == 'neighbor_pose':
                n_pose = []
                for jt in joint_triplets:
                    n_pose.append(pred_pose[:, :, jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2))
                n_pose = torch.cat(n_pose, 2)
                inp_list.append(n_pose)
            if inp_type == 'shape':
                pred_shape = pred_shape[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_shape)
            if inp_type == 'cam':
                pred_cam = pred_cam[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_cam)
        assert len(inp_list) > 0
        return torch.cat(inp_list, 1)

    def _prepare_shape_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        batch_size, num_joints = pred_pose.shape[:2]
        inp_list = []
        for inp_type in self.shape_input_type:
            if inp_type == 'feats':
                inp_list.append(feats)
            if inp_type == 'all_pose':
                pred_pose = pred_pose.reshape(batch_size, -1)
                inp_list.append(pred_pose)
            if inp_type == 'shape':
                inp_list.append(pred_shape)
            if inp_type == 'cam':
                inp_list.append(pred_cam)
        assert len(inp_list) > 0
        return torch.cat(inp_list, 1)

    def forward(self, features, gt_segm=None):
        batch_size = features.shape[0]
        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        if self.use_position_encodings:
            features = torch.cat((features, self.pos_enc.repeat(features.shape[0], 1, 1, 1)), 1)
        output = {}
        part_feats = self._get_2d_branch_feats(features)
        part_attention = self._get_part_attention_map(part_feats, output)
        smpl_feats = self._get_3d_smpl_feats(features, part_feats)
        if gt_segm is not None:
            gt_segm = F.interpolate(gt_segm.unsqueeze(1).float(), scale_factor=(1 / 4, 1 / 4), mode='nearest').long().squeeze(1)
            part_attention = F.one_hot(gt_segm, num_classes=self.num_joints + 1).permute(0, 3, 1, 2).float()[:, 1:, :, :]
            part_attention = part_attention
        point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
        pred_pose, pred_shape, pred_cam = self._get_final_preds(point_local_feat, cam_shape_feats, init_pose, init_shape, init_cam)
        if self.use_coattention:
            for c in range(self.num_coattention_iter):
                smpl_feats, part_feats = self.coattention(smpl_feats, part_feats)
                part_attention = self._get_part_attention_map(part_feats, output)
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
                pred_pose, pred_shape, pred_cam = self._get_final_preds(point_local_feat, cam_shape_feats, pred_pose, pred_shape, pred_cam)
        if self.num_branch_iteration > 0:
            for nbi in range(self.num_branch_iteration):
                if self.use_soft_attention:
                    smpl_feats = self.branch_iter_3d_nonlocal(smpl_feats)
                    part_feats = self.branch_iter_2d_nonlocal(part_feats)
                else:
                    smpl_feats = self.branch_iter_3d_nonlocal(smpl_feats)
                    part_feats = smpl_feats
                part_attention = self._get_part_attention_map(part_feats, output)
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
                pred_pose, pred_shape, pred_cam = self._get_final_preds(point_local_feat, cam_shape_feats, pred_pose, pred_shape, pred_cam)
        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)
        output.update({'pred_pose': pred_rotmat, 'pred_cam': pred_cam, 'pred_shape': pred_shape})
        return output

    def _get_local_feats(self, smpl_feats, part_attention, output):
        cam_shape_feats = self.smpl_final_layer(smpl_feats)
        if self.use_keypoint_attention:
            point_local_feat = self.keypoint_attention(smpl_feats, part_attention)
            cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)
        else:
            point_local_feat = interpolate(smpl_feats, output['pred_kp2d'])
            cam_shape_feats = interpolate(cam_shape_feats, output['pred_kp2d'])
        return point_local_feat, cam_shape_feats

    def _get_2d_branch_feats(self, features):
        part_feats = self.keypoint_deconv_layers(features)
        if self.use_branch_nonlocal:
            part_feats = self.branch_2d_nonlocal(part_feats)
        return part_feats

    def _get_3d_smpl_feats(self, features, part_feats):
        if self.use_keypoint_features_for_smpl_regression:
            smpl_feats = part_feats
        else:
            smpl_feats = self.smpl_deconv_layers(features)
            if self.use_branch_nonlocal:
                smpl_feats = self.branch_3d_nonlocal(smpl_feats)
        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):
        heatmaps = self.keypoint_final_layer(part_feats)
        if self.use_heatmaps == 'hm':
            pred_kp2d, confidence = get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
            output['pred_kp2d_conf'] = confidence
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'hm_soft':
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'part_segm':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:, 1:, :, :]
        elif self.use_heatmaps == 'part_segm_pool':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:, 1:, :, :]
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            for k, v in output.items():
                if torch.any(torch.isnan(v)):
                    logger.debug(f'{k} is Nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if torch.any(torch.isinf(v)):
                    logger.debug(f'{k} is Inf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif self.use_heatmaps == 'attention':
            output['pred_attention'] = heatmaps
        else:
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        return heatmaps

    def _get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam):
        if self.use_hmr_regression:
            return self._hmr_get_final_preds(cam_shape_feats, init_pose, init_shape, init_cam)
        else:
            return self._pare_get_final_preds(pose_feats, cam_shape_feats, init_pose, init_shape, init_cam)

    def _hmr_get_final_preds(self, cam_shape_feats, init_pose, init_shape, init_cam):
        if self.use_final_nonlocal:
            cam_shape_feats = self.final_shape_nonlocal(cam_shape_feats)
        xf = torch.flatten(cam_shape_feats, start_dim=1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(3):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        return pred_pose, pred_shape, pred_cam

    def _pare_get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam):
        pose_feats = pose_feats.unsqueeze(-1)
        if init_pose.shape[-1] == 6:
            init_pose = init_pose.transpose(2, 1).unsqueeze(-1)
        else:
            init_pose = init_pose.reshape(init_pose.shape[0], 6, -1).unsqueeze(-1)
        if self.iterative_regression:
            shape_feats = torch.flatten(cam_shape_feats, start_dim=1)
            pred_pose = init_pose
            pred_cam = init_cam
            pred_shape = init_shape
            for i in range(self.num_iterations):
                pose_mlp_inp = self._prepare_pose_mlp_inp(pose_feats, pred_pose, pred_shape, pred_cam)
                shape_mlp_inp = self._prepare_shape_mlp_inp(shape_feats, pred_pose, pred_shape, pred_cam)
                if self.iter_residual:
                    pred_pose = self.pose_mlp(pose_mlp_inp) + pred_pose
                    pred_cam = self.cam_mlp(shape_mlp_inp) + pred_cam
                    pred_shape = self.shape_mlp(shape_mlp_inp) + pred_shape
                else:
                    pred_pose = self.pose_mlp(pose_mlp_inp)
                    pred_cam = self.cam_mlp(shape_mlp_inp)
                    pred_shape = self.shape_mlp(shape_mlp_inp) + init_shape
        else:
            shape_feats = cam_shape_feats
            if self.use_final_nonlocal:
                pose_feats = self.final_pose_nonlocal(pose_feats.squeeze(-1)).unsqueeze(-1)
                shape_feats = self.final_shape_nonlocal(shape_feats)
            shape_feats = torch.flatten(shape_feats, start_dim=1)
            pred_pose = self.pose_mlp(pose_feats)
            pred_cam = self.cam_mlp(shape_feats)
            pred_shape = self.shape_mlp(shape_feats)
            if self.use_mean_camshape:
                pred_cam = pred_cam + init_cam
                pred_shape = pred_shape + init_shape
            if self.use_mean_pose:
                pred_pose = pred_pose + init_pose
        pred_pose = pred_pose.squeeze(-1).transpose(2, 1)
        return pred_pose, pred_shape, pred_cam

    def forward_pretraining(self, features):
        kp_feats = self.keypoint_deconv_layers(features)
        heatmaps = self.keypoint_final_layer(kp_feats)
        output = {}
        if self.use_heatmaps == 'hm':
            pred_kp2d, confidence = get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
            output['pred_kp2d_conf'] = confidence
        elif self.use_heatmaps == 'hm_soft':
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
        else:
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
        if self.use_keypoint_features_for_smpl_regression:
            smpl_feats = kp_feats
        else:
            smpl_feats = self.smpl_deconv_layers(features)
        cam_shape_feats = self.smpl_final_layer(smpl_feats)
        output.update({'kp_feats': heatmaps, 'heatmaps': heatmaps, 'smpl_feats': smpl_feats, 'cam_shape_feats': cam_shape_feats})
        return output


class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None, use_hands=True, use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()
        extra_joints_idxs = []
        face_keyp_idxs = np.array([vertex_ids['nose'], vertex_ids['reye'], vertex_ids['leye'], vertex_ids['rear'], vertex_ids['lear']], dtype=np.int64)
        extra_joints_idxs = np.concatenate([extra_joints_idxs, face_keyp_idxs])
        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'], vertex_ids['LSmallToe'], vertex_ids['LHeel'], vertex_ids['RBigToe'], vertex_ids['RSmallToe'], vertex_ids['RHeel']], dtype=np.int32)
            extra_joints_idxs = np.concatenate([extra_joints_idxs, feet_keyp_idxs])
        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])
            extra_joints_idxs = np.concatenate([extra_joints_idxs, tips_idxs])
        self.register_buffer('extra_joints_idxs', to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)
        return joints


def convert_pare_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length, crop_res=224):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)
    cx = 2 * (bbox_center[:, 0] - img_w / 2.0) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - img_h / 2.0) / (s * bbox_height)
    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def perspective_projection(points, rotation, translation, focal_length, camera_center, retain_z=False):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    if retain_z:
        return projected_points
    else:
        return projected_points[:, :, :-1]


class SMPLCamHead(nn.Module):

    def __init__(self, img_res=224):
        super(SMPLCamHead, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.add_module('smpl', self.smpl)
        self.img_res = img_res

    def forward(self, rotmat, shape, cam, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h, normalize_joints2d=False):
        """
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :param cam_rotmat (Nx3x3) camera rotation matrix
        :param cam_intrinsics (Nx3x3) camera intrinsics matrix
        :param bbox_scale (N,) bbox height normalized by 200
        :param bbox_center (N,2) bbox center
        :param img_w (N,) original image width
        :param img_h (N,) original image height
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        """
        smpl_output = self.smpl(betas=shape, body_pose=rotmat[:, 1:].contiguous(), global_orient=rotmat[:, 0].unsqueeze(1).contiguous(), pose2rot=False)
        output = {'smpl_vertices': smpl_output.vertices, 'smpl_joints3d': smpl_output.joints}
        joints3d = smpl_output.joints
        cam_t = convert_pare_to_full_img_cam(pare_cam=cam, bbox_height=bbox_scale * 200.0, bbox_center=bbox_center, img_w=img_w, img_h=img_h, focal_length=cam_intrinsics[:, 0, 0], crop_res=self.img_res)
        joints2d = perspective_projection(joints3d, rotation=cam_rotmat, translation=cam_t, cam_intrinsics=cam_intrinsics)
        if normalize_joints2d:
            joints2d = joints2d / (self.img_res / 2.0)
        output['smpl_joints2d'] = joints2d
        output['pred_cam_t'] = cam_t
        return output


def convert_weak_perspective_to_perspective(weak_perspective_camera, focal_length=5000.0, img_res=224):
    perspective_camera = torch.stack([weak_perspective_camera[:, 1], weak_perspective_camera[:, 2], 2 * focal_length / (img_res * weak_perspective_camera[:, 0] + 1e-09)], dim=-1)
    return perspective_camera


class SMPLHead(nn.Module):

    def __init__(self, focal_length=5000.0, img_res=224):
        super(SMPLHead, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.add_module('smpl', self.smpl)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):
        """
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        """
        smpl_output = self.smpl(betas=shape, body_pose=rotmat[:, 1:].contiguous(), global_orient=rotmat[:, 0].unsqueeze(1).contiguous(), pose2rot=False)
        output = {'smpl_vertices': smpl_output.vertices, 'smpl_joints3d': smpl_output.joints}
        if cam is not None:
            joints3d = smpl_output.joints
            batch_size = joints3d.shape[0]
            device = joints3d.device
            cam_t = convert_weak_perspective_to_perspective(cam, focal_length=self.focal_length, img_res=self.img_res)
            joints2d = perspective_projection(joints3d, rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1), translation=cam_t, focal_length=self.focal_length, camera_center=torch.zeros(batch_size, 2, device=device))
            if normalize_joints2d:
                joints2d = joints2d / (self.img_res / 2.0)
            output['smpl_joints2d'] = joints2d
            output['pred_cam_t'] = cam_t
        return output


class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super().__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_rotmat, pred_shape, pred_cam


class SelfAttention(nn.Module):

    def __init__(self, attention_size, batch_first=False, layers=1, dropout=0.0, non_linearity='tanh'):
        super(SelfAttention, self).__init__()
        self.batch_first = batch_first
        if non_linearity == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))
        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, scores


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels, inter_channels=inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels, inter_channels=inter_channels, dimension=3, sub_sample=sub_sample, bn_layer=bn_layer)


class NonLocalAttention(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super(NonLocalAttention, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        """
        input [N, Feats, J, 1]
        output [N, Feats, J, 1]
        """
        batch_size, n_feats, n_joints, _ = input.shape
        input = input.squeeze(-1)
        attention = torch.matmul(input.transpose(2, 1), input)
        norm_attention = F.softmax(attention, dim=-1)
        out = torch.matmul(input, norm_attention)
        out = self.conv1x1(out)
        out = out.unsqueeze(-1)
        return out


def get_backbone_info(backbone):
    info = {'resnet18': {'n_output_channels': 512, 'downsample_rate': 4}, 'resnet34': {'n_output_channels': 512, 'downsample_rate': 4}, 'resnet50': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnet50_adf_dropout': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnet50_dropout': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnet101': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnet152': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnext50_32x4d': {'n_output_channels': 2048, 'downsample_rate': 4}, 'resnext101_32x8d': {'n_output_channels': 2048, 'downsample_rate': 4}, 'wide_resnet50_2': {'n_output_channels': 2048, 'downsample_rate': 4}, 'wide_resnet101_2': {'n_output_channels': 2048, 'downsample_rate': 4}, 'mobilenet_v2': {'n_output_channels': 1280, 'downsample_rate': 4}, 'hrnet_w32': {'n_output_channels': 480, 'downsample_rate': 4}, 'hrnet_w48': {'n_output_channels': 720, 'downsample_rate': 4}, 'dla34': {'n_output_channels': 512, 'downsample_rate': 4}}
    return info[backbone]


def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]
        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()
            updated_pretrained_state_dict = state_dict.copy()
            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(f'size mismatch for "{pk}": copying a param with shape {state_dict[pk].shape} from checkpoint, the shape in current model is {model_state_dict[pk].shape}')
                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat([state_dict[pk], state_dict[pk][:, -7:]], dim=-1)
                            logger.warning(f'Updated "{pk}" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]
            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model


class PARE(nn.Module):

    def __init__(self, num_joints=24, softmax_temp=1.0, num_features_smpl=64, backbone='resnet50', focal_length=5000.0, img_res=224, pretrained=None, iterative_regression=False, iter_residual=False, num_iterations=3, shape_input_type='feats', pose_input_type='feats', pose_mlp_num_layers=1, shape_mlp_num_layers=1, pose_mlp_hidden_size=256, shape_mlp_hidden_size=256, use_keypoint_features_for_smpl_regression=False, use_heatmaps='', use_keypoint_attention=False, keypoint_attention_act='softmax', use_postconv_keypoint_attention=False, use_scale_keypoint_attention=False, use_final_nonlocal=None, use_branch_nonlocal=None, use_hmr_regression=False, use_coattention=False, num_coattention_iter=1, coattention_conv='simple', deconv_conv_kernel_size=4, use_upsampling=False, use_soft_attention=False, num_branch_iteration=0, branch_deeper=False, num_deconv_layers=3, num_deconv_filters=256, use_resnet_conv_hrnet=False, use_position_encodings=None, use_mean_camshape=False, use_mean_pose=False, init_xavier=False, use_cam=False):
        super(PARE, self).__init__()
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            self.backbone = eval(backbone)(pretrained=True, downsample=False, use_conv=use_conv == 'conv')
        else:
            self.backbone = eval(backbone)(pretrained=True)
        self.head = PareHead(num_joints=num_joints, num_input_features=get_backbone_info(backbone)['n_output_channels'], softmax_temp=softmax_temp, num_deconv_layers=num_deconv_layers, num_deconv_filters=[num_deconv_filters] * num_deconv_layers, num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers, num_features_smpl=num_features_smpl, final_conv_kernel=1, iterative_regression=iterative_regression, iter_residual=iter_residual, num_iterations=num_iterations, shape_input_type=shape_input_type, pose_input_type=pose_input_type, pose_mlp_num_layers=pose_mlp_num_layers, shape_mlp_num_layers=shape_mlp_num_layers, pose_mlp_hidden_size=pose_mlp_hidden_size, shape_mlp_hidden_size=shape_mlp_hidden_size, use_keypoint_features_for_smpl_regression=use_keypoint_features_for_smpl_regression, use_heatmaps=use_heatmaps, use_keypoint_attention=use_keypoint_attention, use_postconv_keypoint_attention=use_postconv_keypoint_attention, keypoint_attention_act=keypoint_attention_act, use_scale_keypoint_attention=use_scale_keypoint_attention, use_branch_nonlocal=use_branch_nonlocal, use_final_nonlocal=use_final_nonlocal, backbone=backbone, use_hmr_regression=use_hmr_regression, use_coattention=use_coattention, num_coattention_iter=num_coattention_iter, coattention_conv=coattention_conv, use_upsampling=use_upsampling, use_soft_attention=use_soft_attention, num_branch_iteration=num_branch_iteration, branch_deeper=branch_deeper, use_resnet_conv_hrnet=use_resnet_conv_hrnet, use_position_encodings=use_position_encodings, use_mean_camshape=use_mean_camshape, use_mean_pose=use_mean_pose, init_xavier=init_xavier)
        self.use_cam = use_cam
        if self.use_cam:
            self.smpl = SMPLCamHead(img_res=img_res)
        else:
            self.smpl = SMPLHead(focal_length=focal_length, img_res=img_res)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, images, cam_rotmat=None, cam_intrinsics=None, bbox_scale=None, bbox_center=None, img_w=None, img_h=None, gt_segm=None):
        features = self.backbone(images)
        hmr_output = self.head(features, gt_segm=gt_segm)
        if self.use_cam:
            smpl_output = self.smpl(rotmat=hmr_output['pred_pose'], shape=hmr_output['pred_shape'], cam=hmr_output['pred_cam'], cam_rotmat=cam_rotmat, cam_intrinsics=cam_intrinsics, bbox_scale=bbox_scale, bbox_center=bbox_center, img_w=img_w, img_h=img_h, normalize_joints2d=True)
            smpl_output.update(hmr_output)
        elif isinstance(hmr_output['pred_pose'], list):
            smpl_output = {'smpl_vertices': [], 'smpl_joints3d': [], 'smpl_joints2d': [], 'pred_cam_t': []}
            for idx in range(len(hmr_output['pred_pose'])):
                smpl_out = self.smpl(rotmat=hmr_output['pred_pose'][idx], shape=hmr_output['pred_shape'][idx], cam=hmr_output['pred_cam'][idx], normalize_joints2d=True)
                for k, v in smpl_out.items():
                    smpl_output[k].append(v)
        else:
            smpl_output = self.smpl(rotmat=hmr_output['pred_pose'], shape=hmr_output['pred_shape'], cam=hmr_output['pred_cam'], normalize_joints2d=True)
            smpl_output.update(hmr_output)
        return smpl_output

    def load_pretrained(self, file):
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)
        self.backbone.load_state_dict(state_dict, strict=False)
        load_pretrained_model(self.head, state_dict=state_dict, strict=False, overwrite_shape_mismatch=True)


class FLAMETex(nn.Module):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            None
            raise NotImplementedError
        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode=None):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(vertices: 'Tensor', pose: 'Tensor', dynamic_lmk_faces_idx: 'Tensor', dynamic_lmk_b_coords: 'Tensor', neck_kin_chain: 'List[int]', pose2rot: 'bool'=True) ->Tuple[Tensor, Tensor]:
    """ Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    """
    dtype = vertices.dtype
    batch_size = vertices.shape[0]
    if pose2rot:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)
    rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
    y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39))
    neg_mask = y_rot_angle.lt(0)
    mask = y_rot_angle.lt(-39)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def vertices2landmarks(vertices: 'Tensor', faces: 'Tensor', lmk_faces_idx: 'Tensor', lmk_bary_coords: 'Tensor') ->Tensor:
    """ Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


class ResnetEncoder(nn.Module):

    def __init__(self, append_layers=None):
        super(ResnetEncoder, self).__init__()
        self.feature_dim = 2048
        self.encoder = resnet.load_ResNet50Model()
        self.append_layers = append_layers
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.register_buffer('MEAN', torch.tensor(MEAN)[None, :, None, None])
        self.register_buffer('STD', torch.tensor(STD)[None, :, None, None])

    def forward(self, inputs):
        """ inputs: [bz, 3, h, w], range: [0,1]
        """
        inputs = (inputs - self.MEAN) / self.STD
        features = self.encoder(inputs)
        if self.append_layers:
            features = self.last_op(features)
        return features


class MLP(nn.Module):

    def __init__(self, channels=[2048, 1024, 1], last_op=None):
        super(MLP, self).__init__()
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        if last_op:
            layers.append(last_op)
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outs = self.layers(inputs)
        return outs


class HRNEncoder(nn.Module):

    def __init__(self, append_layers=None):
        super(HRNEncoder, self).__init__()
        self.feature_dim = 2048
        self.encoder = hrnet.load_HRNet(pretrained=True)
        self.append_layers = append_layers
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.register_buffer('MEAN', torch.tensor(MEAN)[None, :, None, None])
        self.register_buffer('STD', torch.tensor(STD)[None, :, None, None])

    def forward(self, inputs):
        """ inputs: [bz, 3, h, w], range: [0,1]
        """
        inputs = (inputs - self.MEAN) / self.STD
        features = self.encoder(inputs)['concat']
        if self.append_layers:
            features = self.last_op(features)
        return features


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()
        use_old_impl = cfg.get('use_old_impl')
        self.use_old_impl = use_old_impl
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = cfg.get('stage1', {})
        num_channels = self.stage1_cfg['num_channels'][0]
        block = blocks_dict[self.stage1_cfg['block']]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        self.layer1 = self._make_layer(block, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = cfg.get('stage2', {})
        num_channels = self.stage2_cfg.get('num_channels', (32, 64))
        block = blocks_dict[self.stage2_cfg.get('block')]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        stage2_num_channels = num_channels
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg.get('stage3')
        num_channels = self.stage3_cfg['num_channels']
        block = blocks_dict[self.stage3_cfg['block']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        stage3_num_channels = num_channels
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg.get('stage4')
        num_channels = self.stage4_cfg['num_channels']
        block = blocks_dict[self.stage4_cfg['block']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        stage_4_out_channels = num_channels
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=not self.use_old_impl)
        stage4_num_channels = num_channels
        self.output_channels_dim = pre_stage_channels
        self.pretrained_layers = cfg['pretrained_layers']
        self.init_weights()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if use_old_impl:
            in_dims = 2 ** 2 * stage2_num_channels[-1] + 2 ** 1 * stage3_num_channels[-1] + stage_4_out_channels[-1]
        else:
            in_dims = 4 * 384
            self.subsample_4 = self._make_subsample_layer(in_channels=stage4_num_channels[0], num_layers=3)
        self.subsample_3 = self._make_subsample_layer(in_channels=stage2_num_channels[-1], num_layers=2)
        self.subsample_2 = self._make_subsample_layer(in_channels=stage3_num_channels[-1], num_layers=1)
        self.conv_layers = self._make_conv_layer(in_channels=in_dims, num_layers=5)

    def get_output_dim(self):
        base_output = {f'layer{idx + 1}': val for idx, val in enumerate(self.output_channels_dim)}
        output = base_output.copy()
        for key in base_output:
            output[f'{key}_avg_pooling'] = output[key]
        output['concat'] = 2048
        return output

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_conv_layer(self, in_channels=2048, num_layers=3, num_filters=2048, stride=1):
        layers = []
        for i in range(num_layers):
            downsample = nn.Conv2d(in_channels, num_filters, stride=1, kernel_size=1, bias=False)
            layers.append(Bottleneck(in_channels, num_filters // 4, downsample=downsample))
            in_channels = num_filters
        return nn.Sequential(*layers)

    def _make_subsample_layer(self, in_channels=96, num_layers=3, stride=2):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=stride, padding=1))
            in_channels = 2 * in_channels
            layers.append(nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, log=False):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = blocks_dict[layer_config['block']]
        fuse_method = layer_config['fuse_method']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            modules[-1].log = log
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['num_branches']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['num_branches']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        if not self.use_old_impl:
            y_list = self.stage4(x_list)
        output = {}
        for idx, x in enumerate(y_list):
            output[f'layer{idx + 1}'] = x
        feat_list = []
        if self.use_old_impl:
            x3 = self.subsample_3(x_list[1])
            x2 = self.subsample_2(x_list[2])
            x1 = x_list[3]
            feat_list = [x3, x2, x1]
        else:
            x4 = self.subsample_4(y_list[0])
            x3 = self.subsample_3(y_list[1])
            x2 = self.subsample_2(y_list[2])
            x1 = y_list[3]
            feat_list = [x4, x3, x2, x1]
        xf = self.conv_layers(torch.cat(feat_list, dim=1))
        xf = xf.mean(dim=(2, 3))
        xf = xf.view(xf.size(0), -1)
        output['concat'] = xf
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def load_weights(self, pretrained=''):
        pretrained = osp.expandvars(pretrained)
        if osp.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            missing, unexpected = self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))


class JointsFromVerticesSelector(nn.Module):

    def __init__(self, fname):
        """ Selects extra joints from vertices
        """
        super(JointsFromVerticesSelector, self).__init__()
        err_msg = 'Either pass a filename or triangle face ids, names and barycentrics'
        assert fname is not None or face_ids is not None and bcs is not None and names is not None, err_msg
        if fname is not None:
            fname = os.path.expanduser(os.path.expandvars(fname))
            with open(fname, 'r') as f:
                data = yaml.safe_load(f)
            names = list(data.keys())
            bcs = []
            face_ids = []
            for name, d in data.items():
                face_ids.append(d['face'])
                bcs.append(d['bc'])
            bcs = np.array(bcs, dtype=np.float32)
            face_ids = np.array(face_ids, dtype=np.int32)
        assert len(bcs) == len(face_ids), 'The number of barycentric coordinates must be equal to the faces'
        assert len(names) == len(face_ids), 'The number of names must be equal to the number of '
        self.names = names
        self.register_buffer('bcs', torch.tensor(bcs, dtype=torch.float32))
        self.register_buffer('face_ids', torch.tensor(face_ids, dtype=torch.long))

    def extra_joint_names(self):
        """ Returns the names of the extra joints
        """
        return self.names

    def forward(self, vertices, faces):
        if len(self.face_ids) < 1:
            return []
        vertex_ids = faces[self.face_ids].reshape(-1)
        triangles = torch.index_select(vertices, 1, vertex_ids).reshape(-1, len(self.bcs), 3, 3)
        return (triangles * self.bcs[None, :, :, None]).sum(dim=2)


class TempSoftmaxFusion(nn.Module):

    def __init__(self, channels=[2048 * 2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(TempSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))

    def forward(self, x, y, work=True):
        """
        x: feature from body
        y: feature from part(head/hand) 
        work: whether to fuse features
        """
        if work:
            f_in = torch.cat([x, y], dim=1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_temp = self.layers(f_in)
            f_weight = F.softmax(f_temp * self.temperature, dim=1)
            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:, [0]] * x + f_weight[:, [1]] * y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight


class GumbelSoftmaxFusion(nn.Module):

    def __init__(self, channels=[2048 * 2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(GumbelSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)

    def forward(self, x, y, work=True):
        """
        x: feature from body
        y: feature from part(head/hand) 
        work: whether to fuse features
        """
        if work:
            f_in = torch.cat([x, y], dim=-1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_weight = self.layers(f_in)
            f_weight = f_weight - f_weight.detach() + f_weight.gt(0.5)
            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:, [0]] * x + f_weight[:, [1]] * y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight


class StandardRasterizer(nn.Module):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """

    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height
        self.w = w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        device = vertices.device
        if h is None:
            h = self.h
        if w is None:
            w = self.h
        bz = vertices.shape[0]
        depth_buffer = torch.zeros([bz, h, w]).float() + 1000000.0
        triangle_buffer = torch.zeros([bz, h, w]).int() - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float()
        vert_vis = torch.zeros([bz, vertices.shape[1]]).float()
        vertices = vertices.clone().float()
        vertices[..., 0] = vertices[..., 0] * w / 2 + w / 2
        vertices[..., 1] = vertices[..., 1] * h / 2 + h / 2
        vertices[..., 2] = vertices[..., 2] * w / 2
        f_vs = util.face_vertices(vertices, faces)
        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[:, :, :, None].long()
        bary_coords = baryw_buffer[:, :, :, None, :]
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]
    verts, uvcoords = [], []
    faces, uv_faces = [], []
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode('utf-8') for el in lines]
    for line in lines:
        tokens = line.strip().split()
        if line.startswith('v '):
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = 'Vertex %s does not have 3 values. Line: %s'
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith('vt '):
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError('Texture %s does not have 2 values. Line: %s' % (str(tx), str(line)))
            uvcoords.append(tx)
        elif line.startswith('f '):
            face = tokens[1:]
            face_list = [f.split('/') for f in face]
            for vert_props in face_list:
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != '':
                        uv_faces.append(int(vert_props[1]))
    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)
    faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long)
    uv_faces = uv_faces.reshape(-1, 3) - 1
    return verts, uvcoords, faces, uv_faces


class SRenderY(nn.Module):

    def __init__(self, image_size, obj_filename, uv_size=256, rasterizer_type='standard'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        if rasterizer_type == 'pytorch3d':
            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
            verts, faces, aux = load_obj(obj_filename)
            uvcoords = aux.verts_uvs[None, ...]
            uvfaces = faces.textures_idx[None, ...]
            faces = faces.verts_idx[None, ...]
        elif rasterizer_type == 'standard':
            self.rasterizer = StandardRasterizer(image_size)
            self.uv_rasterizer = StandardRasterizer(uv_size)
            verts, uvcoords, faces, uvfaces = load_obj(obj_filename)
            verts = verts[None, ...]
            uvcoords = uvcoords[None, ...]
            faces = faces[None, ...]
            uvfaces = uvfaces[None, ...]
        else:
            NotImplementedError
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.0
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('vertex_colors', colors)
        self.register_buffer('face_colors', face_colors)
        pi = np.pi
        constant_factor = torch.tensor([1 / np.sqrt(4 * pi), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * (3 / 2) * np.sqrt(5 / (12 * pi)), pi / 4 * (1 / 2) * np.sqrt(5 / (4 * pi))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point', background=None, h=None, w=None):
        """
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], rnage:[-1,1], projected vertices, in image space, for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        """
        batch_size = vertices.shape[0]
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(), face_normals], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        uvcoords_images = rendering[:, :3, :, :]
        grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            elif light_type == 'point':
                vertice_images = rendering[:, 6:9, :, :].detach()
                shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
                shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            else:
                shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
                shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.0
        if background is None:
            images = images * alpha_images + torch.ones_like(images) * (1 - alpha_images)
        else:
            images = images * alpha_images + background.contiguous() * (1 - alpha_images)
        outputs = {'images': images, 'albedo_images': albedo_images, 'alpha_images': alpha_images, 'pos_mask': pos_mask, 'shading_images': shading_images, 'grid': grid, 'normals': normals, 'normal_images': normal_images, 'transformed_normals': transformed_normals}
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        """
            sh_coeff: [bz, 9, 3]
        """
        N = normal_images
        sh = torch.stack([N[:, 0] * 0.0 + 1.0, N[:, 0], N[:, 1], N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2], N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * N[:, 2] ** 2 - 1], 1)
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)
        return shading

    def add_pointlight(self, vertices, normals, lights):
        """
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        """
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0.0, 1.0)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, colors=None, background=None, detail_normal_images=None, lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None):
        """
        -- rendering shape with detail normal map
        """
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor([[-5, 5, -5], [5, 5, -5], [-5, -5, -5], [5, -5, -5], [0, 0, -5]])[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2)
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        if colors is None:
            colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors, transformed_face_normals.detach(), face_vertices.detach(), face_normals, self.face_uvcoords.expand(batch_size, -1, -1, -1)], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        albedo_images = rendering[:, :3, :, :]
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0).float()
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images
        if uv_detail_normals is not None:
            uvcoords_images = rendering[:, 12:15, :, :]
            grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
            detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)
            normal_images = detail_normal_images
        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2).contiguous()
        shaded_images = albedo_images * shading_images
        if background is None:
            shape_images = shaded_images * alpha_images + torch.ones_like(shaded_images) * (1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + background.contiguous() * (1 - alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]
            grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid
        else:
            return shape_images

    def render_depth(self, transformed_vertices):
        """
        -- rendering depth
        """
        transformed_vertices = transformed_vertices.clone()
        batch_size = transformed_vertices.shape[0]
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_colors(self, transformed_vertices, colors, h=None, w=None):
        """
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        """
        transformed_vertices = transformed_vertices.clone()
        batch_size = colors.shape[0]
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10
        attributes = util.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h=h, w=w)
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :] * alpha_images
        return images

    def world2uv(self, vertices):
        """
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices


class ResNet_Backbone(nn.Module):
    """ Feature Extrator with ResNet backbone
    """

    def __init__(self, model='res50', pretrained=True):
        if model == 'res50':
            block, layers = Bottleneck, [3, 4, 6, 3]
        else:
            pass
        self.inplanes = 64
        super().__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if pretrained:
            resnet_imagenet = resnet.resnet50(pretrained=True)
            self.load_state_dict(resnet_imagenet.state_dict(), strict=False)
            logger.info('loaded resnet50 imagenet pretrained model')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0
            return deconv_kernel, padding, output_padding
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        x_featmap = x4
        return x_featmap, xf


def projection(pred_joints, pred_camera, retain_z=False):
    pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-09)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints, rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1), translation=pred_cam_t, focal_length=5000.0, camera_center=camera_center, retain_z=retain_z)
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


class MAF_Extractor(nn.Module):
    """ Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    """

    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.PyMAF.MLP_DIM
        self.last_op = nn.ReLU(True)
        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1))
            else:
                self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))
            self.add_module('conv%d' % l, self.filters[l])
        self.im_feat = None
        self.cam = None
        smpl_mesh_graph = np.load(MESH_DOWNSAMPLEING, allow_pickle=True, encoding='latin1')
        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D']
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense())
        self.register_buffer('Dmap', Dmap)

    def reduce_dim(self, feature):
        """
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        """
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2]).mean(dim=1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1], feature.shape[2]).mean(dim=1)
        y = self.last_op(y)
        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        """
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        """
        if im_feat is None:
            im_feat = self.im_feat
        batch_size = im_feat.shape[0]
        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0]
        mesh_align_feat = self.reduce_dim(point_feat)
        return mesh_align_feat

    def forward(self, p, s_feat=None, cam=None, **kwargs):
        """ Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        """
        if cam is None:
            cam = self.cam
        p_proj_2d = projection(p, cam, retain_z=False)
        mesh_align_feat = self.sampling(p_proj_2d, s_feat)
        return mesh_align_feat


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


H36M_TO_J14 = H36M_TO_J17[:14]


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: 'torch.Tensor' = quaternion[..., 1]
    q2: 'torch.Tensor' = quaternion[..., 2]
    q3: 'torch.Tensor' = quaternion[..., 3]
    sin_squared_theta: 'torch.Tensor' = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: 'torch.Tensor' = torch.sqrt(sin_squared_theta)
    cos_theta: 'torch.Tensor' = quaternion[..., 0]
    two_theta: 'torch.Tensor' = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: 'torch.Tensor' = two_theta / sin_theta
    k_neg: 'torch.Tensor' = 2.0 * torch.ones_like(sin_theta)
    k: 'torch.Tensor' = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: 'torch.Tensor' = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


class Regressor(nn.Module):

    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()
        npose = 24 * 6
        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis
        output = {'theta': torch.cat([pred_cam, pred_shape, pose], dim=1), 'verts': pred_vertices, 'kp_2d': pred_keypoints_2d, 'kp_3d': pred_joints, 'smpl_kp_3d': pred_smpl_joints, 'rotmat': pred_rotmat, 'pred_cam': pred_cam, 'pred_shape': pred_shape, 'pred_pose': pred_pose}
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis
        output = {'theta': torch.cat([pred_cam, pred_shape, pose], dim=1), 'verts': pred_vertices, 'kp_2d': pred_keypoints_2d, 'kp_3d': pred_joints, 'smpl_kp_3d': pred_smpl_joints, 'rotmat': pred_rotmat, 'pred_cam': pred_cam, 'pred_shape': pred_shape, 'pred_pose': pred_pose}
        return output


class IUV_predict_layer(nn.Module):

    def __init__(self, feat_dim=256, final_cov_k=3, part_out_dim=25, with_uv=True):
        super().__init__()
        self.with_uv = with_uv
        if self.with_uv:
            self.predict_u = nn.Conv2d(in_channels=feat_dim, out_channels=25, kernel_size=final_cov_k, stride=1, padding=1 if final_cov_k == 3 else 0)
            self.predict_v = nn.Conv2d(in_channels=feat_dim, out_channels=25, kernel_size=final_cov_k, stride=1, padding=1 if final_cov_k == 3 else 0)
        self.predict_ann_index = nn.Conv2d(in_channels=feat_dim, out_channels=15, kernel_size=final_cov_k, stride=1, padding=1 if final_cov_k == 3 else 0)
        self.predict_uv_index = nn.Conv2d(in_channels=feat_dim, out_channels=25, kernel_size=final_cov_k, stride=1, padding=1 if final_cov_k == 3 else 0)
        self.inplanes = feat_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return_dict = {}
        predict_uv_index = self.predict_uv_index(x)
        predict_ann_index = self.predict_ann_index(x)
        return_dict['predict_uv_index'] = predict_uv_index
        return_dict['predict_ann_index'] = predict_ann_index
        if self.with_uv:
            predict_u = self.predict_u(x)
            predict_v = self.predict_v(x)
            return_dict['predict_u'] = predict_u
            return_dict['predict_v'] = predict_v
        else:
            return_dict['predict_u'] = None
            return_dict['predict_v'] = None
        return return_dict


resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2]), (34): (BasicBlock, [3, 4, 6, 3]), (50): (Bottleneck, [3, 4, 6, 3]), (101): (Bottleneck, [3, 4, 23, 3]), (152): (Bottleneck, [3, 8, 36, 3])}


class SmplResNet(nn.Module):

    def __init__(self, resnet_nums, in_channels=3, num_classes=229, last_stride=2, n_extra_feat=0, truncate=0, **kwargs):
        super().__init__()
        self.inplanes = 64
        self.truncate = truncate
        block, layers = resnet_spec[resnet_nums]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) if truncate < 2 else None
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride) if truncate < 1 else None
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if num_classes > 0:
            self.final_layer = nn.Linear(512 * block.expansion, num_classes)
            nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        self.n_extra_feat = n_extra_feat
        if n_extra_feat > 0:
            self.trans_conv = nn.Sequential(nn.Conv2d(n_extra_feat + 512 * block.expansion, 512 * block.expansion, kernel_size=1, bias=False), nn.BatchNorm2d(512 * block.expansion, momentum=BN_MOMENTUM), nn.ReLU(True))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, infeat=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) if self.truncate < 2 else x2
        x4 = self.layer4(x3) if self.truncate < 1 else x3
        if infeat is not None:
            x4 = self.trans_conv(torch.cat([infeat, x4], 1))
        if self.num_classes > 0:
            xp = self.avg_pooling(x4)
            cls = self.final_layer(xp.view(xp.size(0), -1))
            if not cfg.DANET.USE_MEAN_PARA:
                scale = F.relu(cls[:, 0]).unsqueeze(1)
                cls = torch.cat((scale, cls[:, 1:]), dim=1)
        else:
            cls = None
        return cls, {'x4': x4}

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict_old = self.state_dict()
                for key in state_dict_old.keys():
                    if key in checkpoint.keys():
                        if state_dict_old[key].shape != checkpoint[key].shape:
                            del checkpoint[key]
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError('No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


class LimbResLayers(nn.Module):

    def __init__(self, resnet_nums, inplanes, outplanes=None, groups=1, **kwargs):
        super().__init__()
        self.inplanes = inplanes
        block, layers = resnet_spec[resnet_nums]
        self.outplanes = 512 if outplanes == None else outplanes
        self.layer4 = self._make_layer(block, self.outplanes, layers[3], stride=2, groups=groups)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes * groups, planes * block.expansion * groups, kernel_size=1, stride=stride, bias=False, groups=groups), nn.BatchNorm2d(planes * block.expansion * groups, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=groups))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avg_pooling(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CoAttention,
     lambda: ([], {'n_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Down,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GMoF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IUV_predict_layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KeypointAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LocallyConnected2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'output_size': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NONLocalBlock1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NONLocalBlock2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet_Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (ResnetFilter,
     lambda: ([], {'opt': SimpleNamespace(use_tanh=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SelfAttention,
     lambda: ([], {'attention_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SmoothConv2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SmoothConv3D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'n_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (Up,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})),
    (VGGLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (VectorQuantizer,
     lambda: ([], {'embed_dim': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

