
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


import random


import numpy as np


import copy


import torch


import torch.utils.data as data


from torchvision import transforms


from torch.utils.data import DataLoader


import math


import torch.nn as nn


import torch.nn.functional as F


from warnings import warn


from collections import OrderedDict


from torch.nn.parameter import Parameter


from torch.distributions.normal import Normal


import torchvision.models as models


import torch.utils.model_zoo as model_zoo


import matplotlib as mpl


import matplotlib.cm as cm


from torchvision import datasets


import time


import torch.optim as optim


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-07):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Project3D_Raw(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-07):
        super(Project3D_Raw, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        raw_pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        raw_pix_coords = raw_pix_coords.view(self.batch_size, 2, self.height, self.width)
        raw_pix_coords = raw_pix_coords.permute(0, 2, 3, 1)
        return raw_pix_coords


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiantiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the source image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border')


class optical_flow(nn.Module):

    def __init__(self, size, batch_size, height, width, eps=1e-07):
        super(optical_flow, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        optical_flow = pix_coords[:, [1, 0], ...] - self.grid
        return optical_flow


def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()
    x = data[:, 0, :, :].view(B, -1)
    y = data[:, 1, :, :].view(B, -1)
    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)
    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out, x_ceil_out | y_floor_out, x_floor_out | y_ceil_out, x_floor_out | y_floor_out], dim=1)
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W, x_ceil + y_floor * W, x_floor + y_ceil * W, x_floor + y_floor * W], 1).long()
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)), (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)), (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)), (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))], 1)
    values[invalid] = 0
    corresponding_map.scatter_add_(1, indices, values)
    corresponding_map = corresponding_map.view(B, H, W)
    return corresponding_map.unsqueeze(1)


class get_occu_mask_backward(nn.Module):

    def __init__(self, size):
        super(get_occu_mask_backward, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, flow, th=0.95):
        new_locs = self.grid + flow
        new_locs = new_locs[:, [1, 0], ...]
        corr_map = get_corresponding_map(new_locs)
        occu_map = corr_map
        occu_mask = (occu_map > th).float()
        return occu_mask, occu_map


class get_occu_mask_bidirection(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super(get_occu_mask_bidirection, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, flow12, flow21, scale=0.01, bias=0.5):
        new_locs = self.grid + flow12
        shape = flow12.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        flow21_warped = F.grid_sample(flow21, new_locs, mode=self.mode, padding_mode='border')
        flow12_diff = torch.abs(flow12 + flow21_warped)
        return flow12_diff


class match(nn.Module):

    def __init__(self, size, batch_size):
        super(match, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.batch_size = batch_size

    def forward(self, flow):
        new_locs = self.grid + flow
        mach = torch.cat((self.grid[:, [1, 0], ...].repeat(self.batch_size, 1, 1, 1), new_locs[:, [1, 0], ...]), 1)
        return mach


class BerHuLoss(nn.Module):

    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), 'inconsistent dimensions'
        diff = pred - target
        abs_diff = diff.abs()
        c = 0.2 * abs_diff.max()
        mask = (abs_diff <= c).float()
        l2_loss = (diff ** 2 + c ** 2) / (2 * c)
        loss = (mask * abs_diff + (1 - mask) * l2_loss).mean()
        return loss


class reduced_ransac(nn.Module):

    def __init__(self, check_num, dataset):
        super(reduced_ransac, self).__init__()
        self.check_num = check_num
        self.dataset = dataset

    def robust_rand_sample(self, match, mask, num, robust=True):
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1))
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:, :, rand_int]
        else:
            select_idxs = []
            if robust:
                num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
            for i in range(b):
                nonzero_idx = torch.nonzero(mask[i, 0, :])
                rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
                select_idx = nonzero_idx[rand_int, :]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0)
            select_match = torch.gather(match.transpose(1, 2), index=select_idxs.repeat(1, 1, 4), dim=1).transpose(1, 2)
        return select_match, num

    def top_ratio_sample(self, match, mask, ratio):
        b, total_num = match.shape[0], match.shape[-1]
        scores, indices = torch.topk(mask, int(ratio * total_num), dim=-1)
        select_match = torch.gather(match.transpose(1, 2), index=indices.squeeze(1).unsqueeze(-1).repeat(1, 1, 4), dim=1).transpose(1, 2)
        return select_match, scores

    def forward(self, match, mask, visualizer=None):
        b, h, w = match.shape[0], match.shape[2], match.shape[3]
        match = match.view([b, 4, -1]).contiguous()
        mask = mask.view([b, 1, -1]).contiguous()
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(match, mask, ratio=0.2)
        check_match, check_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.check_num)
        check_match = check_match.contiguous()
        cv_f = []
        for i in range(b):
            if self.dataset == 'nyuv2':
                f, m = cv2.findFundamentalMat(check_match[i, :2, :].transpose(0, 1).detach().cpu().numpy(), check_match[i, 2:, :].transpose(0, 1).detach().cpu().numpy(), cv2.FM_LMEDS, 0.99)
            else:
                f, m = cv2.findFundamentalMat(check_match[i, :2, :].transpose(0, 1).detach().cpu().numpy(), check_match[i, 2:, :].transpose(0, 1).detach().cpu().numpy(), cv2.FM_RANSAC, 0.1, 0.99)
            cv_f.append(f)
        cv_f = np.stack(cv_f, axis=0)
        cv_f = torch.from_numpy(cv_f).float()
        return cv_f


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')


class TransformDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(TransformDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs['transform_conv', s] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.Tanh = nn.Tanh()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs['upconv', i, 0](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs['upconv', i, 1](x)
            if i in self.scales:
                self.outputs['transform', i] = self.Tanh(self.convs['transform_conv', i](x))
        return self.outputs


class SPPSELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(nn.Linear(channel * 21, channel * 21 // reduction, bias=True), nn.ELU(inplace=True), nn.Linear(channel * 21 // reduction, channel, bias=True), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv0 = nn.Conv2d(num_channels, num_channels // 16, 1)
        self.conv1 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=1)
        self.conv2 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=2)
        self.conv3 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=3)
        self.conv4 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=4)
        self.conv5 = nn.Conv2d(num_channels // 16, 1, 1)
        self.elu = nn.ELU(inplace=True)
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.pad3 = nn.ReflectionPad2d(3)
        self.pad4 = nn.ReflectionPad2d(4)
        self.sigmoid = nn.Sigmoid()
        self.p1 = Parameter(torch.ones(1))
        self.p2 = Parameter(torch.zeros(1))
        self.p3 = Parameter(torch.zeros(1))
        self.p4 = Parameter(torch.zeros(1))

    def forward(self, input_tensor):
        batch_size, channel, a, b = input_tensor.size()
        out0 = self.conv0(input_tensor)
        out1 = self.pad1(out0)
        out1 = self.conv1(out1)
        out1 = self.elu(out1)
        att1 = self.conv5(out1)
        att1 = self.sigmoid(att1)
        out2 = torch.add(out0, out1)
        out2 = self.pad2(out2)
        out2 = self.conv2(out2)
        out2 = self.elu(out2)
        att2 = self.conv5(out2)
        att2 = self.sigmoid(att2)
        out3 = torch.add(out0, out2)
        out3 = self.pad3(out3)
        out3 = self.conv3(out3)
        out3 = self.elu(out3)
        att3 = self.conv5(out3)
        att3 = self.sigmoid(att3)
        out4 = torch.add(out0, out3)
        out4 = self.pad3(out4)
        out4 = self.conv3(out4)
        out4 = self.elu(out4)
        att4 = self.conv5(out4)
        att4 = self.sigmoid(att4)
        att1 = att1.view(batch_size, 1, a, b)
        att2 = att2.view(batch_size, 1, a, b)
        att3 = att3.view(batch_size, 1, a, b)
        att4 = att4.view(batch_size, 1, a, b)
        out1 = torch.mul(input_tensor, att1)
        out2 = torch.mul(input_tensor, att2)
        out3 = torch.mul(input_tensor, att3)
        out4 = torch.mul(input_tensor, att4)
        output_tensor = self.elu(self.p1 * out1 + self.p2 * out2 + self.p3 * out3 + self.p4 * out4)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = SPPSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        self.conv = nn.Conv2d(2 * num_channels, num_channels, 1)
        self.elu = nn.ELU()

    def forward(self, input_tensor):
        output_tensor = torch.cat((self.cSE(input_tensor), self.sSE(input_tensor)), dim=1)
        output_tensor = self.conv(output_tensor)
        output_tensor = torch.add(input_tensor, output_tensor)
        output_tensor = self.elu(output_tensor)
        return output_tensor


class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs['dispconv', s] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.convs['attention', 0] = ChannelSpatialSELayer(self.num_ch_enc[-1], reduction_ratio=16)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        x = self.convs['attention', 0](x)
        for i in range(4, -1, -1):
            x = self.convs['upconv', i, 0](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs['upconv', i, 1](x)
            if i in self.scales:
                self.outputs['disp', i] = self.sigmoid(self.convs['dispconv', i](x))
        return self.outputs


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.elu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class PositionDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=2, use_skips=True):
        super(PositionDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.conv = getattr(nn, 'Conv2d')
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs['position_conv', s] = self.conv(self.num_ch_dec[s], self.num_output_channels, kernel_size=3, padding=1)
            self.convs['position_conv', s].weight = nn.Parameter(Normal(0, 1e-05).sample(self.convs['position_conv', s].weight.shape))
            self.convs['position_conv', s].bias = nn.Parameter(torch.zeros(self.convs['position_conv', s].bias.shape))
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs['upconv', i, 0](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs['upconv', i, 1](x)
            if i in self.scales:
                self.outputs['position', i] = self.convs['position_conv', i](x)
        return self.outputs


class PoseCNN(nn.Module):

    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)
        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
        out = self.pose_conv(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation


class PoseDecoder(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.convs = OrderedDict()
        self.convs['squeeze'] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs['pose', 0] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs['pose', 1] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs['pose', 2] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs['squeeze'](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        for i in range(3):
            out = self.convs['pose', i](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        out = 0.001 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'
    blocks = {(18): [2, 2, 2, 2], (50): [3, 4, 6, 3]}[num_layers]
    block_type = {(18): models.resnet.BasicBlock, (50): models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {(18): models.resnet18, (34): models.resnet34, (50): models.resnet50, (101): models.resnet101, (152): models.resnet152}
        if num_layers not in resnets:
            raise ValueError('{} is not a valid number of resnet layers'.format(num_layers))
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BackprojectDepth,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (BerHuLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChannelSELayer,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PoseCNN,
     lambda: ([], {'num_input_frames': 4}),
     lambda: ([torch.rand([4, 12, 64, 64])], {})),
    (Project3D,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Project3D_Raw,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SPPSELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialTransformer,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])], {})),
    (get_occu_mask_backward,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 2, 4, 4])], {})),
    (get_occu_mask_bidirection,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4])], {})),
    (match,
     lambda: ([], {'size': [4, 4], 'batch_size': 4}),
     lambda: ([torch.rand([4, 2, 4, 4])], {})),
    (optical_flow,
     lambda: ([], {'size': [4, 4], 'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

