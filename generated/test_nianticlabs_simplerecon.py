
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


import numpy as np


import torch


from torchvision import transforms


import functools


import random


from torch.utils.data import Dataset


import re


from scipy.spatial.transform import Rotation as R


import torch.nn.functional as F


from torch import nn


import torch.jit as jit


from torch import Tensor


import torch.nn as nn


from typing import Callable


from typing import Optional


from torchvision import models


from torchvision.ops import FeaturePyramidNetwork


from typing import Tuple


import torch.nn.functional as TF


from torch.utils.data import DataLoader


import torchvision.transforms.functional as TF


import matplotlib.pyplot as plt


import scipy


class MSGradientLoss(nn.Module):

    def __init__(self, num_scales: 'int'=4):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, depth_gt: 'Tensor', depth_pred: 'Tensor') ->Tensor:
        depth_pred_pyr = pyrdown(depth_pred, self.num_scales)
        depth_gtn_pyr = pyrdown(depth_gt, self.num_scales)
        grad_loss = torch.tensor(0, dtype=depth_gt.dtype, device=depth_gt.device)
        for depth_pred_down, depth_gtn_down in zip(depth_pred_pyr, depth_gtn_pyr):
            depth_gtn_grad = kornia.filters.spatial_gradient(depth_gtn_down)
            mask_down_b = depth_gtn_grad.isfinite().all(dim=1, keepdim=True)
            depth_pred_grad = kornia.filters.spatial_gradient(depth_pred_down).masked_select(mask_down_b)
            grad_error = torch.abs(depth_pred_grad - depth_gtn_grad.masked_select(mask_down_b))
            grad_loss += torch.mean(grad_error)
        return grad_loss


class ScaleInvariantLoss(jit.ScriptModule):

    def __init__(self, si_lambda: 'float'=0.85):
        super().__init__()
        self.si_lambda = si_lambda

    @jit.script_method
    def forward(self, log_depth_gt: 'Tensor', log_depth_pred: 'Tensor') ->Tensor:
        log_diff = log_depth_gt - log_depth_pred
        si_loss = torch.sqrt((log_diff ** 2).mean() - self.si_lambda * log_diff.mean() ** 2)
        return si_loss


class NormalsLoss(nn.Module):

    def forward(self, normals_gt_b3hw: 'Tensor', normals_pred_b3hw: 'Tensor') ->Tensor:
        normals_mask_b1hw = torch.logical_and(normals_gt_b3hw.isfinite().all(dim=1, keepdim=True), normals_pred_b3hw.isfinite().all(dim=1, keepdim=True))
        normals_pred_b3hw = normals_pred_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        normals_gt_b3hw = normals_gt_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        with torch.amp.autocast(enabled=False):
            normals_dot_b1hw = 0.5 * (1.0 - torch.einsum('bchw, bchw -> bhw', normals_pred_b3hw, normals_gt_b3hw)).unsqueeze(1)
        normals_loss = normals_dot_b1hw.masked_select(normals_mask_b1hw).mean()
        return normals_loss


@torch.jit.script
def to_homogeneous(input_tensor: 'Tensor', dim: 'int'=0) ->Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified 
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN


class BackprojectDepth(jit.ScriptModule):
    """
    Layer that projects points from 2D camera to 3D space. The 3D points are 
    represented in homogeneous coordinates.
    """

    def __init__(self, height: 'int', width: 'int'):
        super().__init__()
        self.height = height
        self.width = width
        xx, yy = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy')
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5
        pix_coords_13N = to_homogeneous(pix_coords_2hw, dim=0).flatten(1).unsqueeze(0)
        self.register_buffer('pix_coords_13N', pix_coords_13N)

    @jit.script_method
    def forward(self, depth_b1hw: 'Tensor', invK_b44: 'Tensor') ->Tensor:
        """ 
        Backprojects spatial points in 2D image space to world space using 
        invK_b44 at the depths defined in depth_b1hw. 
        """
        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N)
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N


class Project3D(jit.ScriptModule):
    """
    Layer that projects 3D points into the 2D camera
    """

    def __init__(self, eps: 'float'=1e-08):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps).view(1, 1, 1))

    @jit.script_method
    def forward(self, points_b4N: 'Tensor', K_b44: 'Tensor', cam_T_world_b44: 'Tensor') ->Tensor:
        """
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44
        cam_points_b3N = P_b44[:, :3] @ points_b4N
        mask = torch.abs(cam_points_b3N[:, 2:]) > self.eps
        depth_b1N = cam_points_b3N[:, 2:] + self.eps
        scale = torch.where(mask, 1.0 / depth_b1N, torch.tensor(1.0, device=depth_b1N.device))
        pix_coords_b2N = cam_points_b3N[:, :2] * scale
        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


class MVDepthLoss(nn.Module):

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.backproject = BackprojectDepth(self.height, self.width)
        self.project = Project3D()

    def get_valid_mask(self, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]
        cur_cam_points_b4N = self.backproject(cur_depth_b1hw, cur_invK_b44)
        world_points_b4N = cur_world_T_cam_b44 @ cur_cam_points_b4N
        src_cam_points_b3N = self.project(world_points_b4N, src_K_b44, src_cam_T_world_b44)
        cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pix_coords_b2hw = cam_points_b3hw[:, :2]
        proj_src_depths_b1hw = cam_points_b3hw[:, 2:]
        uv_coords = pix_coords_b2hw.permute(0, 2, 3, 1) / torch.tensor([depth_width, depth_height]).view(1, 1, 1, 2).type_as(pix_coords_b2hw)
        uv_coords = 2 * uv_coords - 1
        src_depth_sampled_b1hw = F.grid_sample(input=src_depth_b1hw, grid=uv_coords, padding_mode='zeros', mode='nearest', align_corners=False)
        valid_mask_b1hw = proj_src_depths_b1hw < 1.05 * src_depth_sampled_b1hw
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, proj_src_depths_b1hw > 0)
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, src_depth_sampled_b1hw > 0)
        return valid_mask_b1hw, src_depth_sampled_b1hw

    def get_error_for_pair(self, depth_pred_b1hw, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]
        valid_mask_b1hw, src_depth_sampled_b1hw = self.get_valid_mask(cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44)
        pred_cam_points_b4N = self.backproject(depth_pred_b1hw, cur_invK_b44)
        pred_world_points_b4N = cur_world_T_cam_b44 @ pred_cam_points_b4N
        src_cam_points_b3N = self.project(pred_world_points_b4N, src_K_b44, src_cam_T_world_b44)
        pred_cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pred_src_depths_b1hw = pred_cam_points_b3hw[:, 2:]
        depth_diff_b1hw = torch.abs(torch.log(src_depth_sampled_b1hw) - torch.log(pred_src_depths_b1hw)).masked_select(valid_mask_b1hw)
        depth_loss = depth_diff_b1hw.nanmean()
        return depth_loss

    def forward(self, depth_pred_b1hw, cur_depth_b1hw, src_depth_bk1hw, cur_invK_b44, src_K_bk44, cur_world_T_cam_b44, src_cam_T_world_bk44):
        src_to_iterate = [torch.unbind(src_depth_bk1hw, dim=1), torch.unbind(src_K_bk44, dim=1), torch.unbind(src_cam_T_world_bk44, dim=1)]
        num_src_frames = src_depth_bk1hw.shape[1]
        loss = 0
        for src_depth_b1hw, src_K_b44, src_cam_T_world_b44 in zip(*src_to_iterate):
            error = self.get_error_for_pair(depth_pred_b1hw, cur_depth_b1hw, src_depth_b1hw, cur_invK_b44, src_K_b44, cur_world_T_cam_b44, src_cam_T_world_b44)
            loss += error
        return loss / num_src_frames


class CostVolumeManager(nn.Module):
    """
    Class to build a cost volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    collapsing over views by taking a dot product between each source and 
    reference feature, before summing over source views at each pixel location. 
    The final tensor is size batch_size x num_depths x H x  W tensor.
    """

    def __init__(self, matching_height, matching_width, num_depth_bins=64, matching_dim_size=None, num_source_views=None):
        """
        matching_dim_size and num_source_views are not used for the standard 
        cost volume.

        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            matching_dim_size: number of channels per visual feature; the basic 
                dot product cost volume does not need this information at init.
            num_source_views: number of source views; the basic dot product cost 
                volume does not need this information at init.
        """
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.matching_height = matching_height
        self.matching_width = matching_width
        self.initialise_for_projection()

    def initialise_for_projection(self):
        """
        Set up for backwarping and projection of feature maps

        Args:
            batch_height: height of the current batch of features
            batch_width: width of the current batch of features
        """
        linear_ramp = torch.linspace(0, 1, self.num_depth_bins).view(1, self.num_depth_bins, 1, 1)
        self.register_buffer('linear_ramp_1d11', linear_ramp)
        self.backprojector = BackprojectDepth(height=self.matching_height, width=self.matching_width)
        self.projector = Project3D()

    def get_mask(self, pix_coords_bk2hw):
        """
        Create a mask to ignore features from the edges or outside of source 
        images.
        
        Args:
            pix_coords_bk2hw: sampling locations of source features
            
        Returns:
            mask: a binary mask indicating whether to ignore a pixels
        """
        mask = torch.logical_and(torch.logical_and(pix_coords_bk2hw[:, :, 0] > 2, pix_coords_bk2hw[:, :, 0] < self.matching_width - 2), torch.logical_and(pix_coords_bk2hw[:, :, 1] > 2, pix_coords_bk2hw[:, :, 1] < self.matching_height - 2))
        return mask

    def generate_depth_planes(self, batch_size: 'int', min_depth: 'Tensor', max_depth: 'Tensor') ->Tensor:
        """
        Creates a depth planes tensor of size batch_size x number of depth planes
        x matching height x matching width. Every plane contains the same depths
        and depths will vary with a log scale from min_depth to max_depth.

        Args:
            batch_size: number of these view replications to make for each 
                element in the batch.
            min_depth: minimum depth tensor defining the starting point for 
                depth planes.
            max_depth: maximum depth tensor defining the end point for 
                depth planes.

        Returns:
            depth_planes_bdhw: depth planes tensor.
        """
        linear_ramp_bd11 = self.linear_ramp_1d11.expand(batch_size, self.num_depth_bins, 1, 1)
        log_depth_planes_bd11 = torch.log(min_depth) + torch.log(max_depth / min_depth) * linear_ramp_bd11
        depth_planes_bd11 = torch.exp(log_depth_planes_bd11)
        depth_planes_bdhw = depth_planes_bd11.expand(batch_size, self.num_depth_bins, self.matching_height, self.matching_width)
        return depth_planes_bdhw

    def warp_features(self, src_feats, src_extrinsics, src_Ks, cur_invK, depth_plane_b1hw, batch_size, num_src_frames, num_feat_channels, uv_scale):
        """
        Warps every soruce view feature to the current view at the depth 
        plane defined by depth_plane_b1hw.

        Args:
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            depth_plane_b1hw: depth plane to use for every spatial location. For 
                SimpleRecon, this will be the same value at each location.
            batch_size: the batch size.
            num_src_frames: number of source views.
            num_feat_channels: number of feature channels for feature maps.
            uv_scale: normalization for image space coords before grid_sample.

        Returns:
            world_points_B4N: the world points at every backprojected depth 
                point in depth_plane_b1hw.
            depths: depths for each projected point in every source views.
            src_feat_warped: warped source view for every spatial location at 
                the depth plane.
            mask: depth mask where 1.0 indicated that the point projected to the
                source view is infront of the view.
        """
        world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
        world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, dim=0)
        cam_points_B3N = self.projector(world_points_B4N, src_Ks.view(-1, 4, 4), src_extrinsics.view(-1, 4, 4))
        cam_points_B3hw = cam_points_B3N.view(-1, 3, self.matching_height, self.matching_width)
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        depths = cam_points_B3hw[:, 2:]
        uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1
        src_feat_warped = F.grid_sample(input=src_feats.view(-1, num_feat_channels, self.matching_height, self.matching_width), grid=uv_coords.type_as(src_feats), padding_mode='zeros', mode='bilinear', align_corners=False)
        src_feat_warped = src_feat_warped.view(batch_size, num_src_frames, num_feat_channels, self.matching_height, self.matching_width)
        depths = depths.view(batch_size, num_src_frames, self.matching_height, self.matching_width)
        mask_b = depths > 0
        mask = mask_b.type_as(src_feat_warped)
        return world_points_B4N, depths, src_feat_warped, mask

    def build_cost_volume(self, cur_feats: 'Tensor', src_feats: 'Tensor', src_extrinsics: 'Tensor', src_poses: 'Tensor', src_Ks: 'Tensor', cur_invK: 'Tensor', min_depth: 'Tensor', max_depth: 'Tensor', depth_planes_bdhw: 'Tensor'=None, return_mask: 'bool'=False):
        """
        Build the cost volume. Using hypothesised depths, we backwarp src_feats 
        onto cur_feats using known intrinsics and take the dot product. 
        We sum the dot over all src_feats.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """
        del src_poses, return_mask
        batch_size, num_src_frames, num_feat_channels, _, _ = src_feats.shape
        uv_scale = torch.tensor([1 / self.matching_width, 1 / self.matching_height], dtype=src_extrinsics.dtype, device=src_extrinsics.device).view(1, 1, 1, 2)
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, min_depth, max_depth)
        all_dps = []
        for depth_id in range(self.num_depth_bins):
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            _, _, src_feat_warped, mask = self.warp_features(src_feats, src_extrinsics, src_Ks, cur_invK, depth_plane_b1hw, batch_size, num_src_frames, num_feat_channels, uv_scale)
            dot_product_bkhw = torch.sum(src_feat_warped * cur_feats.unsqueeze(1), dim=2) * mask
            dot_product_b1hw = dot_product_bkhw.sum(dim=1, keepdim=True)
            all_dps.append(dot_product_b1hw)
        cost_volume = torch.cat(all_dps, dim=1)
        return cost_volume, depth_planes_bdhw, None

    def indices_to_disparity(self, indices, depth_planes_bdhw):
        """ Convert cost volume indices to 1/depth for visualisation """
        depth = torch.gather(depth_planes_bdhw, dim=1, index=indices.unsqueeze(1)).squeeze(1)
        return depth

    def forward(self, cur_feats, src_feats, src_extrinsics, src_poses, src_Ks, cur_invK, min_depth, max_depth, depth_planes_bdhw=None, return_mask=False):
        """ Runs the cost volume and gets the lowest cost result """
        cost_volume, depth_planes_bdhw, overall_mask_bhw = self.build_cost_volume(cur_feats=cur_feats, src_feats=src_feats, src_extrinsics=src_extrinsics, src_Ks=src_Ks, cur_invK=cur_invK, src_poses=src_poses, min_depth=min_depth, max_depth=max_depth, depth_planes_bdhw=depth_planes_bdhw, return_mask=return_mask)
        with torch.no_grad():
            lowest_cost = self.indices_to_disparity(torch.argmax(cost_volume.detach(), 1), depth_planes_bdhw)
        return cost_volume, lowest_cost, depth_planes_bdhw, overall_mask_bhw

