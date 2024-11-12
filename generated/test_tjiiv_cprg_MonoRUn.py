
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


import matplotlib.pyplot as plt


import numpy as np


import torch


from torch.nn.modules.utils import _pair


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Upsample


from torch.nn.modules.batchnorm import _NormBase


import copy


import time


class KLLossMV(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KLLossMV, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, inv_cov=None, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * kl_loss_mv(pred, target, weight, inv_cov=inv_cov, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class RobustKLLoss(nn.Module):

    def __init__(self, delta=1.414, reduction='mean', loss_weight=1.0, momentum=1.0, eps=0.0001):
        super(RobustKLLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('mean_inv_std', torch.tensor(1, dtype=torch.float))

    def forward(self, pred, target, logstd=None, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * robust_kl_loss(pred, target, weight, logstd=logstd, delta=self.delta, momentum=self.momentum, mean_inv_std=self.mean_inv_std, eps=self.eps, training=self.training, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


def build_coord_coder(cfg, **default_args):
    return build_from_cfg(cfg, COORD_CODERS, default_args)


def masked_dense_target_single(pos_proposals, pos_assigned_gt_inds, gt_dense, gt_mask, cfg, eps=0.0001):
    dense_size = _pair(cfg.dense_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        maxh, maxw = gt_dense.shape[-2:]
        pos_proposals_clip = torch.empty_like(pos_proposals)
        pos_proposals_clip[:, [0, 2]] = pos_proposals[:, [0, 2]].clamp(0, maxw)
        pos_proposals_clip[:, [1, 3]] = pos_proposals[:, [1, 3]].clamp(0, maxh)
        rois = torch.cat([pos_assigned_gt_inds[:, None], pos_proposals_clip], dim=1)
        targets = roi_align(gt_dense, rois, dense_size, 1.0, 0, 'avg', True).permute(0, 2, 3, 1)
        mask = roi_align(gt_mask, rois, dense_size, 1.0, 0, 'avg', True).permute(0, 2, 3, 1)
        weights = mask.squeeze(-1) > eps
        targets[weights] /= mask[weights]
        targets = targets.permute(0, 3, 1, 2)
        weights = weights.unsqueeze(1)
    else:
        targets = pos_proposals.new_zeros((0, 3) + dense_size)
        weights = pos_proposals.new_zeros((0, 1) + dense_size)
    return targets, weights


def masked_dense_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_dense_list, gt_mask_list, cfg, eps=0.0001):
    targets, weights = zip(*map(lambda a, b, c, d: masked_dense_target_single(a, b, c, d, cfg, eps=eps), pos_proposals_list, pos_assigned_gt_inds_list, gt_dense_list, gt_mask_list))
    targets = torch.cat(targets, dim=0)
    weights = torch.cat(weights, dim=0)
    weights_mean = torch.mean(weights)
    weights = weights / weights_mean.clamp(min=eps)
    return targets, weights


def build_dim_coder(cfg, **default_args):
    return build_from_cfg(cfg, DIM_CODERS, default_args)


class Dropout2d(nn.Dropout2d):

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class Dropout(nn.Dropout):

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


PI = math.pi


def get_devicendarray(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.devices.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel() * 4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [(i * 4) for i in t.stride()], np.dtype('float32'), gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)


def bbox_rotate_overlaps_aligned_torch(boxes, qboxes, criterion=-1):
    n = boxes.shape[0]
    iou = boxes.new_empty([n])
    if n == 0:
        return iou
    threads_per_block = 64
    blockspergrid = div_up(n, threads_per_block)
    boxes_dev = get_devicendarray(boxes.flatten())
    qboxes_dev = get_devicendarray(qboxes.flatten())
    iou_dev = get_devicendarray(iou)
    rotate_iou_kernel_eval_aligned[blockspergrid, threads_per_block](n, boxes_dev, qboxes_dev, iou_dev, criterion)
    cuda.synchronize()
    return iou


def bev_to_box3d_overlaps_aligned_torch(boxes, qboxes, rinc, criterion=-1, z_axis=1, z_center=1.0):
    if boxes.shape[0] == 0:
        return boxes.new_zeros(size=(0,))
    min_z = torch.min(boxes[:, z_axis] + boxes[:, z_axis + 3] * (1 - z_center), qboxes[:, z_axis] + qboxes[:, z_axis + 3] * (1 - z_center))
    max_z = torch.min(boxes[:, z_axis] - boxes[:, z_axis + 3] * z_center, qboxes[:, z_axis] - qboxes[:, z_axis + 3] * z_center)
    iw = (min_z - max_z).clamp(min=0)
    volumn1 = torch.prod(boxes[:, 3:6], dim=1)
    volumn2 = torch.prod(qboxes[:, 3:6], dim=1)
    inc = iw * rinc
    if criterion == -1:
        ua = volumn1 + volumn2 - inc
    elif criterion == 0:
        ua = volumn1
    elif criterion == 1:
        ua = volumn2
    else:
        ua = 1.0
    iou = inc / ua.clamp(min=1e-06)
    return iou.clamp(min=0.0, max=1.0)


def bbox3d_overlaps_aligned_torch(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """
    Args:
        boxes (Tensor): (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        qboxes (Tensor):  (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        criterion: -1 for iou
        z_axis:
        z_center: 1.0 for bottom origin and 0.0 for top origin

    Returns:
        Tensor: (N, 1), ious
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    with torch.no_grad():
        rinc = bbox_rotate_overlaps_aligned_torch(boxes[:, bev_axes], qboxes[:, bev_axes], criterion=2)
        iou = bev_to_box3d_overlaps_aligned_torch(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return iou


def build_pnp(cfg, **default_args):
    return build_from_cfg(cfg, PNP, default_args)


def build_rotation_coder(cfg, **default_args):
    return build_from_cfg(cfg, ROTATION_CODERS, default_args)


def build_proj_error_coder(cfg, **default_args):
    return build_from_cfg(cfg, PROJ_ERROR_CODERS, default_args)


class BatchNormSmooth1D(_NormBase):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormSmooth1D, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        if bn_training and input.size(0) > 1:
            var, mean = torch.var_mean(input, dim=0)
            self.running_mean *= 1 - exponential_average_factor
            self.running_mean += exponential_average_factor * mean
            self.running_var *= 1 - exponential_average_factor
            self.running_var += exponential_average_factor * var
        out = input.sub(self.running_mean).div((self.running_var + self.eps).sqrt()).mul(self.weight).add(self.bias)
        return out

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))


def build_iou3d_sampler(cfg, **default_args):
    return build_from_cfg(cfg, IOU3D_SAMPLERS, default_args)


def forward_proj(coords_2d, coords_3d, cam_mats, z_min, u_range, v_range, yaw, t_vec):
    bn = coords_2d.shape[0]
    sin_yaw = torch.sin(yaw).squeeze(1)
    cos_yaw = torch.cos(yaw).squeeze(1)
    rot_mat = cos_yaw.new_zeros((bn, 3, 3))
    rot_mat[:, 0, 0] = cos_yaw
    rot_mat[:, 2, 2] = cos_yaw
    rot_mat[:, 0, 2] = sin_yaw
    rot_mat[:, 2, 0] = -sin_yaw
    rot_mat[:, 1, 1] = 1
    k_r = torch.matmul(cam_mats, rot_mat)
    k_t = torch.matmul(cam_mats, t_vec.unsqueeze(2)).squeeze(2)
    uvz = torch.einsum('bux,bnx->bnu', k_r, coords_3d) + k_t.unsqueeze(1)
    uv, z = uvz.split([2, 1], dim=2)
    z_clip_mask = z < z_min
    z[z_clip_mask] = z_min
    uv /= z
    u_range = u_range.unsqueeze(1)
    v_range = v_range.unsqueeze(1)
    uv_lb = torch.stack((u_range[..., 0], v_range[..., 0]), dim=2)
    uv_ub = torch.stack((u_range[..., 1], v_range[..., 1]), dim=2)
    uv_clip_mask_lb = uv < uv_lb
    uv_clip_mask_ub = uv > uv_ub
    uv_clip_mask = uv_clip_mask_lb | uv_clip_mask_ub
    uv = torch.max(uv_lb, torch.min(uv_ub, uv))
    error_unweighted = uv - coords_2d
    return uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw, error_unweighted, k_r


def get_pose_jacobians(uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw, inlier_mask, cam_mats, coords_2d_istd, coords_3d):
    if inlier_mask is not None:
        outlier_mask = ~inlier_mask
        zero_mask = z_clip_mask | uv_clip_mask | outlier_mask[..., None]
    else:
        outlier_mask = None
        zero_mask = z_clip_mask | uv_clip_mask
    jac_t_vec_xy = cam_mats[:, None, :2, :2] / z.unsqueeze(3)
    jac_t_vec_z = (cam_mats[:, None, :2, 2:3] - uv.unsqueeze(3)) / z.unsqueeze(3)
    jac_t_vec = torch.cat((jac_t_vec_xy, jac_t_vec_z), dim=3)
    jac_t_vec *= coords_2d_istd.unsqueeze(3)
    jac_t_vec[zero_mask] = 0
    jac_yaw_m1_l = cam_mats[:, 0:2, [0, 2]]
    jac_yaw_m1_r = torch.stack([torch.stack([-sin_yaw, cos_yaw], dim=1), torch.stack([-cos_yaw, -sin_yaw], dim=1)], dim=1)
    jac_yaw_m1 = torch.matmul(jac_yaw_m1_l, jac_yaw_m1_r)
    jac_yaw_m2 = torch.einsum('bnu,bx->bnux', uv, torch.stack([cos_yaw, sin_yaw], dim=1))
    jac_yaw_m = jac_yaw_m1.unsqueeze(1) + jac_yaw_m2
    jac_yaw = torch.einsum('bnux,bnx->bnu', jac_yaw_m, coords_3d[..., [0, 2]]) / z
    jac_yaw *= coords_2d_istd
    jac_yaw[zero_mask] = 0
    jac_yaw.unsqueeze_(3)
    return jac_t_vec, jac_yaw, zero_mask, outlier_mask


def get_jacobian_and_error(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, yaw, t_vec, inlier_mask):
    uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw, error_unweighted, k_r = forward_proj(coords_2d, coords_3d, cam_mats, z_min, u_range, v_range, yaw, t_vec)
    jac_t_vec, jac_yaw, _, _ = get_pose_jacobians(uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw, inlier_mask, cam_mats, coords_2d_istd, coords_3d)
    error = error_unweighted * coords_2d_istd
    return jac_t_vec, jac_yaw, error


def approx_hessian(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, yaw, t_vec, inlier_mask):
    bn, pn = coords_2d.shape[0:2]
    jac_t_vec, jac_yaw, error = get_jacobian_and_error(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, yaw, t_vec, inlier_mask)
    jac_pose = torch.cat((jac_yaw, jac_t_vec), dim=3).view(bn, -1, 4)
    h = torch.matmul(jac_pose.permute(0, 2, 1), jac_pose)
    return h


def exact_hessian(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, yaw, t_vec, inlier_mask):
    bn, pn = coords_2d.shape[0:2]
    coords_2d_ = coords_2d.detach().repeat(1, 4, 1).view(bn * 4, pn, 2)
    coords_2d_istd_ = coords_2d_istd.detach().repeat(1, 4, 1).view(bn * 4, pn, 2)
    coords_3d_ = coords_3d.detach().repeat(1, 4, 1).view(bn * 4, pn, 3)
    if cam_mats.size(0) == 1 < bn:
        cam_mats_ = cam_mats.detach().clone()
    else:
        cam_mats_ = cam_mats.detach().repeat(1, 4, 1).view(bn * 4, 3, 3)
    if u_range.size(0) == 1 < bn:
        u_range_ = u_range
    else:
        u_range_ = u_range.repeat(1, 4).view(bn * 4, 2)
    if v_range.size(0) == 1 < bn:
        v_range_ = v_range
    else:
        v_range_ = v_range.repeat(1, 4).view(bn * 4, 2)
    inlier_mask_ = inlier_mask.detach().repeat(1, 4).view(bn * 4, pn) if inlier_mask is not None else None
    torch.set_grad_enabled(True)
    pose = torch.cat([yaw.detach(), t_vec.detach()], dim=1).repeat(1, 4).view(bn * 4, 4).requires_grad_()
    yaw_ = pose[:, :1]
    t_vec_ = pose[:, 1:]
    jac_t_vec, jac_yaw, error = get_jacobian_and_error(coords_2d_, coords_2d_istd_, coords_3d_, cam_mats_, u_range_, v_range_, z_min, yaw_, t_vec_, inlier_mask_)
    jac_pose = torch.cat((jac_yaw, jac_t_vec), dim=3).view(bn * 4, -1, 4)
    error = error.view(bn * 4, -1, 1)
    jt_error = torch.matmul(jac_pose.permute(0, 2, 1), error)
    jac = jt_error.view(-1, 4, 4)
    sum_grad_jac = torch.diagonal(jac, dim1=1, dim2=2).sum()
    h = torch.autograd.grad(sum_grad_jac, pose)[0].view(-1, 4, 4)
    return h


def u2d_pnp_cpu_single(coord_2d, coord_2d_istd, coord_3d, istd_inlier_mask, cam_mat, u_range, v_range, epnp_ransac_thres, inlier_opt_only=False, z_min=0.5, dist_coeffs=None, with_pose_cov=True):
    pn = coord_2d.shape[0]
    istd_inlier_count = np.count_nonzero(istd_inlier_mask)
    if istd_inlier_count > 4:
        coord_3d_inlier = coord_3d[istd_inlier_mask]
        coord_2d_inlier = coord_2d[istd_inlier_mask]
        coord_2d_istd_inlier = coord_2d_istd[istd_inlier_mask]
    else:
        coord_3d_inlier = coord_3d
        coord_2d_inlier = coord_2d
        coord_2d_istd_inlier = coord_2d_istd
        istd_inlier_mask[:] = True
    if epnp_ransac_thres is not None:
        ret_val, r_vec, t_vec, ransac_inlier_ind = cv2.solvePnPRansac(coord_3d_inlier, coord_2d_inlier, cam_mat, dist_coeffs, reprojectionError=epnp_ransac_thres, iterationsCount=30, flags=cv2.SOLVEPNP_EPNP)
        if ransac_inlier_ind is not None and len(ransac_inlier_ind) > 4:
            ransac_inlier_ind = ransac_inlier_ind.squeeze(1)
            ransac_inlier_mask = np.zeros(coord_3d_inlier.shape[0], dtype=np.bool)
            ransac_inlier_mask[ransac_inlier_ind] = True
            coord_3d_inlier = coord_3d_inlier[ransac_inlier_ind]
            coord_2d_inlier = coord_2d_inlier[ransac_inlier_ind]
            coord_2d_istd_inlier = coord_2d_istd_inlier[ransac_inlier_ind]
            istd_inlier_mask[istd_inlier_mask] = ransac_inlier_mask
    else:
        ret_val, r_vec, t_vec = cv2.solvePnP(coord_3d_inlier, coord_2d_inlier, cam_mat, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    inlier_mask = istd_inlier_mask
    if ret_val:
        if inlier_opt_only:
            coord_3d = coord_3d_inlier
            coord_2d = coord_2d_inlier
            coord_2d_istd = coord_2d_istd_inlier
            pn = coord_2d.shape[0]
        yaw = r_vec[1:2]
        clips = np.array([z_min, u_range[0], u_range[1], v_range[0], v_range[1]], np.float64)
        coord_2d = np.ascontiguousarray(coord_2d, np.float64)
        coord_3d = np.ascontiguousarray(coord_3d, np.float64)
        coord_2d_istd = np.ascontiguousarray(coord_2d_istd, np.float64)
        cam_mat = np.ascontiguousarray(cam_mat, np.float64)
        init_pose = np.ascontiguousarray(np.concatenate([yaw, t_vec], axis=0), np.float64)
        clips = np.ascontiguousarray(clips, np.float64)
        ceres_dtype = 'double*'
        coord_2d_ptr = ffi.cast(ceres_dtype, coord_2d.ctypes.data)
        coord_3d_ptr = ffi.cast(ceres_dtype, coord_3d.ctypes.data)
        coord_2d_istd_ptr = ffi.cast(ceres_dtype, coord_2d_istd.ctypes.data)
        cam_mat_ptr = ffi.cast(ceres_dtype, cam_mat.ctypes.data)
        init_pose_ptr = ffi.cast(ceres_dtype, init_pose.ctypes.data)
        clips_ptr = ffi.cast(ceres_dtype, clips.ctypes.data)
        result_val = np.zeros([1], np.int32)
        result_pose = np.zeros([4], np.float64)
        result_cov = np.eye(4, dtype=np.float64) if with_pose_cov else None
        result_tr = np.zeros([1], np.float64)
        result_val_ptr = ffi.cast('int*', result_val.ctypes.data)
        result_pose_ptr = ffi.cast(ceres_dtype, result_pose.ctypes.data)
        result_cov_ptr = ffi.cast(ceres_dtype, result_cov.ctypes.data) if with_pose_cov else ffi.NULL
        result_tr_ptr = ffi.cast(ceres_dtype, result_tr.ctypes.data)
        lib.pnp_uncert(coord_2d_ptr, coord_3d_ptr, coord_2d_istd_ptr, cam_mat_ptr, init_pose_ptr, result_val_ptr, result_pose_ptr, result_cov_ptr, result_tr_ptr, pn, clips_ptr)
        yaw_refined = result_pose[0:1].astype(np.float32)
        t_vec_refined = result_pose[1:].astype(np.float32)
        pose_cov = result_cov.astype(np.float32) if result_cov is not None else None
        tr_radius = result_tr.astype(np.float32)
        return result_val[0] > 0, yaw_refined, t_vec_refined, pose_cov, tr_radius, inlier_mask
    else:
        return False, np.zeros(1, np.float32), np.zeros(3, np.float32), np.eye(4, dtype=np.float32) if with_pose_cov else None, np.zeros(1, np.float32), inlier_mask


def u2d_pnp_cpu(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min=0.5, epnp_istd_thres=1.0, epnp_ransac_thres=None, inlier_opt_only=False, with_pose_cov=True):
    """
    Args:
        coords_2d (ndarray): shape (Nbatch, Npoint, 2)
        coords_2d_istd (ndarray): shape (Nbatch, Npoint, 2)
        coords_3d (ndarray): shape (Nbatch, Npoint, 3)
        cam_mats (ndarray): shape (Nbatch, 3, 3) or (1, 3, 3)
        u_range (ndarray): shape (Nbatch, 2) or (1, 2)
        v_range (ndarray): shape (Nbatch, 2) or (1, 2)
        z_min (float):
        epnp_istd_thres (float):
        epnp_ransac_thres (None | ndarray): shape (Nbatch, )
        inlier_opt_only (bool):

    Returns:
        ret_val (ndarray): shape (Nbatch, ), validity bool mask
        yaw (ndarray): shape (Nbatch, 1)
        t_vec (ndarray): shape (Nbatch, 3)
        pose_cov (ndarray): shape (Nbatch, 4, 4), covariance matrices
            of [yaw, t_vec]
        tr_radius (ndarray): shape (Nbatch, 1), trust region radius
        inlier_mask (ndarray): shape (Nbatch, Npoint), inlier bool mask
    """
    bn = coords_2d.shape[0]
    pn = coords_2d.shape[1]
    if bn > 0:
        assert coords_2d_istd.shape[1] == coords_3d.shape[1] == pn >= 4
        coord_2d_istd_mean = np.mean(coords_2d_istd, axis=1, keepdims=True)
        istd_inlier_masks = np.min(coords_2d_istd >= epnp_istd_thres * coord_2d_istd_mean, axis=2)
        if cam_mats.shape[0] == 1 < bn:
            cam_mats = [cam_mats.squeeze(0)] * bn
        if u_range.shape[0] == 1 < bn:
            u_range = [u_range.squeeze(0)] * bn
        if v_range.shape[0] == 1 < bn:
            v_range = [v_range.squeeze(0)] * bn
        if epnp_ransac_thres is None:
            epnp_ransac_thres = [None] * bn
        dist_coeffs = np.zeros((8, 1), dtype=np.float32)
        ret_val, yaw, t_vec, pose_cov, tr_radius, inlier_mask = multi_apply(u2d_pnp_cpu_single, coords_2d, coords_2d_istd, coords_3d, istd_inlier_masks, cam_mats, u_range, v_range, epnp_ransac_thres, inlier_opt_only=inlier_opt_only, z_min=z_min, dist_coeffs=dist_coeffs, with_pose_cov=with_pose_cov)
        ret_val = np.array(ret_val, dtype=np.bool)
        yaw = np.stack(yaw, axis=0)
        t_vec = np.stack(t_vec, axis=0)
        pose_cov = np.stack(pose_cov, axis=0) if with_pose_cov else None
        tr_radius = np.stack(tr_radius, axis=0)
        inlier_mask = np.stack(inlier_mask, axis=0)
    else:
        ret_val = np.zeros((0,), dtype=np.bool)
        yaw = np.zeros((0, 1), dtype=np.float32)
        t_vec = np.zeros((0, 3), dtype=np.float32)
        pose_cov = np.zeros((0, 4, 4), dtype=np.float32)
        tr_radius = np.zeros((0, 1), dtype=np.float32)
        inlier_mask = np.zeros((0, pn), dtype=np.bool)
    return ret_val, yaw, t_vec, pose_cov, tr_radius, inlier_mask


def pnp_uncert(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min=0.5, epnp_istd_thres=1.0, epnp_ransac_thres=None, inlier_opt_only=False, forward_exact_hessian=False, use_6dof=False):
    """
    Args:
        coords_2d (torch.Tensor): shape (Nbatch, Npoint, 2)
        coords_2d_istd (torch.Tensor): shape (Nbatch, Npoint, 2)
        coords_3d (torch.Tensor): shape (Nbatch, Npoint, 3)
        cam_mats (torch.Tensor): shape (Nbatch, 3, 3) or (1, 3, 3)
        u_range (torch.Tensor): shape (Nbatch, 2) or (1, 2)
        v_range (torch.Tensor): shape (Nbatch, 2) or (1, 2)
        z_min (float):
        epnp_istd_thres (float):
        epnp_ransac_thres (None | torch.Tensor): shape (Nbatch, )
        inlier_opt_only (bool):

    Returns:
        ret_val (Tensor): shape (Nbatch, ), validity bool mask
        r_vec (Tensor): shape (Nbatch, 1) or (Nbatch, 3)
        t_vec (Tensor): shape (Nbatch, 3)
        pose_cov (Tensor): shape (Nbatch, 4, 4), covariance matrices
            of [yaw, t_vec]
        inlier_mask (Tensor): shape (Nbatch, Npoint), inlier bool mask
    """
    with torch.no_grad():
        coords_2d_np = coords_2d.cpu().numpy()
        coords_2d_istd_np = coords_2d_istd.cpu().numpy()
        coords_3d_np = coords_3d.cpu().numpy()
        cam_mats_np = cam_mats.cpu().numpy()
        u_range_np = u_range.cpu().numpy()
        v_range_np = v_range.cpu().numpy()
        if epnp_ransac_thres is not None:
            epnp_ransac_thres_np = epnp_ransac_thres.cpu().numpy()
        else:
            epnp_ransac_thres_np = None
        ret_val, r_vec, t_vec, _, _, inlier_mask = u2d_pnp_cpu(coords_2d_np, coords_2d_istd_np, coords_3d_np, cam_mats_np, u_range_np, v_range_np, z_min=z_min, epnp_istd_thres=epnp_istd_thres, epnp_ransac_thres=epnp_ransac_thres_np, inlier_opt_only=inlier_opt_only, with_pose_cov=False)
        ret_val = coords_2d.new_tensor(ret_val, dtype=torch.bool)
        r_vec = coords_2d.new_tensor(r_vec)
        t_vec = coords_2d.new_tensor(t_vec)
        inlier_mask = coords_2d.new_tensor(inlier_mask, dtype=torch.bool)
        if ret_val.size(0) == 0:
            pose_cov = coords_2d.new_zeros((0, 4, 4))
        else:
            if forward_exact_hessian:
                h = exact_hessian(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, r_vec, t_vec, inlier_mask)
            else:
                h = approx_hessian(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min, r_vec, t_vec, inlier_mask)
            try:
                pose_cov = torch.inverse(h)
            except RuntimeError:
                eigval, _ = torch.symeig(h, eigenvectors=False)
                valid_mask = eigval[:, 0] > (1e-06 * eigval[:, 3]).clamp(min=0)
                ret_val &= valid_mask
                h[~ret_val] = torch.eye(4, device=h.device, dtype=h.dtype)
                pose_cov = torch.inverse(h)
    return ret_val, r_vec, t_vec, pose_cov, inlier_mask


class PnPUncert(torch.nn.Module):

    def __init__(self, z_min=0.5, epnp_istd_thres=0.6, inlier_opt_only=True, coord_istd_normalize=False, forward_exact_hessian=False, use_6dof=False, eps=1e-06):
        """Uncertainty-2D PnP v3.

        This algorithm uses the exact derivative of L-M iteration. Instead of
        computing the derivative matrix directly, this implementation computes
        the derivative of the product of output gradient w.r.t pose and L-M
        step using auto grad, which directly yields the output gradient w.r.t.
        PnP inputs.

        Args:
            z_min (float):
            epnp_istd_thres (float): points with istd greater than (thres
                * istd_mean) will be kept as inliers
            inlier_opt_only (bool): whether to use inliers or all points for
                non-linear optimization, note that this will affect back-
                propagation
        """
        super(PnPUncert, self).__init__()
        self.z_min = z_min
        self.epnp_istd_thres = epnp_istd_thres
        self.inlier_opt_only = inlier_opt_only
        self.coord_istd_normalize = coord_istd_normalize
        self.forward_exact_hessian = forward_exact_hessian
        self.use_6dof = use_6dof
        self.eps = eps

    def forward(self, coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, epnp_ransac_thres=None):
        if self.coord_istd_normalize:
            mean = torch.mean(coords_2d_istd, dim=(1, 2), keepdim=True)
            coords_2d_istd = coords_2d_istd / mean.clamp(min=self.eps)
        return pnp_uncert(coords_2d, coords_2d_istd, coords_3d, cam_mats, u_range, v_range, z_min=self.z_min, epnp_istd_thres=self.epnp_istd_thres, epnp_ransac_thres=epnp_ransac_thres, inlier_opt_only=self.inlier_opt_only, forward_exact_hessian=self.forward_exact_hessian, use_6dof=self.use_6dof)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BatchNormSmooth1D,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Dropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Dropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

