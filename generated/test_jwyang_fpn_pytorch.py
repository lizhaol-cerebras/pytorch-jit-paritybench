
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


import time


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.optim as optim


import torchvision.transforms as transforms


import torchvision.datasets as dset


import random


import torch.nn.functional as F


from torch.autograd import gradcheck


from torch.autograd.gradcheck import gradgradcheck


import torchvision.models as models


import torchvision.utils as vutils


import math


import torch.utils.model_zoo as model_zoo


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import numpy.random as npr


import scipy


import torch.utils.data as data


import logging


from torch.utils.data.sampler import Sampler


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois, output)
        else:
            raise NotImplementedError
        return output

    def backward(self, grad_output):
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, grad_output, self.rois, grad_input)
        return grad_input, None


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois, scale):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, 2] - anchors[:, :, 0] + 1
        anchors_boxes_y = anchors[:, :, 3] - anchors[:, :, 1] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)
        iw = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) - torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)
        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, 2] - anchors[:, :, 0] + 1
        anchors_boxes_y = anchors[:, :, 3] - anchors[:, :, 1] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)
        iw = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) - torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')
    return overlaps


def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)
    return targets


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        labels, rois, gt_assign, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, self._num_classes)
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return rois, labels, gt_assign, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        for b in range(batch_size):
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4
        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        targets = bbox_transform_batch(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            targets = (targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)) / self.BBOX_NORMALIZE_STDS.expand_as(targets)
        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = gt_boxes[:, :, 4].contiguous().view(-1).index(offset.view(-1)).view(batch_size, -1)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_assign_batch = all_rois.new(batch_size, rois_per_image).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError('bg_num_rois = 0 and fg_num_rois = 0, this should not happen!')
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            labels_batch[i].copy_(labels[i][keep_inds])
            labels_batch[i][fg_rois_per_this_image:] = 0
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i
            gt_assign_batch[i] = gt_assignment[i][keep_inds]
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])
        bbox_targets, bbox_inside_weights = self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        return labels_batch, rois_batch, gt_assign_batch, bbox_targets, bbox_inside_weights


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_anchors_all_pyramids(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors_single_pyramid(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


class _AnchorTargetLayer_FPN(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer_FPN, self).__init__()
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE
        self._allowed_border = 0

    def forward(self, input):
        scores = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]
        feat_shapes = input[4]
        height, width = 0, 0
        batch_size = gt_boxes.size(0)
        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(scores)
        total_anchors = anchors.size(0)
        keep = (anchors[:, 0] >= -self._allowed_border) & (anchors[:, 1] >= -self._allowed_border) & (anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) & (anchors[:, 3] < long(im_info[0][0]) + self._allowed_border)
        inds_inside = torch.nonzero(keep).view(-1)
        anchors = anchors[inds_inside, :]
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-05
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        for i in range(batch_size):
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples
            negative_weights = 1.0 / num_examples
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)
        outputs = []
        outputs.append(labels)
        outputs.append(bbox_targets)
        outputs.append(bbox_inside_weights)
        outputs.append(bbox_outside_weights)
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_boxes = deltas.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    return boxes

