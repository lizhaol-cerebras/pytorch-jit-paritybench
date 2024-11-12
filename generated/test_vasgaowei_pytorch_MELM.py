
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


import numpy.random as npr


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Function


from torch import nn


from torch.nn.modules.module import Module


import copy


from torch.nn.parameter import Parameter


import torch.autograd.gradcheck


from copy import deepcopy


import torch.optim as optim


import time


from torch.autograd import Variable


from collections import namedtuple


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


import torchvision


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


import torchvision.models as models


import matplotlib.pyplot as plt


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)
        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)
        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(grad_outputs, boxes, box_ind, grad_image)
        else:
            _backend.crop_and_resize_backward(grad_outputs, boxes, box_ind, grad_image)
        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]
        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)
            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)
            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)
        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(featuremap, boxes, box_ind)


class RoIPoolFunction(Function):

    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, features, rois, output, ctx.argmax)
        return output

    def backward(ctx, grad_output):
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, grad_output, ctx.rois, grad_input, ctx.argmax)
        return grad_input, None


class _RoIPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)


def RectangularRing(rois, processed_rois, spatial_scale, scale_inner, scale_outer):
    ctr_x = (rois[:, 1] + rois[:, 3]) / 2
    ctr_y = (rois[:, 2] + rois[:, 4]) / 2
    w_half = (rois[:, 3] - rois[:, 1]) / 2
    h_half = (rois[:, 4] - rois[:, 2]) / 2
    processed_rois[:, 1] = torch.tensor(ctr_x - w_half * scale_outer, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 2] = torch.tensor(ctr_y - h_half * scale_outer, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 3] = torch.tensor(ctr_x + w_half * scale_outer, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 4] = torch.tensor(ctr_y + h_half * scale_outer, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 5] = torch.tensor(ctr_x - w_half * scale_inner, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 6] = torch.tensor(ctr_y - h_half * scale_inner, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 7] = torch.tensor(ctr_x + w_half * scale_inner, dtype=rois.dtype, device=rois.device)
    processed_rois[:, 8] = torch.tensor(ctr_y + h_half * scale_inner, dtype=rois.dtype, device=rois.device)
    if scale_inner == 0:
        processed_rois[:, 5:] = 0
    return 1


class RoIRingPoolFunction(Function):

    def __init__(ctx, pooled_height, pooled_width, spatial_scale, scale_inner, scale_outer):
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.scale_inner = scale_inner
        ctx.scale_outer = scale_outer
        ctx.feature_size = None

    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        ctx.processed_rois = features.new(rois.size(0), 9).zero_()
        RectangularRing(rois, ctx.processed_rois, ctx.spatial_scale, ctx.scale_inner, ctx.scale_outer)
        roi_ring_pooling.roi_ring_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, features, ctx.processed_rois, output, ctx.argmax)
        return output

    def backward(ctx, grad_output):
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_ring_pooling.roi_ring_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, grad_output, ctx.processed_rois, grad_input, ctx.argmax)
        return grad_input, None


class _RoIRingPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale, scale_inner, scale_outer):
        super(_RoIRingPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer

    def forward(self, features, rois):
        return RoIRingPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.scale_inner, self.scale_outer)(features, rois)


class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_fc7, input_refine3, weight, bias=None):
        ctx.save_for_backward(input_fc7, input_refine3, weight, bias)
        output = (input_fc7 * input_refine3).mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fc7, input_refine3, weight, bias = ctx.saved_tensors
        grad_input_fc7 = grad_input_refine3 = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input_fc7 = grad_output.mm(weight) * input_refine3
        if ctx.needs_input_grad[1]:
            grad_input_refine3 = None
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(input_fc7 * input_refine3)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input_fc7, None, grad_weight, grad_bias


class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_fc7, recurrent_refine):
        return LinearFunction.apply(input_fc7, recurrent_refine, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class RecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, refine, alpha):
        ctx.save_for_backward(input, refine, alpha)
        output = input * (1 - alpha + alpha * refine)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, refine, alpha = ctx.saved_tensors
        grad_input = grad_refine = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (1 - alpha + alpha * refine)
        if ctx.needs_input_grad[1]:
            grad_refine = None
        if ctx.needs_input_grad[2]:
            grad_alpha = None
        return grad_input, grad_refine, grad_alpha


class RecurrentLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(RecurrentLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, fc7, recurrent_refine, alpha):
        return RecurrentFunction.apply(fc7, recurrent_refine, alpha)


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)
    return targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = (targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    return torch.cat([labels.unsqueeze(1), targets], 1)


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()
    else:
        out_fn = lambda x: x
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors
    _allowed_border = 0
    height, width = rpn_cls_score.shape[1:3]
    inds_inside = np.where((all_anchors[:, 0] >= -_allowed_border) & (all_anchors[:, 1] >= -_allowed_border) & (all_anchors[:, 2] < im_info[1] + _allowed_border) & (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]
    anchors = all_anchors[inds_inside, :]
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
        labels[disable_inds] = -1
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=len(bg_inds) - num_bg, replace=False)
        labels[disable_inds] = -1
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
        positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
        negative_weights = (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
    rpn_bbox_targets = bbox_targets
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


STANDARD_COLORS = ['AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen']


NUM_COLORS = len(STANDARD_COLORS)


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
    return image


def draw_bounding_boxes(image, gt_boxes, im_info):
    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new[:, :4] = np.round(gt_boxes_new[:, :4].copy() / im_info[2])
    disp_image = Image.fromarray(np.uint8(image[0]))
    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        disp_image = _draw_single_box(disp_image, gt_boxes_new[i, 0], gt_boxes_new[i, 1], gt_boxes_new[i, 2], gt_boxes_new[i, 3], 'N%02d-C%02d' % (i, this_class), FONT, color=STANDARD_COLORS[this_class % NUM_COLORS])
    image[0, :] = np.array(disp_image)
    return image


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _whctrs(anchor):
    """
  Return width, height, x center, and y center for an anchor (window).
  """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])
    return anchors, length


def bbox_transform_inv(boxes, deltas):
    if len(boxes) == 0:
        return deltas.detach() * 0
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)
    pred_boxes = torch.cat([_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
  Clip boxes to image boundaries.
  boxes must be tensor or Variable, im_shape can be anything but Variable
  """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()
    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([boxes[:, :, 0].clamp(0, im_shape[1] - 1), boxes[:, :, 1].clamp(0, im_shape[0] - 1), boxes[:, :, 2].clamp(0, im_shape[1] - 1), boxes[:, :, 3].clamp(0, im_shape[0] - 1)], 2).view(boxes.size(0), -1)
    return boxes

