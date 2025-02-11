
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


from collections import OrderedDict


from typing import Optional


from typing import Tuple


from typing import Sequence


import numpy as np


import torch.nn as nn


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import remove_small_boxes


from typing import Dict


from typing import List


import torch.utils.data as data


import torch.utils.data


import random


import math


from copy import deepcopy


import functools


import logging


import torch.distributed as dist


from functools import partial


from typing import Callable


from typing import Union


import torch.nn.functional as F


import abc


import time


import warnings


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


import torch.nn.parallel


def get_feat_sizes(image_size: 'Tuple[int, int]', max_level: 'int'):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = (feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1
        feat_sizes.append(feat_size)
    return feat_sizes


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size: 'Tuple[int, int]'):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: Sequence specifying input image size of model (H, W).
                The image_size should be divided by the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        if isinstance(anchor_scale, Sequence):
            assert len(anchor_scale) == max_level - min_level + 1
            self.anchor_scales = anchor_scale
        else:
            self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)
        assert isinstance(image_size, Sequence) and len(image_size) == 2
        self.image_size = tuple(image_size)
        self.feat_sizes = get_feat_sizes(image_size, max_level)
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    @classmethod
    def from_config(cls, config):
        return cls(config.min_level, config.max_level, config.num_scales, config.aspect_ratios, config.anchor_scale, config.image_size)

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(((feat_sizes[0][0] / float(feat_sizes[level][0]), feat_sizes[0][1] / float(feat_sizes[level][1])), scale_octave / float(self.num_scales), aspect, self.anchor_scales[level - self.min_level]))
        return anchor_configs

    def _generate_boxes(self):
        """Generates multi-scale anchor boxes."""
        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = []
            for config in configs:
                stride, octave_scale, aspect, anchor_scale = config
                base_anchor_size_x = anchor_scale * stride[1] * 2 ** octave_scale
                base_anchor_size_y = anchor_scale * stride[0] * 2 ** octave_scale
                if isinstance(aspect, Sequence):
                    aspect_x, aspect_y = aspect
                else:
                    aspect_x = np.sqrt(aspect)
                    aspect_y = 1.0 / aspect_x
                anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
                anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0
                x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
                y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).float()
        return anchor_boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


def pairwise_iou(boxes1, boxes2) ->torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter), torch.zeros(1, dtype=inter.dtype, device=inter.device))
    return iou


def soft_nms(boxes, scores, method_gaussian: 'bool'=True, sigma: 'float'=0.5, iou_threshold: 'float'=0.5, score_threshold: 'float'=0.005):
    """
    Soft non-max suppression algorithm.

    Implementation of [Soft-NMS -- Improving Object Detection With One Line of Codec]
    (https://arxiv.org/abs/1704.04503)

    Args:
        boxes_remain (Tensor[N, ?]):
           boxes where NMS will be performed
           if Boxes, in (x1, y1, x2, y2) format
           if RotatedBoxes, in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores_remain (Tensor[N]):
           scores for each one of the boxes
        method_gaussian (bool): use gaussian method if True, otherwise linear        
        sigma (float):
           parameter for Gaussian penalty function
        iou_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        score_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]

    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    device = boxes.device
    boxes_remain = boxes.clone()
    scores_remain = scores.clone()
    num_elem = scores_remain.size()[0]
    idxs = torch.arange(num_elem)
    idxs_out = torch.zeros(num_elem, dtype=torch.int64, device=device)
    scores_out = torch.zeros(num_elem, dtype=torch.float32, device=device)
    count: 'int' = 0
    while scores_remain.numel() > 0:
        top_idx = torch.argmax(scores_remain)
        idxs_out[count] = idxs[top_idx]
        scores_out[count] = scores_remain[top_idx]
        count += 1
        top_box = boxes_remain[top_idx]
        ious = pairwise_iou(top_box.unsqueeze(0), boxes_remain)[0]
        if method_gaussian:
            decay = torch.exp(-torch.pow(ious, 2) / sigma)
        else:
            decay = torch.ones_like(ious)
            decay_mask = ious > iou_threshold
            decay[decay_mask] = 1 - ious[decay_mask]
        scores_remain *= decay
        keep = scores_remain > score_threshold
        keep[top_idx] = torch.tensor(False, device=device)
        boxes_remain = boxes_remain[keep]
        scores_remain = scores_remain[keep]
        idxs = idxs[keep]
    return idxs_out[:count], scores_out[:count]


def batched_soft_nms(boxes, scores, idxs, method_gaussian: 'bool'=True, sigma: 'float'=0.5, iou_threshold: 'float'=0.5, score_threshold: 'float'=0.001):
    """
    Performs soft non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]):
           boxes where NMS will be performed. They
           are expected to be in (x1, y1, x2, y2) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        sigma (float):
           parameter for Gaussian penalty function
        iou_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        score_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device), torch.empty((0,), dtype=torch.float32, device=scores.device)
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return soft_nms(boxes_for_nms, scores, method_gaussian=method_gaussian, sigma=sigma, iou_threshold=iou_threshold, score_threshold=score_threshold)


def clip_boxes_xyxy(boxes: 'torch.Tensor', size: 'torch.Tensor'):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size)
    return boxes


def decode_box_outputs(rel_codes, anchors, output_xyxy: 'bool'=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[:, 0] + anchors[:, 2]) / 2
    xcenter_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]
    ty, tx, th, tw = rel_codes.unbind(dim=1)
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


def generate_detections(cls_outputs, box_outputs, anchor_boxes, indices, classes, img_scale: 'Optional[torch.Tensor]', img_size: 'Optional[torch.Tensor]', max_det_per_image: 'int'=100, soft_nms: 'bool'=False):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels.

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [max_det_per_image, 6],
            each row representing [x_min, y_min, x_max, y_max, score, class]
    """
    assert box_outputs.shape[-1] == 4
    assert anchor_boxes.shape[-1] == 4
    assert cls_outputs.shape[-1] == 1
    anchor_boxes = anchor_boxes[indices, :]
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
    if img_scale is not None and img_size is not None:
        boxes = clip_boxes_xyxy(boxes, img_size / img_scale)
    scores = cls_outputs.sigmoid().squeeze(1).float()
    if soft_nms:
        top_detection_idx, soft_scores = batched_soft_nms(boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=0.001)
        scores[top_detection_idx] = soft_scores
    else:
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)
    top_detection_idx = top_detection_idx[:max_det_per_image]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx, None]
    classes = classes[top_detection_idx, None] + 1
    if img_scale is not None:
        boxes = boxes * img_scale
    num_det = len(top_detection_idx)
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if num_det < max_det_per_image:
        detections = torch.cat([detections, torch.zeros((max_det_per_image - num_det, 6), device=detections.device, dtype=detections.dtype)], dim=0)
    return detections


@torch.jit.script
def _batch_detection(batch_size: 'int', class_out, box_out, anchor_boxes, indices, classes, img_scale: 'Optional[torch.Tensor]'=None, img_size: 'Optional[torch.Tensor]'=None, max_det_per_image: 'int'=100, soft_nms: 'bool'=False):
    batch_detections = []
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(class_out[i], box_out[i], anchor_boxes, indices[i], classes[i], img_scale_i, img_size_i, max_det_per_image=max_det_per_image, soft_nms=soft_nms)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


def _post_process(cls_outputs: 'List[torch.Tensor]', box_outputs: 'List[torch.Tensor]', num_levels: 'int', num_classes: 'int', max_detection_points: 'int'=5000):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].

        num_levels (int): number of feature levels

        num_classes (int): number of output classes
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, num_classes]) for level in range(num_levels)], 1)
    box_outputs_all = torch.cat([box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4]) for level in range(num_levels)], 1)
    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes
    box_outputs_all_after_topk = torch.gather(box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))
    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


class DetBenchPredict(nn.Module):

    def __init__(self, model):
        super(DetBenchPredict, self).__init__()
        self.model = model
        self.config = model.config
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms

    def forward(self, x, img_info: 'Optional[Dict[str, torch.Tensor]]'=None):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(class_out, box_out, num_levels=self.num_levels, num_classes=self.num_classes, max_detection_points=self.max_detection_points)
        if img_info is None:
            img_scale, img_size = None, None
        else:
            img_scale, img_size = img_info['img_scale'], img_info['img_size']
        return _batch_detection(x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes, img_scale, img_size, max_det_per_image=self.max_det_per_image, soft_nms=self.soft_nms)


def one_hot_bool(x, num_classes: 'int'):
    onehot = torch.zeros(x.size(0), num_classes, device=x.device, dtype=torch.bool)
    return onehot.scatter_(1, x.unsqueeze(1), 1)


EPS = 1e-07


class FasterRcnnBoxCoder(object):
    """Faster RCNN box coder."""

    def __init__(self, scale_factors: 'Optional[List[float]]'=None, eps: 'float'=EPS):
        """Constructor for FasterRcnnBoxCoder.

        Args:
            scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
                If set to None, does not perform scaling. For Faster RCNN,
                the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
        """
        self._scale_factors = scale_factors
        if scale_factors is not None:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self.eps = eps

    def code_size(self):
        return 4

    def encode(self, boxes: 'BoxList', anchors: 'BoxList'):
        """Encode a box collection with respect to anchor collection.

        Args:
            boxes: BoxList holding N boxes to be encoded.
            anchors: BoxList of anchors.

        Returns:
            a tensor representing N anchor-encoded boxes of the format [ty, tx, th, tw].
        """
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
        ha += self.eps
        wa += self.eps
        h += self.eps
        w += self.eps
        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = torch.log(w / wa)
        th = torch.log(h / ha)
        if self._scale_factors is not None:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return torch.stack([ty, tx, th, tw]).t()

    def decode(self, rel_codes, anchors: 'BoxList'):
        """Decode relative codes to boxes.

        Args:
            rel_codes: a tensor representing N anchor-encoded boxes.
            anchors: BoxList of anchors.

        Returns:
            boxes: BoxList holding N bounding boxes.
        """
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ty, tx, th, tw = rel_codes.t().unbind()
        if self._scale_factors is not None:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0
        return BoxList(torch.stack([ymin, xmin, ymax, xmax]).t())


def area(boxlist: 'BoxList'):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes

    Returns:
        a tensor with shape [N] representing box areas.
    """
    y_min, x_min, y_max, x_max = boxlist.boxes().chunk(4, dim=1)
    out = (y_max - y_min).squeeze(1) * (x_max - x_min).squeeze(1)
    return out


def intersection(boxlist1: 'BoxList', boxlist2: 'BoxList'):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    y_min1, x_min1, y_max1, x_max1 = boxlist1.boxes().chunk(4, dim=1)
    y_min2, x_min2, y_max2, x_max2 = boxlist2.boxes().chunk(4, dim=1)
    all_pairs_min_ymax = torch.min(y_max1, y_max2.t())
    all_pairs_max_ymin = torch.max(y_min1, y_min2.t())
    intersect_heights = torch.clamp(all_pairs_min_ymax - all_pairs_max_ymin, min=0)
    all_pairs_min_xmax = torch.min(x_max1, x_max2.t())
    all_pairs_max_xmin = torch.max(x_min1, x_min2.t())
    intersect_widths = torch.clamp(all_pairs_min_xmax - all_pairs_max_xmin, min=0)
    return intersect_heights * intersect_widths


def iou(boxlist1: 'BoxList', boxlist2: 'BoxList'):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxlist1: BoxList holding N boxes
        boxlist2: BoxList holding M boxes

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersections
    return torch.where(intersections == 0.0, torch.zeros_like(intersections), intersections / unions)


KEYPOINTS_FIELD_NAME = 'keypoints'


class TargetAssigner(object):
    """Target assigner to compute classification and regression targets."""

    def __init__(self, similarity_calc: 'IouSimilarity', matcher: 'ArgMaxMatcher', box_coder: 'FasterRcnnBoxCoder', negative_class_weight: 'float'=1.0, unmatched_cls_target: 'Optional[float]'=None, keypoints_field_name: 'str'=KEYPOINTS_FIELD_NAME):
        """Construct Object Detection Target Assigner.

        Args:
            similarity_calc: a RegionSimilarityCalculator

            matcher: Matcher used to match groundtruth to anchors.

            box_coder: BoxCoder used to encode matching groundtruth boxes with respect to anchors.

            negative_class_weight: classification weight to be associated to negative
                anchors (default: 1.0). The weight must be in [0., 1.].

            unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
                which is consistent with the classification target for each
                anchor (and can be empty for scalar targets).  This shape must thus be
                compatible with the groundtruth labels that are passed to the "assign"
                function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
                If set to None, unmatched_cls_target is set to be [0] for each anchor.

        Raises:
            ValueError: if similarity_calc is not a RegionSimilarityCalculator or
                if matcher is not a Matcher or if box_coder is not a BoxCoder
        """
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight
        if unmatched_cls_target is not None:
            self._unmatched_cls_target = unmatched_cls_target
        else:
            self._unmatched_cls_target = 0.0
        self._keypoints_field_name = keypoints_field_name

    def assign(self, anchors: 'BoxList', groundtruth_boxes: 'BoxList', groundtruth_labels=None, groundtruth_weights=None):
        """Assign classification and regression targets to each anchor.

        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth_boxes and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).

        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.

        Args:
            anchors: a BoxList representing N anchors

            groundtruth_boxes: a BoxList representing M groundtruth boxes

            groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]
                with labels for each of the ground_truth boxes. The subshape
                [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
                to None, groundtruth_labels assumes a binary problem where all
                ground_truth boxes get a positive label (of 1).

            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box. The weights
                must be in [0., 1.]. If None, all weights are set to 1.

            **params: Additional keyword arguments for specific implementations of the Matcher.

        Returns:
            cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
                where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
                which has shape [num_gt_boxes, d_1, d_2, ... d_k].

            cls_weights: a float32 tensor with shape [num_anchors]

            reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]

            reg_weights: a float32 tensor with shape [num_anchors]

            match: a matcher.Match object encoding the match between anchors and groundtruth boxes,
                with rows corresponding to groundtruth boxes and columns corresponding to anchors.

        Raises:
            ValueError: if anchors or groundtruth_boxes are not of type box_list.BoxList
        """
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')
        match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes, anchors)
        match = self._matcher.match(match_quality_matrix)
        reg_targets = self._create_regression_targets(anchors, groundtruth_boxes, match)
        cls_targets = self._create_classification_targets(groundtruth_labels, match)
        return cls_targets, reg_targets, match

    def _create_regression_targets(self, anchors: 'BoxList', groundtruth_boxes: 'BoxList', match: 'Match'):
        """Returns a regression target for each anchor.

        Args:
            anchors: a BoxList representing N anchors

            groundtruth_boxes: a BoxList representing M groundtruth_boxes

            match: a matcher.Match object

        Returns:
            reg_targets: a float32 tensor with shape [N, box_code_dimension]
        """
        device = anchors.device()
        zero_box = torch.zeros((1, 4), device=device)
        matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes.boxes(), unmatched_value=zero_box, ignored_value=zero_box)
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(self._keypoints_field_name):
            groundtruth_keypoints = groundtruth_boxes.get_field(self._keypoints_field_name)
            zero_kp = torch.zeros((1,) + groundtruth_keypoints.shape[1:], device=device)
            matched_keypoints = match.gather_based_on_match(groundtruth_keypoints, unmatched_value=zero_kp, ignored_value=zero_kp)
            matched_gt_boxlist.add_field(self._keypoints_field_name, matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        unmatched_ignored_reg_targets = self._default_regression_target(device).repeat(match.match_results.shape[0], 1)
        matched_anchors_mask = match.matched_column_indicator()
        reg_targets = torch.where(matched_anchors_mask.unsqueeze(1), matched_reg_targets, unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self, device: 'torch.device'):
        """Returns the default target for anchors to regress to.

        Default regression targets are set to zero (though in this implementation what
        these targets are set to should not matter as the regression weight of any box
        set to regress to the default target is zero).

        Returns:
            default_target: a float32 tensor with shape [1, box_code_dimension]
        """
        return torch.zeros(1, self._box_coder.code_size(), device=device)

    def _create_classification_targets(self, groundtruth_labels, match: 'Match'):
        """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Args:
            groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
                with labels for each of the ground_truth boxes. The subshape
                [d_1, ... d_k] can be empty (corresponding to scalar labels).
            match: a matcher.Match object that provides a matching between anchors
                and groundtruth boxes.

        Returns:
            a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
            subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
            shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
        return match.gather_based_on_match(groundtruth_labels, unmatched_value=self._unmatched_cls_target, ignored_value=self._unmatched_cls_target)

    def _create_regression_weights(self, match: 'Match', groundtruth_weights):
        """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Args:
            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.
            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box.

        Returns:
            a float32 tensor with shape [num_anchors] representing regression weights.
        """
        return match.gather_based_on_match(groundtruth_weights, ignored_value=0.0, unmatched_value=0.0)

    def _create_classification_weights(self, match: 'Match', groundtruth_weights):
        """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Args:
            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.
            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box.

        Returns:
            a float32 tensor with shape [num_anchors] representing classification weights.
        """
        return match.gather_based_on_match(groundtruth_weights, ignored_value=0.0, unmatched_value=self._negative_class_weight)

    def box_coder(self):
        """Get BoxCoder of this TargetAssigner.

        Returns:
            BoxCoder object.
        """
        return self._box_coder


class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, num_classes: 'int', match_threshold: 'float'=0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            num_classes: integer number representing number of classes in the dataset.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        similarity_calc = IouSimilarity()
        matcher = ArgMaxMatcher(match_threshold, unmatched_threshold=match_threshold, negatives_lower_than_unmatched=True, force_match_for_each_row=True)
        box_coder = FasterRcnnBoxCoder()
        self.target_assigner = TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes
        self.indices_cache = {}

    def label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_classes: A integer tensor with shape [N, 1] representing groundtruth classes.

            filter_valid: Filter out any boxes w/ gt class <= -1 before assigning

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        cls_targets_out = []
        box_targets_out = []
        if filter_valid:
            valid_idx = gt_classes > -1
            gt_boxes = gt_boxes[valid_idx]
            gt_classes = gt_classes[valid_idx]
        cls_targets, box_targets, matches = self.target_assigner.assign(BoxList(self.anchors.boxes), BoxList(gt_boxes), gt_classes)
        cls_targets = (cls_targets - 1).long()
        """Unpacks an array of cls/box into multiple scales."""
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.anchors.feat_sizes[level]
            steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
            cls_targets_out.append(cls_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
            box_targets_out.append(box_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
            count += steps
        num_positives = (matches.match_results > -1).float().sum()
        return cls_targets_out, box_targets_out, num_positives

    def batch_label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        batch_size = len(gt_boxes)
        assert batch_size == len(gt_classes)
        num_levels = self.anchors.max_level - self.anchors.min_level + 1
        cls_targets_out = [[] for _ in range(num_levels)]
        box_targets_out = [[] for _ in range(num_levels)]
        num_positives_out = []
        anchor_box_list = BoxList(self.anchors.boxes)
        for i in range(batch_size):
            last_sample = i == batch_size - 1
            if filter_valid:
                valid_idx = gt_classes[i] > -1
                gt_box_list = BoxList(gt_boxes[i][valid_idx])
                gt_class_i = gt_classes[i][valid_idx]
            else:
                gt_box_list = BoxList(gt_boxes[i])
                gt_class_i = gt_classes[i]
            cls_targets, box_targets, matches = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_class_i)
            cls_targets = (cls_targets - 1).long()
            """Unpacks an array of cls/box into multiple scales."""
            count = 0
            for level in range(self.anchors.min_level, self.anchors.max_level + 1):
                level_idx = level - self.anchors.min_level
                feat_size = self.anchors.feat_sizes[level]
                steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
                cls_targets_out[level_idx].append(cls_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
                box_targets_out[level_idx].append(box_targets[count:count + steps].reshape([feat_size[0], feat_size[1], -1]))
                count += steps
                if last_sample:
                    cls_targets_out[level_idx] = torch.stack(cls_targets_out[level_idx])
                    box_targets_out[level_idx] = torch.stack(box_targets_out[level_idx])
            num_positives_out.append((matches.match_results > -1).float().sum())
            if last_sample:
                num_positives_out = torch.stack(num_positives_out)
        return cls_targets_out, box_targets_out, num_positives_out


def huber_loss(input, target, delta: 'float'=1.0, weights: 'Optional[torch.Tensor]'=None, size_average: 'bool'=True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss = loss.mul(weights)
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def _box_loss(box_outputs, box_targets, num_positives, delta: 'float'=0.1):
    """Computes box regression loss."""
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_outputs, box_targets, weights=mask, delta=delta, size_average=False)
    return box_loss / normalizer


def focal_loss_legacy(logits, targets, alpha: 'float', gamma: 'float', normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    'Legacy focal loss matches the loss used in the official Tensorflow impl for initial
    model releases and some time after that. It eventually transitioned to the 'New' loss
    defined below.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    return weighted_loss / normalizer


def new_focal_loss(logits, targets, alpha: 'float', gamma: 'float', normalizer, label_smoothing: 'float'=0.01):
    """Compute the focal loss between `logits` and the golden `target` values.

    'New' is not the best descriptor, but this focal loss impl matches recent versions of
    the official Tensorflow impl of EfficientDet. It has support for label smoothing, however
    it is a bit slower, doesn't jit optimize well, and uses more memory.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: Divide loss by this value.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    pred_prob = logits.sigmoid()
    targets = targets
    onem_targets = 1.0 - targets
    p_t = targets * pred_prob + onem_targets * (1.0 - pred_prob)
    alpha_factor = targets * alpha + onem_targets * (1.0 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return 1 / normalizer * alpha_factor * modulating_factor * ce


def one_hot(x, num_classes: 'int'):
    x_non_neg = (x >= 0).unsqueeze(-1)
    onehot = torch.zeros(x.shape + (num_classes,), device=x.device, dtype=torch.float32)
    return onehot.scatter(-1, x.unsqueeze(-1) * x_non_neg, 1) * x_non_neg


def loss_fn(cls_outputs: 'List[torch.Tensor]', box_outputs: 'List[torch.Tensor]', cls_targets: 'List[torch.Tensor]', box_targets: 'List[torch.Tensor]', num_positives: 'torch.Tensor', num_classes: 'int', alpha: 'float', gamma: 'float', delta: 'float', box_loss_weight: 'float', label_smoothing: 'float'=0.0, legacy_focal: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
            at each feature level (index)

        box_outputs: a List with values representing box regression targets in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)

        cls_targets: groundtruth class targets.

        box_targets: groundtrusth box targets.

        num_positives: num positive grountruth anchors

    Returns:
        total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

        cls_loss: an integer tensor representing total class loss.

        box_loss: an integer tensor representing total box regression loss.
    """
    num_positives_sum = (num_positives.sum() + 1.0).float()
    levels = len(cls_outputs)
    cls_losses = []
    box_losses = []
    for l in range(levels):
        cls_targets_at_level = cls_targets[l]
        box_targets_at_level = box_targets[l]
        cls_targets_at_level_oh = one_hot(cls_targets_at_level, num_classes)
        bs, height, width, _, _ = cls_targets_at_level_oh.shape
        cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1).float()
        if legacy_focal:
            cls_loss = focal_loss_legacy(cls_outputs_at_level, cls_targets_at_level_oh, alpha=alpha, gamma=gamma, normalizer=num_positives_sum)
        else:
            cls_loss = new_focal_loss(cls_outputs_at_level, cls_targets_at_level_oh, alpha=alpha, gamma=gamma, normalizer=num_positives_sum, label_smoothing=label_smoothing)
        cls_loss = cls_loss.view(bs, height, width, -1, num_classes)
        cls_loss = cls_loss * (cls_targets_at_level != -2).unsqueeze(-1)
        cls_losses.append(cls_loss.sum())
        box_losses.append(_box_loss(box_outputs[l].permute(0, 2, 3, 1).float(), box_targets_at_level, num_positives_sum, delta=delta))
    cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
    box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
    total_loss = cls_loss + box_loss_weight * box_loss
    return total_loss, cls_loss, box_loss


loss_jit = torch.jit.script(loss_fn)


class DetectionLoss(nn.Module):
    __constants__ = ['num_classes']

    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight
        self.label_smoothing = config.label_smoothing
        self.legacy_focal = config.legacy_focal
        self.use_jit = config.jit_loss

    def forward(self, cls_outputs: 'List[torch.Tensor]', box_outputs: 'List[torch.Tensor]', cls_targets: 'List[torch.Tensor]', box_targets: 'List[torch.Tensor]', num_positives: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_fn = loss_fn
        if not torch.jit.is_scripting() and self.use_jit:
            l_fn = loss_jit
        return l_fn(cls_outputs, box_outputs, cls_targets, box_targets, num_positives, num_classes=self.num_classes, alpha=self.alpha, gamma=self.gamma, delta=self.delta, box_loss_weight=self.box_loss_weight, label_smoothing=self.label_smoothing, legacy_focal=self.legacy_focal)


class DetBenchTrain(nn.Module):

    def __init__(self, model, create_labeler=True):
        super(DetBenchTrain, self).__init__()
        self.model = model
        self.config = model.config
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(self.anchors, self.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(model.config)

    def forward(self, x, target: 'Dict[str, torch.Tensor]'):
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(target['bbox'], target['cls'])
        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
        if not self.training:
            class_out_pp, box_out_pp, indices, classes = _post_process(class_out, box_out, num_levels=self.num_levels, num_classes=self.num_classes, max_detection_points=self.max_detection_points)
            output['detections'] = _batch_detection(x.shape[0], class_out_pp, box_out_pp, self.anchors.boxes, indices, classes, target['img_scale'], target['img_size'], max_det_per_image=self.max_det_per_image, soft_nms=self.soft_nms)
        return output


class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: 'List[torch.Tensor]') ->List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class Interpolate2d(nn.Module):
    """Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: 'str'
    size: 'Optional[Union[int, Tuple[int, int]]]'
    scale_factor: 'Optional[Union[float, Tuple[float, float]]]'
    mode: 'str'
    align_corners: 'Optional[bool]'

    def __init__(self, size: 'Optional[Union[int, Tuple[int, int]]]'=None, scale_factor: 'Optional[Union[float, Tuple[float, float]]]'=None, mode: 'str'='nearest', align_corners: 'bool'=False) ->None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False)


_USE_SCALE = False


class ResampleFeatureMap(nn.Sequential):

    def __init__(self, in_channels, out_channels, input_size, output_size, pad_type='', downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_bn=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        if in_channels != out_channels:
            self.add_module('conv', ConvBnAct2d(in_channels, out_channels, kernel_size=1, padding=pad_type, norm_layer=norm_layer if apply_bn else None, bias=not apply_bn or redundant_bias, act_layer=None))
        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    kernel_size = stride_size_h + 1, stride_size_w + 1
                    stride = stride_size_h, stride_size_w
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type)
            elif _USE_SCALE:
                scale = output_size[0] / input_size[0], output_size[1] / input_size[1]
                down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
            else:
                down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module('downsample', down_inst)
        elif input_size[0] < output_size[0] or input_size[1] < output_size[1]:
            if _USE_SCALE:
                scale = output_size[0] / input_size[0], output_size[1] / input_size[1]
                self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
            else:
                self.add_module('upsample', Interpolate2d(size=output_size, mode=upsample))


class FpnCombine(nn.Module):

    def __init__(self, feature_info, fpn_channels, inputs_offsets, output_size, pad_type='', downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False, redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method
        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(feature_info[offset]['num_chs'], fpn_channels, input_size=feature_info[offset]['size'], output_size=output_size, pad_type=pad_type, downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn, redundant_bias=redundant_bias)
        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x: 'List[torch.Tensor]'):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)
        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights, dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights)
            weights_sum = torch.sum(edge_weights)
            out = torch.stack([(nodes[i] * edge_weights[i] / (weights_sum + 0.0001)) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: 'nn.Module', after_combine: 'nn.Module'):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: 'List[torch.Tensor]') ->torch.Tensor:
        return self.after_combine(self.combine(x))


def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {(min_level + i): [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)
    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_last_id(i), level_last_id(i + 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    for i in range(min_level + 1, max_level + 1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    return p


def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.

    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.

    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {(min_level + i): [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)
    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        offsets = [level_last_id(i), level_last_id(i + 1)] if i != max_level else [level_last_id(i)]
        p.nodes.append({'feat_level': i, 'inputs_offsets': offsets, 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    for i in range(min_level, max_level + 1):
        offsets = [level_last_id(i), level_last_id(i - 1)] if i != min_level else [level_last_id(i)]
        p.nodes.append({'feat_level': i, 'inputs_offsets': offsets, 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    return p


def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.

    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).

    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {(min_level + i): [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)
    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_last_id(i), level_last_id(i + 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])
    for i in range(min_level + 1, max_level):
        p.nodes.append({'feat_level': i, 'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    i = max_level
    p.nodes.append({'feat_level': i, 'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)], 'weight_method': weight_method})
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])
    for i in range(min_level + 1, max_level + 1, 1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])
    for i in range(max_level - 1, min_level, -1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)], 'weight_method': weight_method})
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)], 'weight_method': weight_method})
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])
    for i in range(min_level, max_level + 1):
        p.nodes.append({'feat_level': i, 'inputs_offsets': [node_ids[i][2], node_ids[i][4]], 'weight_method': quad_method})
        node_ids[i].append(next(id_cnt))
    return p


def get_fpn_config(fpn_name, min_level=3, max_level=7):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {'bifpn_sum': bifpn_config(min_level=min_level, max_level=max_level, weight_method='sum'), 'bifpn_attn': bifpn_config(min_level=min_level, max_level=max_level, weight_method='attn'), 'bifpn_fa': bifpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'), 'pan_sum': panfpn_config(min_level=min_level, max_level=max_level, weight_method='sum'), 'pan_fa': panfpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'), 'qufpn_sum': qufpn_config(min_level=min_level, max_level=max_level, weight_method='sum'), 'qufpn_fa': qufpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn')}
    return name_to_config[fpn_name]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(config.fpn_name, min_level=config.min_level, max_level=config.max_level)
        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                self.resample[str(level)] = ResampleFeatureMap(in_channels=in_chs, out_channels=config.fpn_channels, input_size=prev_feat_size, output_size=feat_size, pad_type=config.pad_type, downsample=config.downsample_type, upsample=config.upsample_type, norm_layer=norm_layer, apply_bn=config.apply_resample_bn, redundant_bias=config.redundant_bias)
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size
        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(feature_info=feature_info, feat_sizes=feat_sizes, fpn_config=fpn_config, fpn_channels=config.fpn_channels, num_levels=config.num_levels, pad_type=config.pad_type, downsample=config.downsample_type, upsample=config.upsample_type, norm_layer=norm_layer, act_layer=act_layer, separable_conv=config.separable_conv, apply_resample_bn=config.apply_resample_bn, pre_act=not config.conv_bn_relu_pattern, redundant_bias=config.redundant_bias)
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: 'List[torch.Tensor]'):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


class HeadNet(nn.Module):

    def __init__(self, config, num_outputs):
        super(HeadNet, self).__init__()
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, 'head_bn_level_first', False)
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_type = config.head_act_type if getattr(config, 'head_act_type', None) else config.act_type
        act_layer = get_act_layer(act_type) or _ACT_LAYER
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        conv_kwargs = dict(in_channels=config.fpn_channels, out_channels=config.fpn_channels, kernel_size=3, padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
        self.conv_rep = nn.ModuleList([conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats)])
        self.bn_rep = nn.ModuleList()
        if self.bn_level_first:
            for _ in range(self.num_levels):
                self.bn_rep.append(nn.ModuleList([norm_layer(config.fpn_channels) for _ in range(config.box_class_repeats)]))
        else:
            for _ in range(config.box_class_repeats):
                self.bn_rep.append(nn.ModuleList([nn.Sequential(OrderedDict([('bn', norm_layer(config.fpn_channels))])) for _ in range(self.num_levels)]))
        self.act = act_layer(inplace=True)
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(in_channels=config.fpn_channels, out_channels=num_outputs * num_anchors, kernel_size=3, padding=config.pad_type, bias=True, norm_layer=None, act_layer=None)
        self.predict = conv_fn(**predict_kwargs)

    @torch.jit.ignore()
    def toggle_bn_level_first(self):
        """ Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        with torch.no_grad():
            new_bn_rep = nn.ModuleList()
            for i in range(len(self.bn_rep[0])):
                bn_first = nn.ModuleList()
                for r in self.bn_rep.children():
                    m = r[i]
                    bn_first.append(m[0] if isinstance(m, nn.Sequential) else nn.Sequential(OrderedDict([('bn', m)])))
                new_bn_rep.append(bn_first)
            self.bn_level_first = not self.bn_level_first
            self.bn_rep = new_bn_rep

    @torch.jit.ignore()
    def _forward(self, x: 'List[torch.Tensor]') ->List[torch.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def _forward_level_first(self, x: 'List[torch.Tensor]') ->List[torch.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(self.bn_rep):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def forward(self, x: 'List[torch.Tensor]') ->List[torch.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x)


def _init_weight(m, n=''):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, (fan_in + fan_out) / 2.0)
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, fan_in)
        std = math.sqrt(gain)
        w.data.normal_(std=std)
    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=0.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(m, n=''):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

