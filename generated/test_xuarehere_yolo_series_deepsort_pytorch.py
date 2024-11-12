
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


import torchvision.transforms as transforms


import numpy as np


import logging


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import torchvision


import time


import matplotlib.pyplot as plt


import warnings


from collections import OrderedDict


import math


from typing import Dict


from typing import List


from typing import Tuple


import torch.distributed as dist


import random


import re


from torch.utils.cpp_extension import CppExtension


from torch import nn


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


import uuid


from torch.utils.data.dataloader import DataLoader as torchDataLoader


from torch.utils.data.dataloader import default_collate


from functools import wraps


from torch.utils.data.dataset import ConcatDataset as torchConcatDataset


from torch.utils.data.dataset import Dataset as torchDataset


import itertools


from typing import Optional


from torch.utils.data.sampler import BatchSampler as torchBatchSampler


from torch.utils.data.sampler import Sampler


from collections import ChainMap


from abc import ABCMeta


from abc import abstractmethod


from torch.nn import Module


import copy


from torch.hub import load_state_dict_from_url


from torch import distributed as dist


import functools


from copy import deepcopy


import inspect


from collections import defaultdict


from collections import deque


from typing import Sequence


from numpy import random


from warnings import warn


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data


from torch.cuda import amp


from scipy.cluster.vq import kmeans


from itertools import repeat


from torch.utils.data import Dataset


from torchvision.utils import save_image


import matplotlib


from copy import copy


from scipy.signal import butter


from scipy.signal import filtfilt


from torch.optim.optimizer import Optimizer


import collections


import torchvision.models as models


import torch.hub as hub


import pandas as pd


from torch.utils.mobile_optimizer import optimize_for_mobile


from collections import namedtuple


import tensorflow as tf


from tensorflow import keras


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


from torch.utils.data import dataloader


from torch.utils.data import distributed


from logging import Logger


from torchvision.ops import DeformConv2d


from torchvision.ops import roi_pool


from torchvision.ops import roi_align


from torchvision.ops import ps_roi_pool


from torchvision.ops import ps_roi_align


import torchvision.transforms as T


import torchvision.transforms.functional as TF


import logging.config


import typing


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CUDAExtension


from torch.backends import cudnn


from torch.utils.data import Sampler


from torch.nn.parallel import DistributedDataParallel


from collections import Counter


from sklearn import metrics


from torch.nn import Parameter


from torch.nn.modules.batchnorm import BatchNorm2d


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn.parameter import Parameter


from torch.nn import Conv2d


from torch.nn.modules.utils import _pair


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import required


from typing import Any


from typing import NamedTuple


from typing import Iterable


from torch.nn.parallel import DataParallel


from typing import IO


from typing import Callable


from typing import MutableMapping


from typing import Union


from scipy.stats import norm


from torch.onnx import OperatorExportTypes


from torch.nn.modules.utils import _list_with_default


from torch.nn.modules.batchnorm import _BatchNorm


from functools import partial


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


from torch.utils.data import DistributedSampler as _DistributedSampler


import torch.utils.checkpoint as cp


from torch.nn.modules.utils import _pair as to_2tuple


from logging import warning


from math import ceil


from math import log


from inspect import signature


from torch.nn import BatchNorm2d


from torch.utils.checkpoint import checkpoint


from math import sqrt


from torch.autograd import Function


from torch.nn import functional as F


from torch import nn as nn


from torch.nn.init import normal_


from torch.utils.cpp_extension import BuildExtension


from torch.nn.modules import GroupNorm


from torch.nn.modules import AvgPool2d


from torch.autograd import gradcheck


from torch.nn.init import constant_


from scipy.optimize import differential_evolution


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample)]
        else:
            blocks += [BasicBlock(c_out, c_out)]
    return nn.Sequential(*blocks)


class Net(nn.Module):

    def __init__(self, num_classes=625, reid=False):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ELU(inplace=True), nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ELU(inplace=True), nn.MaxPool2d(3, 2, padding=1))
        self.layer1 = make_layers(32, 32, 2, False)
        self.layer2 = make_layers(32, 64, 2, True)
        self.layer3 = make_layers(64, 128, 2, True)
        self.dense = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(128 * 16 * 8, 128), nn.BatchNorm1d(128), nn.ELU(inplace=True))
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        x = self.classifier(x)
        return x


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: 'str') ->None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: 'str' = name
        self._obj_map: 'Dict[str, object]' = {}

    def _do_register(self, name: 'str', obj: 'object') ->None:
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj: 'object'=None) ->Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:

            def deco(func_or_class: 'object') ->object:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: 'str') ->object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret


IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)


class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self, topk, iou_calculator=dict(type='BboxOverlaps2D'), ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self, bboxes, num_level_bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)
        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        if self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0:
            ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


INF = 100000000.0


class TaskAlignedAssigner(BaseAssigner):
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
    """

    def __init__(self, topk, iou_calculator=dict(type='BboxOverlaps2D')):
        assert topk >= 1
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, pred_scores, decode_bboxes, anchors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, alpha=1, beta=6):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_scores (Tensor): predicted class probability,
                shape(n, num_classes)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        assigned_gt_inds = anchors.new_full((num_bboxes,), 0, dtype=torch.long)
        assign_metrics = anchors.new_zeros((num_bboxes,))
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = anchors.new_zeros((num_bboxes,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = anchors.new_full((num_bboxes,), -1, dtype=torch.long)
            assign_result = AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result
        alignment_metrics = bbox_scores ** alpha * overlaps ** beta
        topk = min(self.topk, alignment_metrics.size(0))
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gt)]
        is_pos = candidate_metrics > 0
        anchors_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchors_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_anchors_cx = anchors_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_anchors_cy = anchors_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        l_ = ep_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        return assign_result


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class ConvBNLayer(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=filter_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CSPResStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, stride, act='relu', attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for i in range(n)])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class RepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        if self.deploy == False:
            self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        else:
            self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        if self.deploy:
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def switch_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__(self.conv1)
        self.__delattr__(self.conv2)
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPResNet(nn.Module):

    def __init__(self, layers=[3, 6, 6, 3], channels=[64, 128, 256, 512, 1024], act='swish', return_idx=[0, 1, 2, 3, 4], use_large_stem=False, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        if use_large_stem:
            self.stem = nn.Sequential(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        else:
            self.stem = nn.Sequential(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        n = len(channels) - 1
        self.stages = nn.Sequential(*[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act) for i in range(n)])
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = inputs
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


def drop_block_2d(x, drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: 'torch.Tensor', drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    if batchwise:
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0, with_noise=False, inplace=False, batchwise=False, fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


class GIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class VarifocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, reduction='mean', loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * varifocal_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, iou_weighted=self.iou_weighted, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def py_focal_loss_with_prob(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            elif torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss
            loss_cls = self.loss_weight * calculate_loss_func(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        x2_max = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        y1_min = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        y2_max = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0)
        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl + F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()
        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_module(str(i), BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):

    def __init__(self, in_channels=[256, 512, 1024], out_channels=[1024, 512, 256], norm_type='bn', act='leaky', stage_fn='CSPStage', block_fn='BasicBlock', stage_num=1, block_num=3, drop_block=False, block_size=3, keep_prob=0.9, spp=False, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        self.fpn_stages = nn.ModuleList()
        self.fpn_routes = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            stage = nn.Sequential()
            for j in range(stage_num):
                if stage_fn == 'CSPStage':
                    stage.add_module(str(j), CSPStage(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act, spp=spp and i == 0))
                else:
                    raise NotImplementedError
            if drop_block:
                stage.append(DropBlock2d(drop_prob=1 - keep_prob, block_size=block_size))
            self.fpn_stages.append(stage)
            if i < self.num_blocks - 1:
                self.fpn_routes.append(ConvBNLayer(ch_in=ch_out, ch_out=ch_out // 2, filter_size=1, stride=1, padding=0, act=act))
            ch_pre = ch_out
        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(ConvBNLayer(ch_in=out_channels[i + 1], ch_out=out_channels[i + 1], filter_size=3, stride=2, padding=1, act=act))
            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(str(j), eval(stage_fn)(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act, spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock2d(block_size, keep_prob))
            pan_stages.append(stage)
        self.pan_stages = nn.Sequential(*pan_stages[::-1])
        self.pan_routes = nn.Sequential(*pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], axis=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.0)
        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], axis=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)
        return pan_feats[::-1]


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PPYOLOE(nn.Module):

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, targets=None, extra_info=None):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)
        if self.training:
            assert targets is not None and extra_info is not None
            yolo_losses = self.head(neck_feats, targets, extra_info)
            return yolo_losses
        else:
            outputs = self.head(neck_feats)
            return outputs


class ESEAttn(nn.Module):

    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


def batch_distance2bbox(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def batch_distance2bbox_cxcywh(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    cxcy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    out_bbox = torch.cat([cxcy, wh], -1)
    return out_bbox


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class IOUloss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = torch.min(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'giou':
            c_tl = torch.min(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
            c_br = torch.max(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self, in_channels, mid_channels, dilation, norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule(in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(mid_channels, mid_channels, 3, padding=dilation, dilation=dilation, norm_cfg=norm_cfg)
        self.conv3 = ConvModule(mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=53, in_features=['dark3', 'dark4', 'dark5']):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act='lrelu')

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(*[self._make_cbl(in_filters, filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1)])
        return m

    def load_pretrained_model(self, filename='./weights/darknet53.mix.pth'):
        with open(filename, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        None
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)
        outputs = out_dark3, out_dark4, x0
        return outputs


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_mapping(bboxes, img_shape, scale_factor, flip, flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale."""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


logger = logging.getLogger(__name__)


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip, flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if not torch.onnx.is_in_onnx_export():
        valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    if not torch.onnx.is_in_onnx_export():
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return DummyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, outputs=1)


def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of             TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            if 0 < k < size:
                ret_k = k
        else:
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        pass
    return ret_k


def add_dummy_nms_for_onnx(boxes, scores, max_output_boxes_per_class=1000, iou_threshold=0.5, score_threshold=0.05, pre_top_k=-1, after_top_k=-1, labels=None):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (bool): Number of top K boxes to keep before nms.
            Defaults to -1.
        after_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        labels (Tensor, optional): It not None, explicit labels would be used.
            Otherwise, labels would be automatically generated using
            num_classed. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
            and class labels of shape [N, num_det].
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    num_class = scores.shape[2]
    nms_pre = torch.tensor(pre_top_k, device=scores.device, dtype=torch.long)
    nms_pre = get_k_for_topk(nms_pre, boxes.shape[1])
    if nms_pre > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(nms_pre)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds).long()
        transformed_inds = boxes.shape[1] * batch_inds + topk_inds
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
        scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(batch_size, -1, num_class)
        if labels is not None:
            labels = labels.reshape(-1, 1)[transformed_inds].reshape(batch_size, -1)
    scores = scores.permute(0, 2, 1)
    num_box = boxes.shape[1]
    state = torch._C._get_tracing_state()
    num_fake_det = 2
    batch_inds = torch.randint(batch_size, (num_fake_det, 1))
    cls_inds = torch.randint(num_class, (num_fake_det, 1))
    box_inds = torch.randint(num_box, (num_fake_det, 1))
    indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
    output = indices
    setattr(DummyONNXNMSop, 'output', output)
    torch._C._set_tracing_state(state)
    selected_indices = DummyONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
    box_inds = selected_indices[:, 2]
    if labels is None:
        labels = torch.arange(num_class, dtype=torch.long)
        labels = labels.view(1, num_class, 1).expand_as(scores)
    scores = scores.reshape(-1, 1)
    boxes = boxes.reshape(batch_size, -1).repeat(1, num_class).reshape(-1, 4)
    pos_inds = (num_class * batch_inds + cls_inds) * num_box + box_inds
    mask = scores.new_zeros(scores.shape)
    mask[pos_inds, :] += 1
    scores = scores * mask
    boxes = boxes * mask
    scores = scores.reshape(batch_size, -1)
    boxes = boxes.reshape(batch_size, -1, 4)
    labels = labels.reshape(batch_size, -1)
    nms_after = torch.tensor(after_top_k, device=scores.device, dtype=torch.long)
    nms_after = get_k_for_topk(nms_after, num_box * num_class)
    if nms_after > 0:
        _, topk_inds = scores.topk(nms_after)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
        transformed_inds = scores.shape[1] * batch_inds + topk_inds
        scores = scores.reshape(-1, 1)[transformed_inds, :].reshape(batch_size, -1)
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
        labels = labels.reshape(-1, 1)[transformed_inds, :].reshape(batch_size, -1)
    scores = scores.unsqueeze(2)
    dets = torch.cat([boxes, scores], dim=2)
    return dets, labels


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered,                 shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape                 (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape                 (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional):                 The filtered results. The shape of each item is                 (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)
    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)
    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [(1) for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device='cuda', with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device, with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda', with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_x = shift_x
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        shift_y = shift_y
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w)
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size, device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level                 feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self, prior_idxs, featmap_size, level_idx, dtype=torch.float32, device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = (prior_idxs // width % height + self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1)
        prioris = prioris
        return prioris


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return torch.cat(bbox_new, dim=-1)


BBOX_ASSIGNERS = Registry('bbox_assigner')


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


BBOX_SAMPLERS = Registry('bbox_sampler')


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains             a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=1.0, width=1.0, in_features=('dark3', 'dark4', 'dark5'), in_channels=[256, 512, 1024], depthwise=False, act='silu'):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)
        outputs = pan_out2, pan_out1, pan_out0
        return outputs


EPS = 1e-15


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]], [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=thickness, alpha=alpha)
    ax.add_collection(p)
    return ax


def draw_labels(ax, labels, positions, scores=None, class_names=None, color='w', font_size=8, scales=None, horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[label] if class_names is not None else f'class {label}'
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color
        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(pos[0], pos[1], f'{label_text}', bbox={'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}, color=text_color, fontsize=font_size_mask, verticalalignment='top', horizontalalignment=horizontal_alignment)
    return ax


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]
        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))
        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha
    p = PatchCollection(polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)
    return ax, img


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        self.transforms.append(transform)

    def tolist(self):
        return self.transforms

    def __repr__(self):
        format_string = f'{self.__class__.__name__}('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 0.001, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def get_cls_group_ofs(annotations, class_id):
    """Get `gt_group_of` of a certain class, which is used in Open Images.

    Args:
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        list[np.ndarray]: `gt_group_of` of a certain class.
    """
    gt_group_ofs = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        if ann.get('gt_is_group_ofs', None) is not None:
            gt_group_ofs.append(ann['gt_is_group_ofs'][gt_inds])
        else:
            gt_group_ofs.append(np.empty((0, 1), dtype=np.bool))
    return gt_group_ofs


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])
        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
    return cls_dets, cls_gts, cls_gts_ignore


dataset_aliases = {'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'], 'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'], 'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'], 'coco': ['coco', 'mscoco', 'ms_coco'], 'wider_face': ['WIDERFaceDataset', 'wider_face', 'WIDERFace'], 'cityscapes': ['cityscapes'], 'oid_challenge': ['oid_challenge', 'openimages_challenge'], 'oid_v6': ['oid_v6', 'openimages_v6']}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name
    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def print_map_summary(mean_ap, results, dataset=None, scale_ranges=None, logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """
    if logger == 'silent':
        return
    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1
    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales
    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']
    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset
    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [label_names[j], num_gts[i, j], results[j]['num_dets'], f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}']
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def tpfp_default(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_imagenet(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, default_iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        default_iou_thr (float): IoU threshold to be considered as matched for
            medium and large bboxes (small ones have special rules).
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes - 1, use_legacy_coordinate=use_legacy_coordinate)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length
    iou_thrs = np.minimum(gt_w * gt_h / ((gt_w + 10.0) * (gt_h + 10.0)), default_iou_thr)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            for j in range(num_gts):
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_openimages(det_bboxes, gt_bboxes, gt_bboxes_ignore=None, iou_thr=0.5, area_ranges=None, use_legacy_coordinate=False, gt_bboxes_group_of=None, use_group_of=True, ioa_thr=0.5):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        gt_bboxes_group_of (ndarray): GT group_of of this image, of shape
            (k, 1). Default: None
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: True.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: 0.5.

    Returns:
        tuple[np.ndarray]: Returns a tuple (tp, fp, det_bboxes), where
        (tp, fp) whose elements are 0 and 1. The shape of each array is
        (num_scales, m). (det_bboxes) whose will filter those are not
        matched by group of gts when processing Open Images evaluation.
        The shape is (num_scales, m).
    """
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool), np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp, det_bboxes
    if gt_bboxes_group_of is not None and use_group_of:
        assert gt_bboxes_group_of.shape[0] == gt_bboxes.shape[0]
        non_group_gt_bboxes = gt_bboxes[~gt_bboxes_group_of]
        group_gt_bboxes = gt_bboxes[gt_bboxes_group_of]
        num_gts_group = group_gt_bboxes.shape[0]
        ious = bbox_overlaps(det_bboxes, non_group_gt_bboxes)
        ioas = bbox_overlaps(det_bboxes, group_gt_bboxes, mode='iof')
    else:
        ious = bbox_overlaps(det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
        ioas = None
    if ious.shape[1] > 0:
        ious_max = ious.max(axis=1)
        ious_argmax = ious.argmax(axis=1)
        sort_inds = np.argsort(-det_bboxes[:, -1])
        for k, (min_area, max_area) in enumerate(area_ranges):
            gt_covered = np.zeros(num_gts, dtype=bool)
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            else:
                gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            for i in sort_inds:
                if ious_max[i] >= iou_thr:
                    matched_gt = ious_argmax[i]
                    if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            tp[k, i] = 1
                        else:
                            fp[k, i] = 1
                elif min_area is None:
                    fp[k, i] = 1
                else:
                    bbox = det_bboxes[i, :4]
                    area = (bbox[2] - bbox[0] + extra_length) * (bbox[3] - bbox[1] + extra_length)
                    if area >= min_area and area < max_area:
                        fp[k, i] = 1
    elif area_ranges == [(None, None)]:
        fp[...] = 1
    else:
        det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
        for i, (min_area, max_area) in enumerate(area_ranges):
            fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
    if ioas is None or ioas.shape[1] <= 0:
        return tp, fp, det_bboxes
    else:
        det_bboxes_group = np.zeros((num_scales, ioas.shape[1], det_bboxes.shape[1]), dtype=float)
        match_group_of = np.zeros((num_scales, num_dets), dtype=bool)
        tp_group = np.zeros((num_scales, num_gts_group), dtype=np.float32)
        ioas_max = ioas.max(axis=1)
        ioas_argmax = ioas.argmax(axis=1)
        sort_inds = np.argsort(-det_bboxes[:, -1])
        for k, (min_area, max_area) in enumerate(area_ranges):
            box_is_covered = tp[k]
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            else:
                gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
                gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            for i in sort_inds:
                matched_gt = ioas_argmax[i]
                if not box_is_covered[i]:
                    if ioas_max[i] >= ioa_thr:
                        if not (gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]):
                            if not tp_group[k, matched_gt]:
                                tp_group[k, matched_gt] = 1
                                match_group_of[k, i] = True
                            else:
                                match_group_of[k, i] = True
                            if det_bboxes_group[k, matched_gt, -1] < det_bboxes[i, -1]:
                                det_bboxes_group[k, matched_gt] = det_bboxes[i]
        fp_group = (tp_group <= 0).astype(float)
        tps = []
        fps = []
        for i in range(num_scales):
            tps.append(np.concatenate((tp[i][~match_group_of[i]], tp_group[i])))
            fps.append(np.concatenate((fp[i][~match_group_of[i]], fp_group[i])))
            det_bboxes = np.concatenate((det_bboxes[~match_group_of[i]], det_bboxes_group[i]))
        tp = np.vstack(tps)
        fp = np.vstack(fps)
        return tp, fp, det_bboxes


def eval_map(det_results, annotations, scale_ranges=None, iou_thr=0.5, ioa_thr=None, dataset=None, logger=None, tpfp_fn=None, nproc=4, use_legacy_coordinate=False, use_group_of=False):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: False.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = [(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges] if scale_ranges is not None else None
    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(det_results, annotations, i)
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            elif dataset in ['oid_challenge', 'oid_v6'] or use_group_of is True:
                tpfp_fn = tpfp_openimages
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
        args = []
        if use_group_of:
            gt_group_ofs = get_cls_group_ofs(annotations, i)
            args.append(gt_group_ofs)
            args.append([use_group_of for _ in range(num_imgs)])
        if ioa_thr is not None:
            args.append([ioa_thr for _ in range(num_imgs)])
        tpfp = pool.starmap(tpfp_fn, zip(cls_dets, cls_gts, cls_gts_ignore, [iou_thr for _ in range(num_imgs)], [area_ranges for _ in range(num_imgs)], [use_legacy_coordinate for _ in range(num_imgs)], *args))
        if use_group_of:
            tp, fp, cls_dets = tuple(zip(*tpfp))
        else:
            tp, fp = tuple(zip(*tpfp))
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum(tp + fp, eps)
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({'num_gts': num_gts, 'num_dets': num_dets, 'recall': recalls, 'precision': precisions, 'ap': ap})
    pool.close()
    if scale_ranges is not None:
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack([cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    print_map_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)
    return mean_ap, eval_results


def _recalls(all_ious, proposal_nums, thrs):
    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])
    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros(ious.shape[0])
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious
    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)
    return recalls


def print_recall_summary(recalls, proposal_nums, iou_thrs, row_idxs=None, col_idxs=None, logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        iou_thrs (ndarray or list): iou thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(iou thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format."""
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums
    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs
    return _proposal_nums, _iou_thrs


def eval_recalls(gts, proposals, proposal_nums=None, iou_thrs=0.5, logger=None, use_legacy_coordinate=False):
    """Calculate recalls.

    Args:
        gts (list[ndarray]): a list of arrays of shape (n, 4)
        proposals (list[ndarray]): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        use_legacy_coordinate (bool): Whether use coordinate system
            in mmdet v1.x. "1" was added to both height and width
            which means w, h should be
            computed as 'x2 - x1 + 1` and 'y2 - y1 + 1'. Default: False.


    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    img_num = len(gts)
    assert img_num == len(proposals)
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)
    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4], use_legacy_coordinate=use_legacy_coordinate)
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls


class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """
    CLASSES = None
    PALETTE = None

    def __init__(self, ann_file, pipeline, classes=None, data_root=None, img_prefix='', seg_prefix=None, proposal_file=None, test_mode=False, filter_empty_gt=True, file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(f'The used MMCV version does not have get_local_path. We treat the {self.ann_file} as local paths and it might cause errors if the path is not a local path. Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(f'The used MMCV version does not have get_local_path. We treat the {self.ann_file} as local paths and it might cause errors if the path is not a local path. Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn('CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set                 True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys                 introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by                 pipeline.
        """
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES
        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def get_cat2imgs(self):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self, results, metric='mAP', logger=None, proposal_nums=(100, 300, 1000), iou_thr=0.5, scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f"\n{'-' * 15}iou_thr: {iou_thr}{'-' * 15}")
                mean_ap, _ = eval_map(results, annotations, scale_ranges=scale_ranges, iou_thr=iou_thr, dataset=self.CLASSES, logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = f'\n{self.__class__.__name__} {dataset_type} dataset with number of images {len(self)}, and instance counts: \n'
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                instance_count[unique] += counts
            else:
                instance_count[-1] += 1
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)
        table = AsciiTable(table_data)
        result += table.table
        return result


class CocoDataset(CustomDataset):
    CLASSES = 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208)]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,                labels, masks, seg_map. "masks" are raw annotations and not                 decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        seg_map = img_info['filename'].replace('jpg', 'png')
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann, seg_map=seg_map)
        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """
        _bbox = bbox.tolist()
        return [_bbox[0], _bbox[1], _bbox[2] - _bbox[0], _bbox[3] - _bbox[1]]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and                 values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)
        recalls = eval_recalls(gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing                 the json filepaths, tmp_dir is the temporal directory created                 for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), 'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_det_segm(self, results, result_files, coco_gt, metrics, logger=None, classwise=False, proposal_nums=(100, 300, 1000), iou_thrs=None, metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for instance segmentation result.')
                ar = self.fast_eval_recall(results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn('The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.', UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log('The testing results of the whole dataset is empty.', logger=logger, level=logging.ERROR)
                break
            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            coco_metric_names = {'mAP': 0, 'mAP_50': 1, 'mAP_75': 2, 'mAP_s': 3, 'mAP_m': 4, 'mAP_l': 5, 'AR@100': 6, 'AR@300': 7, 'AR@1000': 8, 'AR_s@1000': 9, 'AR_m@1000': 10, 'AR_l@1000': 11}
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(f'metric item {metric_item} is not supported')
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)
                if metric_items is None:
                    metric_items = ['AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000']
                for item in metric_items:
                    val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)
                if classwise:
                    precisions = cocoEval.eval['precision']
                    assert len(self.cat_ids) == precisions.shape[2]
                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append((f"{nm['name']}", f'{float(ap):0.3f}'))
                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)
                if metric_items is None:
                    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}'
        return eval_results

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, classwise=False, proposal_nums=(100, 300, 1000), iou_thrs=None, metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt, metrics, logger, classwise, proposal_nums, iou_thrs, metric_items)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


INSTANCE_OFFSET = 1000


def parse_pq_results(pq_results):
    """Parse the Panoptic Quality results."""
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_th'] = 100 * pq_results['Things']['pq']
    result['SQ_th'] = 100 * pq_results['Things']['sq']
    result['RQ_th'] = 100 * pq_results['Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']
    return result


def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories, file_client=None):
    """The single core function to evaluate the metric of Panoptic
    Segmentation.

    Same as the function with the same name in `panopticapi`. Only the function
    to load the images is changed to use the file client.

    Args:
        proc_id (int): The id of the mini process.
        gt_folder (str): The path of the ground truth images.
        pred_folder (str): The path of the prediction images.
        categories (str): The categories of the dataset.
        file_client (object): The file client of the dataset. If None,
            the backend will be set to `disk`.
    """
    if PQStat is None:
        raise RuntimeError('panopticapi is not installed, please install it by: pip install git+https://github.com/cocodataset/panopticapi.git.')
    if file_client is None:
        file_client_args = dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
    pq_stat = PQStat()
    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            None
        idx += 1
        img_bytes = file_client.get(os.path.join(gt_folder, gt_ann['file_name']))
        pan_gt = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
        pan_gt = rgb2id(pan_gt)
        pan_pred = mmcv.imread(os.path.join(pred_folder, pred_ann['file_name']), flag='color', channel_order='rgb')
        pan_pred = rgb2id(pan_pred)
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[gt_id, pred_id] = intersection
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue
            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    None
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, file_client=None, nproc=32):
    """Evaluate the metrics of Panoptic Segmentation with multithreading.

    Same as the function with the same name in `panopticapi`.

    Args:
        matched_annotations_list (list): The matched annotation list. Each
            element is a tuple of annotations of the same image with the
            format (gt_anns, pred_anns).
        gt_folder (str): The path of the ground truth images.
        pred_folder (str): The path of the prediction images.
        categories (str): The categories of the dataset.
        file_client (object): The file client of the dataset. If None,
            the backend will be set to `disk`.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When `nproc` exceeds the number of cpu cores,
            the number of cpu cores is used.
    """
    if PQStat is None:
        raise RuntimeError('panopticapi is not installed, please install it by: pip install git+https://github.com/cocodataset/panopticapi.git.')
    if file_client is None:
        file_client_args = dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
    cpu_num = min(nproc, multiprocessing.cpu_count())
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    None
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core, (proc_id, annotation_set, gt_folder, pred_folder, categories, file_client))
        processes.append(p)
    workers.close()
    workers.join()
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def print_panoptic_table(pq_results, classwise_results=None, logger=None):
    """Print the panoptic evaluation results table.

    Args:
        pq_results(dict): The Panoptic Quality results.
        classwise_results(dict | None): The classwise Panoptic Quality results.
            The keys are class names and the values are metrics.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
    """
    headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
    data = [headers]
    for name in ['All', 'Things', 'Stuff']:
        numbers = [f'{pq_results[name][k] * 100:0.3f}' for k in ['pq', 'sq', 'rq']]
        row = [name] + numbers + [pq_results[name]['n']]
        data.append(row)
    table = AsciiTable(data)
    print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)
    if classwise_results is not None:
        class_metrics = [((name,) + tuple(f'{metrics[k] * 100:0.3f}' for k in ['pq', 'sq', 'rq'])) for name, metrics in classwise_results.items()]
        num_columns = min(8, len(class_metrics) * 4)
        results_flatten = list(itertools.chain(*class_metrics))
        headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
        results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
        data = [headers]
        data += [result for result in results_2d]
        table = AsciiTable(data)
        print_log('Classwise Panoptic Evaluation Results:\n' + table.table, logger=logger)


class CocoPanopticDataset(CocoDataset):
    """Coco dataset for Panoptic segmentation.

    The annotation format is shown as follows. The `ann` field is optional
    for testing.

    .. code-block:: none

        [
            {
                'filename': f'{image_id:012}.png',
                'image_id':9
                'segments_info': {
                    [
                        {
                            'id': 8345037, (segment_id in panoptic png,
                                            convert from rgb)
                            'category_id': 51,
                            'iscrowd': 0,
                            'bbox': (x1, y1, w, h),
                            'area': 24315,
                            'segmentation': list,(coded mask)
                        },
                        ...
                    }
                }
            },
            ...
        ]

    Args:
        ann_file (str): Panoptic segmentation annotation file path.
        pipeline (list[dict]): Processing pipeline.
        ins_ann_file (str): Instance segmentation annotation file path.
            Defaults to None.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Defaults to None.
        data_root (str, optional): Data root for ``ann_file``,
            ``ins_ann_file`` ``img_prefix``, ``seg_prefix``, ``proposal_file``
            if specified. Defaults to None.
        img_prefix (str, optional): Prefix of path to images. Defaults to ''.
        seg_prefix (str, optional): Prefix of path to segmentation files.
            Defaults to None.
        proposal_file (str, optional): Path to proposal file. Defaults to None.
        test_mode (bool, optional): If set True, annotation will not be loaded.
            Defaults to False.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Defaults to True.
        file_client_args (:obj:`mmcv.ConfigDict` | dict): file client args.
            Defaults to dict(backend='disk').
    """
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
    THING_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    STUFF_CLASSES = ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203), (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176), (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55), (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255), (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74), (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149), (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153), (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143), (102, 102, 156), (250, 141, 255)]

    def __init__(self, ann_file, pipeline, ins_ann_file=None, classes=None, data_root=None, img_prefix='', seg_prefix=None, proposal_file=None, test_mode=False, filter_empty_gt=True, file_client_args=dict(backend='disk')):
        super().__init__(ann_file, pipeline, classes=classes, data_root=data_root, img_prefix=img_prefix, seg_prefix=seg_prefix, proposal_file=proposal_file, test_mode=test_mode, filter_empty_gt=filter_empty_gt, file_client_args=file_client_args)
        self.ins_ann_file = ins_ann_file

    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCOPanoptic(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        ann_info = [i for i in ann_info if i['image_id'] == img_id]
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]
            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False
            mask_info = {'id': ann['id'], 'category': contiguous_cat_id, 'is_thing': is_thing}
            gt_mask_infos.append(mask_info)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, masks=gt_mask_infos, seg_map=img_info['segm_file'])
        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = []
        for lists in self.coco.anns.values():
            for item in lists:
                category_id = item['category_id']
                is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
                if not is_thing:
                    continue
                ids_with_ann.append(item['image_id'])
        ids_with_ann = set(ids_with_ann)
        valid_inds = []
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _pan2json(self, results, outfile_prefix):
        """Convert panoptic results to COCO panoptic json style."""
        label2cat = dict((v, k) for k, v in self.cat2label.items())
        pred_annotations = []
        outdir = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            segm_file = self.data_infos[idx]['segm_file']
            pan = results[idx]
            pan_labels = np.unique(pan)
            segm_info = []
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                if sem_label == len(self.CLASSES):
                    continue
                cat_id = label2cat[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                mask = pan == pan_label
                area = mask.sum()
                segm_info.append({'id': int(pan_label), 'category_id': cat_id, 'isthing': is_thing, 'area': int(area)})
            pan[pan % INSTANCE_OFFSET == len(self.CLASSES)] = VOID
            pan = id2rgb(pan).astype(np.uint8)
            mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
            record = {'image_id': img_id, 'segments_info': segm_info, 'file_name': segm_file}
            pred_annotations.append(record)
        pan_json_results = dict(annotations=pred_annotations)
        return pan_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the results to a COCO style json file.

        There are 4 types of results: proposals, bbox predictions, mask
        predictions, panoptic segmentation predictions, and they have
        different data types. This method will automatically recognize
        the type, and dump them to json files.

        .. code-block:: none

            [
                {
                    'pan_results': np.array, # shape (h, w)
                    # ins_results which includes bboxes and RLE encoded masks
                    # is optional.
                    'ins_results': (list[np.array], list[list[str]])
                },
                ...
            ]

        Args:
            results (list[dict]): Testing results of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.panoptic.json", "somepath/xxx.bbox.json",
                "somepath/xxx.segm.json"

        Returns:
            dict[str: str]: Possible keys are "panoptic", "bbox", "segm",                 "proposal", and values are corresponding filenames.
        """
        result_files = dict()
        if 'pan_results' in results[0]:
            pan_results = [result['pan_results'] for result in results]
            pan_json_results = self._pan2json(pan_results, outfile_prefix)
            result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
            mmcv.dump(pan_json_results, result_files['panoptic'])
        if 'ins_results' in results[0]:
            ins_results = [result['ins_results'] for result in results]
            bbox_json_results, segm_json_results = self._segm2json(ins_results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(bbox_json_results, result_files['bbox'])
            mmcv.dump(segm_json_results, result_files['segm'])
        return result_files

    def evaluate_pan_json(self, result_files, outfile_prefix, logger=None, classwise=False, nproc=32):
        """Evaluate PQ according to the panoptic results json file."""
        imgs = self.coco.imgs
        gt_json = self.coco.img_ann_map
        gt_json = [{'image_id': k, 'segments_info': v, 'file_name': imgs[k]['segm_file']} for k, v in gt_json.items()]
        pred_json = mmcv.load(result_files['panoptic'])
        pred_json = dict((el['image_id'], el) for el in pred_json['annotations'])
        matched_annotations_list = []
        for gt_ann in gt_json:
            img_id = gt_ann['image_id']
            if img_id not in pred_json.keys():
                raise Exception('no prediction for the image with id: {}'.format(img_id))
            matched_annotations_list.append((gt_ann, pred_json[img_id]))
        gt_folder = self.seg_prefix
        pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')
        pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, self.categories, self.file_client, nproc=nproc)
        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}
        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results
        classwise_results = None
        if classwise:
            classwise_results = {k: v for k, v in zip(self.CLASSES, pq_results['classwise'].values())}
        print_panoptic_table(pq_results, classwise_results, logger=logger)
        results = parse_pq_results(pq_results)
        results['PQ_copypaste'] = f"{results['PQ']:.3f} {results['SQ']:.3f} {results['RQ']:.3f} {results['PQ_th']:.3f} {results['SQ_th']:.3f} {results['RQ_th']:.3f} {results['PQ_st']:.3f} {results['SQ_st']:.3f} {results['RQ_st']:.3f}"
        return results

    def evaluate(self, results, metric='PQ', logger=None, jsonfile_prefix=None, classwise=False, nproc=32, **kwargs):
        """Evaluation in COCO Panoptic protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'PQ', 'bbox',
                'segm', 'proposal' are supported. 'pq' will be regarded as 'PQ.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to print classwise evaluation results.
                Default: False.
            nproc (int): Number of processes for panoptic quality computing.
                Defaults to 32. When `nproc` exceeds the number of cpu cores,
                the number of cpu cores is used.

        Returns:
            dict[str, float]: COCO Panoptic style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        metrics = [('PQ' if metric == 'pq' else metric) for metric in metrics]
        allowed_metrics = ['PQ', 'bbox', 'segm', 'proposal']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}
        outfile_prefix = os.path.join(tmp_dir.name, 'results') if tmp_dir is not None else jsonfile_prefix
        if 'PQ' in metrics:
            eval_pan_results = self.evaluate_pan_json(result_files, outfile_prefix, logger, classwise, nproc=nproc)
            eval_results.update(eval_pan_results)
            metrics.remove('PQ')
        if 'bbox' in metrics or 'segm' in metrics or 'proposal' in metrics:
            assert 'ins_results' in results[0], 'instance segmentationresults are absent from results'
            assert self.ins_ann_file is not None, 'Annotation file for instance segmentation or object detection shuold not be None'
            coco_gt = COCO(self.ins_ann_file)
            panoptic_cat_ids = self.cat_ids
            self.cat_ids = coco_gt.get_cat_ids(cat_names=self.THING_CLASSES)
            eval_ins_results = self.evaluate_det_segm(results, result_files, coco_gt, metrics, logger, classwise, **kwargs)
            self.cat_ids = panoptic_cat_ids
            eval_results.update(eval_ins_results)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


class XMLDataset(CustomDataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
    """

    def __init__(self, min_size=None, img_subdir='JPEGImages', ann_subdir='Annotations', **kwargs):
        assert self.CLASSES or kwargs.get('classes', None), 'CLASSES in `XMLDataset` can not be None.'
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super(XMLDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            bbox = [int(float(bnd_box.find('xmin').text)), int(float(bnd_box.find('ymin').text)), int(float(bnd_box.find('xmax').text)), int(float(bnd_box.find('ymax').text))]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(bboxes=bboxes.astype(np.float32), labels=labels.astype(np.int64), bboxes_ignore=bboxes_ignore.astype(np.float32), labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)
        return cat_ids


class VOCDataset(XMLDataset):
    CLASSES = 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self, results, metric='mAP', logger=None, proposal_nums=(100, 300, 1000), iou_thr=0.5, scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f"\n{'-' * 15}iou_thr: {iou_thr}{'-' * 15}")
                mean_ap, _ = eval_map(results, annotations, scale_ranges=None, iou_thr=iou_thr, dataset=ds_name, logger=logger, use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(gt_bboxes, results, proposal_nums, iou_thrs, logger=logger, use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


def get_palette(palette, num_classes):
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple | :obj:`Color`): palette inputs.
        num_classes (int): the number of classes.

    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)
    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        dataset_palette = CocoDataset.PALETTE
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.PALETTE
    elif palette == 'citys':
        dataset_palette = CityscapesDataset.PALETTE
    elif palette == 'voc':
        dataset_palette = VOCDataset.PALETTE
    elif mmcv.is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')
    assert len(dataset_palette) >= num_classes, 'The length of palette should not be less than `num_classes`.'
    return dataset_palette


def palette_val(palette):
    """Convert palette to matplotlib palette.

    Args:
        palette List[tuple]: A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [(c / 255) for c in color]
        new_palette.append(tuple(color))
    return new_palette


def imshow_det_bboxes(img, bboxes=None, labels=None, segms=None, class_names=None, score_thr=0, bbox_color='green', text_color='green', mask_color=None, thickness=2, font_size=8, win_name='', show=True, wait_time=0, out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], 'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], 'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, 'segms and bboxes should not be None at the same time.'
    img = mmcv.imread(img).astype(np.uint8)
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]
    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)
        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(ax, labels[:num_bboxes], positions, scores=scores, class_names=class_names, color=text_colors, font_size=font_size, scales=scales, horizontal_alignment=horizontal_alignment)
    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)
        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(ax, labels[num_bboxes:], positions, class_names=class_names, color=text_colors, font_size=font_size, scales=scales, horizontal_alignment=horizontal_alignment)
    plt.imshow(img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    plt.close()
    return img


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


BACKBONE_REGISTRY = Registry('BACKBONE')


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back
    return frame.f_globals['__name__']


def get_logger(name='root'):
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)
    return logger


def log_img_scale(img_scale, shape_order='hw', skip_square=False):
    """Log image size.

    Args:
        img_scale (tuple): Image size to be logged.
        shape_order (str, optional): The order of image shape.
            'hw' for (height, width) and 'wh' for (width, height).
            Defaults to 'hw'.
        skip_square (bool, optional): Whether to skip logging for square
            img_scale. Defaults to False.

    Returns:
        bool: Whether to have done logging.
    """
    if shape_order == 'hw':
        height, width = img_scale
    elif shape_order == 'wh':
        width, height = img_scale
    else:
        raise ValueError(f'Invalid shape_order {shape_order}.')
    if skip_square and height == width:
        return False
    logger = get_root_logger()
    caller = get_caller_name()
    logger.info(f'image shape: height={height}, width={width} in {caller}')
    return True


ONNX_EXPORT = False


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec
            self.anchor_wh = self.anchor_wh

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            w = torch.sigmoid(p[:, -n:]) * (2 / n)
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh
        else:
            io = p.sigmoid()
            io[..., :2] = io[..., :2] * 2.0 - 0.5 + self.grid
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            return io.view(bs, -1, self.no), p


class JDELayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(JDELayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec
            self.anchor_wh = self.anchor_wh

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            w = torch.sigmoid(p[:, -n:]) * (2 / n)
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) * 2.0 - 0.5 + self.grid
            io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            io[..., 4:] = F.softmax(io[..., 4:])
            return io.view(bs, -1, self.no), p


class Hardswish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MemoryEfficientMish(nn.Module):


    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(nn.Module):

    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * (W // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class FeatureConcat2(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat2, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach()], 1)


class FeatureConcat3(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat3, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach(), outputs[self.layers[2]].detach()], 1)


class FeatureConcat_l(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat_l, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i][:, :outputs[i].shape[1] // 2, :, :] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]][:, :outputs[self.layers[0]].shape[1] // 2, :, :]


class WeightedFeatureFusion(nn.Module):

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]
            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
        return x


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class HardSwish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


class DeformConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.avg_pool(x)


class Silence(nn.Module):

    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        return x


class ScaleChannel(nn.Module):

    def __init__(self, layers):
        super(ScaleChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a


class ShiftChannel(nn.Module):

    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x


class ShiftChannel2D(nn.Module):

    def __init__(self, layers):
        super(ShiftChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.expand_as(x) + x


class ControlChannel(nn.Module):

    def __init__(self, layers):
        super(ControlChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) * x


class ControlChannel2D(nn.Module):

    def __init__(self, layers):
        super(ControlChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.expand_as(x) * x


class AlternateChannel(nn.Module):

    def __init__(self, layers):
        super(AlternateChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return torch.cat([a.expand_as(x), x], dim=1)


class AlternateChannel2D(nn.Module):

    def __init__(self, layers):
        super(AlternateChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return torch.cat([a.expand_as(x), x], dim=1)


class SelectChannel(nn.Module):

    def __init__(self, layers):
        super(SelectChannel, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.sigmoid().expand_as(x) * x


class SelectChannel2D(nn.Module):

    def __init__(self, layers):
        super(SelectChannel2D, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1, -1, 1, 1)
        return a.sigmoid().expand_as(x) * x


class ScaleSpatial(nn.Module):

    def __init__(self, layers):
        super(ScaleSpatial, self).__init__()
        self.layers = layers

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x * a


class ImplicitA(nn.Module):

    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitC(nn.Module):

    def __init__(self, channel):
        super(ImplicitC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):

    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class Implicit2DA(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class Implicit2DC(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, atom, channel, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self):
        return self.implicit


class Implicit2DM(nn.Module):

    def __init__(self, atom, channel):
        super(Implicit2DM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, atom, channel, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self):
        return self.implicit


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class BottleneckCSP2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(VoVCSP, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1 // 2, c_ // 2, 3, 1)
        self.cv2 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(torch.cat((x1, x2), dim=1))


class SPPCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class MP(nn.Module):

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class FrozenBatchNorm(BatchNorm):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """
    _version = 3

    def __init__(self, num_features, eps=1e-05, **kwargs):
        super().__init__(num_features, weight_freeze=True, bias_freeze=True, **kwargs)
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self.running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.running_var)
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info('FrozenBatchNorm {} is upgraded to version 3.'.format(prefix.rstrip('.')))
            state_dict[prefix + 'running_var'] -= self.eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = bn_module.BatchNorm2d, bn_module.SyncBatchNorm
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class GhostBatchNorm(BatchNorm):

    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var, self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits), True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)


class SyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that thakes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm, 'GhostBN': GhostBatchNorm, 'FrozenBN': FrozenBatchNorm, 'GN': lambda channels, **args: nn.GroupNorm(32, channels), 'syncBN': SyncBatchNorm}[norm]
    return norm(out_channels, **kwargs)


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, bn_norm, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = in_channels
        kernel = 3
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class CombConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(Conv(inch, outch, k=3))
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if i == 0 and self.keepBase or i == t - 1 or i % 2 == 1:
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class BRLayer(nn.Sequential):

    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock2(nn.Module):

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)
        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum(self.out_partition[i])
            real_out_ch = self.out_partition[i][0]
            conv_layers_.append(nn.Conv2d(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias=True))
            bnrelu_layers_.append(BRLayer(real_out_ch))
            cur_ch = real_out_ch
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += real_out_ch
        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [(blk.layers[k - 1][0].weight.shape[0] if k > 0 else blk.layers[0][0].weight.shape[1]) for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias
            self.conv_layers[i].weight[0:part[0], :, :, :] = w_src[:, 0:in_ch, :, :]
            self.layer_bias.append(b_src)
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None
            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link)):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chos = sum(self.out_partition[ly][0:part_id])
                    choe = chos + part[0]
                    chis = sum(link_ch[0:j])
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :, :, :] = w_src[:, chis:chie, :, :]
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2d):
                self.bnrelu_layers[i] = nn.Sequential(blk.layers[i][1], blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]
            xout = self.conv_layers[i](xin)
            layers_.append(xout)
            xin = xout[:, 0:part[0], :, :] if len(part) > 1 else xout
            if len(link) > 1:
                for j in range(len(link) - 1):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chs = sum(self.out_partition[ly][0:part_id])
                    che = chs + part[0]
                    xin += layers_[ly][:, chs:che, :, :]
            xin = self.bnrelu_layers[i](xin)
            if i % 2 == 0 or i == len(self.conv_layers) - 1:
                outs_.append(xin)
        out = torch.cat(outs_, 1)
        return out


class ConvSig(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvSqu(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvSqu, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CrossConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class DFL(nn.Module):

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


LOGGING_NAME = 'yolov5'


LOGGER = logging.getLogger(LOGGING_NAME)


def emojis(string=''):
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(s)
    if verbose and not result:
        LOGGER.warning(s)
    return result


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class IAuxDetect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(IAuxDetect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = x[i + self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class SigmoidBin(nn.Module):
    stride = None
    export = False

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0
        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight
        start = min + self.scale / 2.0 / self.bin_count
        end = max - self.scale / 2.0 / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)
        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps
        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:1 + self.bin_count]
        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]
        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)
        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device
        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:1 + self.bin_count]
        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias
        target_bins = torch.full_like(pred_bin, self.cn, device=device)
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp
        loss_bin = self.BCEbins(pred_bin, target_bins)
        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin
        out_result = result.clamp(min=self.min, max=self.max)
        return loss, out_result


class IBin(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):
        super(IBin, self).__init__()
        self.nc = nc
        self.bin_count = bin_count
        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.no = nc + 3 + self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]
                y[..., 2] = pw
                y[..., 3] = ph
                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)
                z.append(y.view(bs, -1, y.shape[-1]))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IDetect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(IDetect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def box_iou(box1, box2, eps=1e-07):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nm=0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi]
        if time.time() - t > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break
    return output


class NMS(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class RepConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        padding_11 = autopad(k, p) - k // 2
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False), nn.BatchNorm2d(num_features=c2))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False), nn.BatchNorm2d(num_features=c2))

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t
        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True, padding_mode=conv.padding_mode)
        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        None
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        if isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm):
            identity_conv_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))
        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
        self.rbr_reparam = self.rbr_dense
        self.deploy = True
        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None
        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None
        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * (gamma / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * gamma / std


class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


class OREPA_3x3_RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepConv, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.branch_counter = 0
        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1
        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
        else:
            raise NotImplementedError
        self.branch_counter += 1
        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)
        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1
        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels * expand_ratio, 1, 1))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1
        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1
        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.fre_init()
        nn.init.constant_(self.vector[0, :], 0.25)
        nn.init.constant_(self.vector[1, :], 0.25)
        nn.init.constant_(self.vector[2, :], 0.0)
        nn.init.constant_(self.vector[3, :], 0.5)
        nn.init.constant_(self.vector[4, :], 0.5)

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):
        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])
        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])
        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2
        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)
        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])
        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])
        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv
        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))


class RepConv_OREPA(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, nonlinear=nn.SiLU()):
        super(RepConv_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = c1
        self.out_channels = c2
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert k == 3
        assert padding == 1
        padding_11 = padding - k // 2
        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear
        if use_se:
            self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
            self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=s, padding=padding_11, groups=groups, dilation=1)
            None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3
        return self.nonlinearity(self.se(out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepConv):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        None
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels, kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride, padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


def is_writeable(dir, test=False):
    if not test:
        return os.access(dir, os.W_OK)
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):
            pass
        file.unlink()
        return True
    except OSError:
        return False


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}
        path = Path.home() / cfg.get(platform.system(), '')
        path = (path if is_writeable(path) else Path('/tmp')) / dir
    path.mkdir(exist_ok=True)
    return path


FONT = 'Arial.ttf'


def check_font(font=FONT, progress=False):
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_python(minimum='3.7.0'):
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m', 'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m', 'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m', 'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m', 'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m', 'underline': '\x1b[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def check_pil_font(font=FONT, size=10):
    font = Path(font)
    font = font if font.exists() else CONFIG_DIR / font.name
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')
        except URLError:
            return ImageFont.load_default()


def is_ascii(s=''):
    s = str(s)
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


class Annotator:

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)
        self.pil = pil or non_ascii
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle((box[0], box[1] - h if outside else box[1], box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1), fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]
        masks = masks.unsqueeze(3)
        masks_color = masks * (colors * alpha)
        inv_alph_masks = (1 - masks * alpha).cumprod(0)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2
        im_gpu = im_gpu.flip(dims=[0])
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        self.im[:] = im_mask if retina_masks else scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        if anchor == 'bottom':
            w, h = self.font.getsize(text)
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        return np.asarray(self.im)


class Colors:

    def __init__(self):
        hexs = 'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = new_shape, new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def clip_coords(boxes, shape):
    """yolov5

    Args:
        boxes ([type]): [description]
        shape ([type]): [description]
    """
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """yolov5

    Args:
        img1_shape ([type]): [description]
        coords ([type]): [description]
        img0_shape ([type]): [description]
        ratio_pad ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class autoShape(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        None
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_synchronized()]
        p = next(self.model.parameters())
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.type_as(p), augment, profile)
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, str):
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = size / max(s)
            shape1.append([(y * g) for y in s])
            imgs[i] = im
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).type_as(p) / 255.0
        t.append(time_synchronized())
        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)[0]
            t.append(time_synchronized())
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da and da.sign() != ds.sign():
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def fuse_conv_and_bn(conv: 'nn.Conv2d', bn: 'nn.BatchNorm2d') ->nn.Conv2d:
    """
    Fuse convolution and batchnorm layers.
    check more info on https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (nn.Conv2d): convolution to fuse.
        bn (nn.BatchNorm2d): batchnorm to fuse.

    Returns:
        nn.Conv2d: fused convolution behaves the same as the input conv and bn.
    """
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def model_info(model, verbose=False, imgsz=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        None
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            None
    try:
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1000000000.0 * 2
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'
    except Exception:
        fs = ''
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


class C1(nn.Module):

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3Ghost(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3x(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class ConvTranspose(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))


class DWConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Proto(nn.Module):

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


def parse_model(d, ch, verbose=True):
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in {Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(args[2] * gw, 8)
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = int(h * ratio), int(w * ratio)
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class Model(nn.Module):

    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        if isinstance(m, IDetect):
            s = 256
            m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        if isinstance(m, IAuxDetect):
            s = 256
            m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, s, s))[:4]])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_aux_biases()
        if isinstance(m, IBin):
            s = 256
            m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases_bin()
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if not hasattr(self, 'traced'):
                self.traced = False
            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                    break
            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                None
            x = m(x)
            y.append(x if m.i in self.save else None)
        if profile:
            None
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):
        m = self.model[-1]
        for mi, mi2, s in zip(m.m, m.m2, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):
        m = self.model[-1]
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            old = b[:, (0, 1, 2, bc + 3)].data
            obj_idx = 2 * bc + 4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)
            b[:, obj_idx + 1:].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            b[:, (0, 1, 2, bc + 3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            None

    def fuse(self):
        None
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):
        present = type(self.model[-1]) is NMS
        if mode and not present:
            None
            m = NMS()
            m.f = -1
            m.i = self.model[-1].i + 1
            self.model.add_module(name='%s' % m.i, module=m)
            self.eval()
        elif not mode and present:
            None
            self.model = self.model[:-1]
        return self

    def autoshape(self):
        None
        m = autoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


class C3SPP(C3):

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


def attempt_download(weights):
    weights = weights.strip().replace("'", '')
    file = Path(weights).name
    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
    if file in models and not os.path.isfile(weights):
        try:
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            None
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1000000.0
        except Exception as e:
            None
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            None
            r = os.system('curl -L %s -o %s' % (url, weights))
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1000000.0):
                os.remove(weights) if os.path.exists(weights) else None
                None
            None
            return


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    if len(model) == 1:
        return model[-1]
    else:
        None
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in (file if isinstance(file, (list, tuple)) else [file]):
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


def export_formats():
    x = [['PyTorch', '-', '.pt', True, True], ['TorchScript', 'torchscript', '.torchscript', True, True], ['ONNX', 'onnx', '.onnx', True, True], ['OpenVINO', 'openvino', '_openvino_model', True, False], ['TensorRT', 'engine', '.engine', False, True], ['CoreML', 'coreml', '.mlmodel', True, False], ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True], ['TensorFlow GraphDef', 'pb', '.pb', True, True], ['TensorFlow Lite', 'tflite', '.tflite', True, False], ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False], ['TensorFlow.js', 'tfjs', '_web_model', False, False], ['PaddlePaddle', 'paddle', '_paddle_model', True, True]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def yaml_load(file='data.yaml'):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    model = Ensemble()
    for w in (weights if isinstance(weights, list) else [weights]):
        ckpt = torch.load(attempt_download(w), map_location='cpu')
        args = {**DEFAULT_CONFIG_DICT, **ckpt['train_args']}
        ckpt = (ckpt.get('ema') or ckpt['model']).float()
        ckpt.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}
        ckpt.pt_path = weights
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.0])
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    if len(model) == 1:
        return model[-1]
    None
    for k in ('names', 'nc', 'yaml'):
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


def is_url(url, check=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        return urllib.request.urlopen(url).getcode() == 200 if check else True
    except (AssertionError, urllib.request.HTTPError):
        return False


class LetterBox:

    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im):
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = round(imh * r), round(imw * r)
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.size(1) == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * 0.5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * 0.5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * 0.5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_scaled = torch.zeros_like(bboxes)
    boxes_scaled[:, 0] = x_c - w_half
    boxes_scaled[:, 2] = x_c + w_half
    boxes_scaled[:, 1] = y_c - h_half
    boxes_scaled[:, 3] = y_c + h_half
    return boxes_scaled


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, profile=False, visualize=False):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.tensor): The input image tensor
            profile (bool): Whether to profile the model, defaults to False
            visualize (bool): Whether to return the intermediate feature maps, defaults to False

        Returns:
            (torch.tensor): The output of the network.
        """
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False

        Returns:
            (torch.tensor): The last output of the model.
        """
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                LOGGER.info('visualize feature not yet supported')
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1]
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self

    def info(self, verbose=False, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        model_info(self, verbose, imgsz)

    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights):
        """
        This function loads the weights of the model from a file

        Args:
            weights (str): The weights to load into the model.
        """
        raise NotImplementedError('This function needs to be implemented by derived classes!')


def check_file(file, suffix=''):
    check_suffix(file, suffix)
    file = str(file)
    if os.path.isfile(file) or not file:
        return file
    elif file.startswith(('http:/', 'https:/')):
        url = file
        file = Path(urllib.parse.unquote(file).split('?')[0]).name
        if os.path.isfile(file):
            LOGGER.info(f'Found {url} locally at {file}')
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'
        return file
    elif file.startswith('clearml://'):
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:
        files = []
        for d in ('data', 'models', 'utils'):
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))
        assert len(files), f'File not found: {file}'
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]


def check_yaml(file, suffix=('.yaml', '.yml')):
    return check_file(file, suffix)


def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


class DetectionModel(BaseModel):

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(check_yaml(cfg), append_filename=True)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([(s / x.shape[-2]) for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = y[0].shape[-1] // g * sum(4 ** x for x in range(e))
        y[0] = y[0][..., :-i]
        i = y[-1].shape[-1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def load(self, weights, verbose=True):
        csd = weights.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')


class ClassificationModel(BaseModel):

    def __init__(self, cfg=None, model=None, ch=3, nc=1000, cutoff=10, verbose=True):
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, verbose)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        if isinstance(model, AutoBackend):
            model = model.model
        model.model = model.model[:cutoff]
        m = model.model[-1]
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels
        c = Classify(ch, nc)
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'
        model.model[-1] = c
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, verbose):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(check_yaml(cfg), append_filename=True)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.info()

    def load(self, weights):
        model = weights['model'] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)

    @staticmethod
    def reshape_outputs(model, nc):
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]
        if isinstance(m, Classify):
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


class AconC(nn.Module):
    """ ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    """ ACON activation (activate or not)
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        beta = torch.sigmoid(self.fc2(self.fc1(y)))
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SimConv(nn.Module):
    """Normal Conv with ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Transpose:
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to                 ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys}, order={self.order})'


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """Basic cell for rep-style block, including conv and bn"""
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        """Forward process"""
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepBlock(nn.Module):
    """
        RepBlock is a stage block with rep-style basic block
    """

    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.block = nn.Sequential(*(RepVGGBlock(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPadLayer(nn.Module):

    def __init__(self, pad_pixels, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def bn_weight(self):
        return self.bn.weight

    @property
    def bn_bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


def conv_bn_v2(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


class DiverseBranchBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=nn.ReLU(), single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = conv_bn_v2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn_v2(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3, kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))
        if single_init:
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)
        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0
        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second
        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels, kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride, padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))
        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, 'dbb_origin'):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, 'dbb_1x1'):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, 'dbb_avg'):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, 'dbb_1x1_kxk'):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, 'dbb_origin'):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DBBBlock(nn.Module):
    """
        RepBlock is a stage block with rep-style basic block
    """

    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = DiverseBranchBlock(in_channels, out_channels)
        self.block = nn.Sequential(*(DiverseBranchBlock(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


def fuse_model(model: 'nn.Module') ->nn.Module:
    """fuse conv and bn in model

    Args:
        model (nn.Module): model to fuse

    Returns:
        nn.Module: fused model
    """
    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, 'bn')
            m.forward = m.fuseforward
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info('Loading checkpoint from {}'.format(weights))
    ckpt = torch.load(weights, map_location=map_location)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info('\nFusing model...')
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


class DetectBackend(nn.Module):

    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):
        super().__init__()
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} format is not supported.'
        model = load_checkpoint(weights, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())

    def forward(self, im, val=False):
        y = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


class EfficientRep(nn.Module):
    """EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """

    def __init__(self, in_channels=3, channels_list=None, num_repeats=None):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None
        self.stem = RepVGGBlock(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)
        self.ERBlock_2 = nn.Sequential(RepVGGBlock(in_channels=channels_list[0], out_channels=channels_list[1], kernel_size=3, stride=2), RepBlock(in_channels=channels_list[1], out_channels=channels_list[1], n=num_repeats[1]))
        self.ERBlock_3 = nn.Sequential(RepVGGBlock(in_channels=channels_list[1], out_channels=channels_list[2], kernel_size=3, stride=2), RepBlock(in_channels=channels_list[2], out_channels=channels_list[2], n=num_repeats[2]))
        self.ERBlock_4 = nn.Sequential(RepVGGBlock(in_channels=channels_list[2], out_channels=channels_list[3], kernel_size=3, stride=2), RepBlock(in_channels=channels_list[3], out_channels=channels_list[3], n=num_repeats[3]))
        self.ERBlock_5 = nn.Sequential(RepVGGBlock(in_channels=channels_list[3], out_channels=channels_list[4], kernel_size=3, stride=2), RepBlock(in_channels=channels_list[4], out_channels=channels_list[4], n=num_repeats[4]), SimSPPF(in_channels=channels_list[4], out_channels=channels_list[4], kernel_size=5))

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        return tuple(outputs)


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class=torch.tensor([100]), iou_threshold=torch.tensor([0.45]), score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.max_obj = torch.tensor([max_obj])
        self.iou_threshold = torch.tensor([iou_thres])
        self.score_threshold = torch.tensor([score_thres])
        self.max_wh = max_wh
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=torch.float32, device=self.device)

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        objScore, objCls = score.max(2, keepdim=True)
        dis = objCls.float() * self.max_wh
        nmsbox = box + dis
        objScore1 = objScore.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, objScore1, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        resBoxes = box[X, Y, :]
        resClasses = objCls[X, Y, :].float()
        resScores = objScore[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, resBoxes, resClasses, resScores], 1)


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(ctx, boxes, scores, background_class=-1, box_coding=1, iou_threshold=0.45, max_output_boxes=100, plugin_version='1', score_activation=0, score_threshold=0.25):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes, scores, background_class=-1, box_coding=1, iou_threshold=0.45, max_output_boxes=100, plugin_version='1', score_activation=0, score_threshold=0.25):
        out = g.op('TRT::EfficientNMS_TRT', boxes, scores, background_class_i=background_class, box_coding_i=box_coding, iou_threshold_f=iou_threshold, max_output_boxes_i=max_output_boxes, plugin_version_s=plugin_version, score_activation_i=score_activation, score_threshold_f=score_threshold, outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(box, score, self.background_class, self.box_coding, self.iou_threshold, self.max_obj, self.plugin_version, self.score_activation, self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:, [2, 1, 0], ...]
            x = x * (1 / 255)
        x = self.model(x)
        x = self.end2end(x)
        return x


class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(self, channels_list=None, num_repeats=None):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None
        self.Rep_p4 = RepBlock(in_channels=channels_list[3] + channels_list[5], out_channels=channels_list[5], n=num_repeats[5])
        self.Rep_p3 = RepBlock(in_channels=channels_list[2] + channels_list[6], out_channels=channels_list[6], n=num_repeats[6])
        self.Rep_n3 = RepBlock(in_channels=channels_list[6] + channels_list[7], out_channels=channels_list[8], n=num_repeats[7])
        self.Rep_n4 = RepBlock(in_channels=channels_list[5] + channels_list[9], out_channels=channels_list[10], n=num_repeats[8])
        self.reduce_layer0 = SimConv(in_channels=channels_list[4], out_channels=channels_list[5], kernel_size=1, stride=1)
        self.upsample0 = Transpose(in_channels=channels_list[5], out_channels=channels_list[5])
        self.reduce_layer1 = SimConv(in_channels=channels_list[5], out_channels=channels_list[6], kernel_size=1, stride=1)
        self.upsample1 = Transpose(in_channels=channels_list[6], out_channels=channels_list[6])
        self.downsample2 = SimConv(in_channels=channels_list[6], out_channels=channels_list[7], kernel_size=3, stride=2)
        self.downsample1 = SimConv(in_channels=channels_list[8], out_channels=channels_list[9], kernel_size=3, stride=2)

    def forward(self, input):
        x2, x1, x0 = input
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)
        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)
        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)
        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs


class SP(nn.Module):

    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class ReOrg(nn.Module):

    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Chuncat(nn.Module):

    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1 + x2, self.d)


class Shortcut(nn.Module):

    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0] + x[1]


class Foldcut(nn.Module):

    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1 + x2


class RobustConv(nn.Module):

    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True, layer_scale_init_value=1e-06):
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


class RobustConv2(nn.Module):

    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-06):
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s, padding=0, bias=True, dilation=1, groups=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


class Stem(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Stem, self).__init__()
        c_ = int(c2 / 2)
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):

    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2 // 2, 3, k)
        self.cv3 = Conv(c1, c2 // 2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class Res(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Res, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):

    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, shortcu, g, e)
        c_ = int(c2 * e)


class Ghost(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SPPCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class GhostSPPCSPC(SPPCSPC):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2 / 2)
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)


class BottleneckCSPA(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):

    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class RepBottleneck(Bottleneck):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):

    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):

    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):

    def __init__(self, c1, c2, n=1, shortcut=False, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):

    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0, 'feature map h and w can not divide by window size'
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, _, H_, W_ = x.shape
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            Padding = True
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W)
        else:
            attn_mask = None
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        if Padding:
            x = x[:, :, :H_, :W_]
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(STCSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class WindowAttention_v2(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class Mlp_v2(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_v2(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_v2(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer_v2(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_v2(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, pretrained_window_size=(pretrained_window_size, pretrained_window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, _, H_, W_ = x.shape
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            Padding = True
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W)
        else:
            attn_mask = None
        shortcut = x
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition_v2(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        if Padding:
            x = x[:, :, :H_, :W_]
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class SwinTransformer2Block(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class ST2CSPA(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(ST2CSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPB(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(ST2CSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(ST2CSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()
        None
        self.stride = model.stride
        self.names = model.names
        self.model = model
        self.model = revert_sync_batchnorm(self.model)
        self.model
        self.model.eval()
        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        rand_example = torch.rand(1, 3, img_size, img_size)
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        traced_script_module.save('traced_model.pt')
        None
        self.model = traced_script_module
        self.model
        self.detect_layer
        None

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out


class ChannelAttention(nn.Module):

    def __init__(self, channels: 'int') ->None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):

    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class SegmentationModel(DetectionModel):

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H * hs, W * ws)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class EmptyModule(nn.Module):

    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0] - w1 / 2.0, boxes2[0] - w2 / 2.0)
        x2_max = torch.max(boxes1[0] + w1 / 2.0, boxes2[0] + w2 / 2.0)
        y1_min = torch.min(boxes1[1] - h1 / 2.0, boxes2[1] - h2 / 2.0)
        y2_max = torch.max(boxes1[1] + h1 / 2.0, boxes2[1] + h2 / 2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (w_cross <= 0) + (h_cross <= 0) > 0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


class RegionLayer(nn.Module):

    def __init__(self, num_classes=0, anchors=[], num_anchors=1, use_cuda=None):
        super(RegionLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.anchors = torch.FloatTensor(anchors).view(self.num_anchors, self.anchor_step)
        self.rescore = 1
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def build_targets(self, pred_boxes, target, nH, nW):
        nB = target.size(0)
        nA = self.num_anchors
        conf_mask = torch.ones(nB, nA, nH, nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        anchors = self.anchors
        if self.seen < 12800:
            tcoord[0].fill_(0.5)
            tcoord[1].fill_(0.5)
            coord_mask.fill_(1)
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5)
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gw = [(i * nW) for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [(i * nH) for i in (tbox[t][2], tbox[t][4])]
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = cur_ious > self.thresh
            conf_mask[b][ignore_ix.view(nA, nH, nW)] = 0
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gw = [(i * nW) for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [(i * nH) for i in (tbox[t][2], tbox[t][4])]
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors), 1).t()
                tmp_ious = multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False)
                best_iou, best_n = torch.max(tmp_ious, 0)
                if self.anchor_step == 4:
                    tmp_ious_mask = tmp_ious == best_iou
                    if tmp_ious_mask.sum() > 0:
                        gt_pos = torch.FloatTensor([gi, gj, gx, gy]).repeat(nA, 1).t()
                        an_pos = anchor_boxes[4:6]
                        dist = pow(gt_pos[0] + an_pos[0] - gt_pos[2], 2) + pow(gt_pos[1] + an_pos[1] - gt_pos[3], 2)
                        dist[1 - tmp_ious_mask] = 10000
                        _, best_n = torch.min(dist, 0)
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = self.object_scale
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.0
                if iou > 0.5:
                    nRecall += 1
        return nGT, nRecall, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def get_mask_boxes(self, output):
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step)
        masked_anchors = self.anchors.view(-1)
        num_anchors = torch.IntTensor([self.num_anchors])
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}

    def forward(self, output, target):
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB * nA * nH * nW
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step)
        output = output.view(nB, nA, 5 + nC, nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long()
        ix = torch.LongTensor(range(0, 5))
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim)
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1, cls_anchor_dim)
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim)
        anchor_w = self.anchors.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        anchor_h = self.anchors.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
        t2 = time.time()
        nGT, nRecall, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target.detach(), nH, nW)
        cls_mask = cls_mask == 1
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
        cls = cls[cls_mask].view(-1, nC)
        nProposals = int((conf > 0.25).sum())
        tcoord = tcoord.view(4, cls_anchor_dim)
        tconf, tcls = tconf, tcls
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim), conf_mask.sqrt()
        t3 = time.time()
        loss_coord = self.coord_scale * nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / 2
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / 2
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        loss = loss_coord + loss_conf + loss_cls
        t4 = time.time()
        if False:
            None
            None
            None
            None
            None
            None
        None
        if math.isnan(loss.item()):
            None
            sys.exit(0)
        return loss


class YoloLayer(nn.Module):

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.rescore = 0
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.0
        self.stride = 32
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0

    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [(anchor / self.stride) for anchor in masked_anchors]
        masked_anchors = torch.FloatTensor(masked_anchors)
        num_anchors = torch.IntTensor([len(self.anchor_mask)])
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        nB = target.size(0)
        anchor_step = anchors.size(1)
        conf_mask = torch.ones(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
        twidth, theight = self.net_width / self.stride, self.net_height / self.stride
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0
        anchors = anchors
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5)
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = cur_ious > self.ignore_thresh
            conf_mask[b][ignore_ix.view(nA, nH, nW)] = 0
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors), 1).t()
                _, best_n = torch.max(multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False), 0)
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = 1
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.0
                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1
        return nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def forward(self, output, target):
        mask_tuple = self.get_mask_boxes(output)
        t0 = time.time()
        nB = output.data.size(0)
        nA = mask_tuple['n'].item()
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0) // nA
        anchors = mask_tuple['a'].view(nA, anchor_step)
        cls_anchor_dim = nB * nA * nH * nW
        output = output.view(nB, nA, 5 + nC, nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long()
        ix = torch.LongTensor(range(0, 5))
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim)
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1, cls_anchor_dim)
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim)
        anchor_w = anchors.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
        t2 = time.time()
        nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)
        cls_mask = cls_mask == 1
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
        cls = cls[cls_mask].view(-1, nC)
        nProposals = int((conf > 0.25).sum())
        tcoord = tcoord.view(4, cls_anchor_dim)
        tconf, tcls = tconf, tcls
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim), conf_mask
        t3 = time.time()
        loss_coord = nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / 2
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask)
        loss_cls = nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        loss = loss_coord + loss_conf + loss_cls
        t4 = time.time()
        if False:
            None
            None
            None
            None
            None
            None
        None
        if math.isnan(loss.item()):
            None
            sys.exit(0)
        return loss


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class AMSoftmax(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_feat: size of each input sample
        num_classes: size of each output sample
    """

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_features = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        phi = cosine - self.m
        targets = F.one_hot(targets, num_classes=self._num_classes)
        output = targets * phi + (1.0 - targets) * cosine
        output *= self.s
        return output

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(self.in_feat, self._num_classes, self.s, self.m)


class ArcSoftmax(nn.Module):

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self.s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(self.in_feat, self._num_classes, self.s, self.m)


class BatchDrop(nn.Module):
    """ref: https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    batch drop mask
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class IBN(nn.Module):

    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class CircleSoftmax(nn.Module):

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.0)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)
        targets = F.one_hot(targets, num_classes=self._num_classes)
        pred_class_logits = targets * s_p + (1.0 - targets) * s_n
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(self.in_feat, self._num_classes, self.s, self.m)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        if hasattr(m[-1], 'bias') and m[-1].bias is not None:
            nn.init.constant_(m[-1].bias, 0)
    else:
        nn.init.constant_(m.weight, val=0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
                nn.init.constant_(self.conv_mask.bias, 0)
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class TLU(nn.Module):

    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau.view(1, self.num_features, 1, 1))


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-06, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        if is_eps_leanable:
            self.eps = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.weight.view(1, self.num_features, 1, 1) * x + self.bias.view(1, self.num_features, 1, 1)
        return x


class Non_local(nn.Module):

    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), get_norm(bn_norm, self.in_channels))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
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
        return z


class GeneralizedMeanPooling(nn.Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-06):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.p) + ', ' + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-06):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class FastGlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.gap = FastGlobalAvgPool2d()
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat


class ClipGlobalAvgPool2d(nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, num_splits=1, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = get_norm(norm_layer, channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm_layer, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, bn_norm):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = nn.Identity()
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, bn_norm, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels, bn_norm)
        self.conv2a = LightConv3x3(mid_channels, mid_channels, bn_norm)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn_norm)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels, bn_norm)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return self.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, blocks, layers, channels, bn_norm, IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.conv1 = ConvLayer(3, channels[0], 7, bn_norm, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], bn_norm, reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], bn_norm, reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], bn_norm, reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3], bn_norm)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, bn_norm, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, bn_norm, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, bn_norm, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels, bn_norm), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class EffHead(nn.Module):
    """EfficientNet head: 1x1, BN, Swish, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, bn_norm):
        super(EffHead, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = get_norm(bn_norm, w_out)
        self.conv_swish = Swish()

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.construct(w_in, w_se)

    def construct(self, w_in, w_se):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, kernel_size=1, bias=True), nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE), nn.Conv2d(w_se, w_in, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, bn_norm):
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1, padding=0, bias=False)
            self.exp_bn = get_norm(bn_norm, w_exp)
            self.exp_swish = Swish()
        dwise_args = {'groups': w_exp, 'padding': (kernel - 1) // 2, 'bias': False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel, stride=stride, **dwise_args)
        self.dwise_bn = get_norm(bn_norm, w_exp)
        self.dwise_swish = Swish()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = get_norm(bn_norm, w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and effnet_cfg.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, effnet_cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d, bn_norm):
        super(EffStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = 'b{}'.format(i + 1)
            self.add_module(name, MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out, bn_norm))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out, bn_norm):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn and regnet_cfg.BN.ZERO_INIT_FINAL_GAMMA
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class EffNet(nn.Module):
    """EfficientNet model."""

    @staticmethod
    def get_args():
        return {'stem_w': effnet_cfg.EN.STEM_W, 'ds': effnet_cfg.EN.DEPTHS, 'ws': effnet_cfg.EN.WIDTHS, 'exp_rs': effnet_cfg.EN.EXP_RATIOS, 'se_r': effnet_cfg.EN.SE_R, 'ss': effnet_cfg.EN.STRIDES, 'ks': effnet_cfg.EN.KERNELS, 'head_w': effnet_cfg.EN.HEAD_W}

    def __init__(self, last_stride, bn_norm, **kwargs):
        super(EffNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        self._construct(**kwargs, last_stride=last_stride, bn_norm=bn_norm)
        self.apply(init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, last_stride, bn_norm):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w, bn_norm)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            if i == 5:
                stride = last_stride
            self.add_module(name, EffStage(prev_w, exp_r, kernel, stride, se_r, w, d, bn_norm))
            prev_w = w
        self.head = EffHead(prev_w, head_w, bn_norm)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Vanilla block does not support bm, gw, and se_r options'
        super(VanillaBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm):
        super(BasicTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Basic transform does not support bm, gw, and se_r options'
        super(ResBasicBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BasicTransform(w_in, w_out, stride, bn_norm)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        num_gs = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = get_norm(bn_norm, w_b)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = get_norm(bn_norm, w_b)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = get_norm(bn_norm, w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BottleneckTransform(w_in, w_out, stride, bn_norm, bm, gw, se_r)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemCifar, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemIN, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w, bn_norm):
        super(SimpleStemIN, self).__init__()
        self.construct(in_w, out_w, bn_norm)

    def construct(self, in_w, out_w, bn_norm):
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = get_norm(bn_norm, out_w)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            self.add_module('b{}'.format(i + 1), block_fun(b_w_in, w_out, b_stride, bn_norm, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {'vanilla_block': VanillaBlock, 'res_basic_block': ResBasicBlock, 'res_bottleneck_block': ResBottleneckBlock}
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(block_type)
    return block_funs[block_type]


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {'res_stem_cifar': ResStemCifar, 'res_stem_in': ResStemIN, 'simple_stem_in': SimpleStemIN}
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(stem_type)
    return stem_funs[stem_type]


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self.construct(stem_type=kwargs['stem_type'], stem_w=kwargs['stem_w'], block_type=kwargs['block_type'], ds=kwargs['ds'], ws=kwargs['ws'], ss=kwargs['ss'], bn_norm=kwargs['bn_norm'], bms=kwargs['bms'], gws=kwargs['gws'], se_r=kwargs['se_r'])
        else:
            self.construct(stem_type=regnet_cfg.ANYNET.STEM_TYPE, stem_w=regnet_cfg.ANYNET.STEM_W, block_type=regnet_cfg.ANYNET.BLOCK_TYPE, ds=regnet_cfg.ANYNET.DEPTHS, ws=regnet_cfg.ANYNET.WIDTHS, ss=regnet_cfg.ANYNET.STRIDES, bn_norm=regnet_cfg.ANYNET.BN_NORM, bms=regnet_cfg.ANYNET.BOT_MULS, gws=regnet_cfg.ANYNET.GROUP_WS, se_r=regnet_cfg.ANYNET.SE_R if regnet_cfg.ANYNET.SE_ON else None)
        self.apply(init_weights)

    def construct(self, stem_type, stem_w, block_type, ds, ws, ss, bn_norm, bms, gws, se_r):
        bms = bms if bms else [(1.0) for _d in ds]
        gws = gws if gws else [(1) for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w, bn_norm)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), AnyStage(prev_w, w, s, bn_norm, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.in_planes = prev_w

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class ResNest(nn.Module):
    """ResNet Variants ResNest
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, last_stride, bn_norm, with_ibn, with_nl, block, layers, non_layers, radix=1, groups=1, bottleneck_width=64, dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, rectified_conv=False, rectify_avg=False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0, last_gamma=False):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super().__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs), get_norm(bn_norm, stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs), get_norm(bn_norm, stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs)
        self.bn1 = get_norm(bn_norm, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn=with_ibn, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn=with_ibn)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], 1, bn_norm, with_ibn=with_ibn, dilation=2, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], 1, bn_norm, with_ibn=with_ibn, dilation=4, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn=with_ibn, dilation=1, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], 1, bn_norm, with_ibn=with_ibn, dilation=2, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn=with_ibn, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_ibn=with_ibn, dropblock_prob=dropblock_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN', with_ibn=False, dilation=1, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(get_norm(bn_norm, planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=dilation, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList([Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([(layers[0] - (i + 1)) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList([Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([(layers[1] - (i + 1)) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([(layers[2] - (i + 1)) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([(layers[3] - (i + 1)) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        return x


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, last_stride, bn_norm, with_ibn, with_nl, block, layers, non_layers, baseWidth=4, cardinality=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64
        self.output_size = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn=with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn=with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_ibn=with_ibn)
        self.random_init()
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN', with_ibn=False):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), get_norm(bn_norm, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, self.baseWidth, self.cardinality, 1, None))
        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList([Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([(layers[0] - (i + 1)) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList([Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([(layers[1] - (i + 1)) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([(layers[2] - (i + 1)) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([(layers[3] - (i + 1)) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        return x

    def random_init(self):
        self.conv1.weight.data.normal_(0, math.sqrt(2.0 / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class AttrHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == 'avgmaxpool':
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == 'identity':
            self.pool_layer = nn.Identity()
        elif pool_type == 'flatten':
            self.pool_layer = Flatten()
        else:
            raise KeyError(f'{pool_type} is not supported!')
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':
            self.classifier = AMSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f'{cls_type} is not supported!')
        bottleneck = [nn.BatchNorm1d(num_classes)]
        self.bottleneck = nn.Sequential(*bottleneck)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]
        classifier_name = self.classifier.__class__.__name__
        if classifier_name == 'Linear':
            cls_outputs = self.classifier(global_feat)
        else:
            cls_outputs = self.classifier(global_feat, targets)
        cls_outputs = self.bottleneck(cls_outputs)
        if self.training:
            return {'cls_outputs': cls_outputs}
        else:
            cls_outputs = torch.sigmoid(cls_outputs)
            return cls_outputs


class EmbeddingHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == 'avgmaxpool':
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == 'identity':
            self.pool_layer = nn.Identity()
        elif pool_type == 'flatten':
            self.pool_layer = Flatten()
        else:
            raise KeyError(f'{pool_type} is not supported!')
        self.neck_feat = neck_feat
        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim
        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bottleneck = nn.Sequential(*bottleneck)
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':
            self.classifier = AMSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f'{cls_type} is not supported!')
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        if not self.training:
            return bn_feat
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat), F.normalize(self.classifier.weight))
        if self.neck_feat == 'before':
            feat = global_feat[..., 0, 0]
        elif self.neck_feat == 'after':
            feat = bn_feat
        else:
            raise KeyError(f'{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT')
        return {'cls_outputs': cls_outputs, 'pred_class_logits': pred_class_logits, 'features': feat}


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class SmoothAP_old(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super().__init__()
        assert batch_size % num_id == 0
        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """
        preds = F.normalize(preds, dim=1)
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        sim_all = torch.mm(preds, preds.t())
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
        ap = torch.zeros(1)
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / sim_all_rk[ind * group:(ind + 1) * group, ind * group:(ind + 1) * group])
            ap = ap + pos_divide / group / self.batch_size
        return 1 - ap


REID_HEADS_REGISTRY = Registry('HEADS')


def build_heads(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def circle_loss(embedding: 'torch.Tensor', targets: 'torch.Tensor', margin: 'float', alpha: 'float') ->torch.Tensor:
    embedding = nn.functional.normalize(embedding, dim=1)
    if comm.get_world_size() > 1:
        all_embedding = concat_all_gather(embedding)
        all_targets = concat_all_gather(targets)
    else:
        all_embedding = embedding
        all_targets = targets
    dist_mat = torch.matmul(all_embedding, all_embedding.t())
    N = dist_mat.size(0)
    is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t()).float()
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())
    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.0)
    delta_p = 1 - margin
    delta_n = margin
    logit_p = -alpha * alpha_p * (s_p - delta_p)
    logit_n = alpha * alpha_n * (s_n - delta_n)
    loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    return loss


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)
    if eps >= 0:
        smooth_param = eps
    else:
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)
    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), 1 - smooth_param)
    loss = (-targets * log_probs).sum(dim=1)
    """
    # confidence penalty
    conf_penalty = 0.3
    probs = F.softmax(pred_class_logits, dim=1)
    entropy = torch.sum(-probs * log_probs, dim=1)
    loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
    """
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    loss = loss.sum() / non_zero_cnt
    return loss


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_STORAGE_STACK), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))
    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1.0 / bsz))
    storage = get_event_storage()
    storage.put_scalar('cls_accuracy', ret[0])


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    N = dist_mat.size(0)
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-06
    W = torch.exp(diff) * mask / Z
    return W


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2
    is_pos = is_pos.float()
    is_neg = is_neg.float()
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg
    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)
    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    if norm_feat:
        embedding = normalize(embedding, axis=-1)
    if comm.get_world_size() > 1:
        all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        all_targets = concat_all_gather(targets)
    else:
        all_embedding = embedding
        all_targets = targets
    dist_mat = euclidean_dist(all_embedding, all_embedding)
    N, N = dist_mat.size()
    is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())
    is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())
    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    return loss


class Baseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer('pixel_mean', torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        if self.training:
            assert 'targets' in batched_inputs, 'Person ID annotation are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            outputs = self.heads(features, targets)
            return {'outputs': outputs, 'targets': targets}
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError('batched_inputs must be dict or torch.Tensor, but get {}'.format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outs):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        outputs = outs['outputs']
        gt_labels = outs['targets']
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        log_accuracy(pred_class_logits, gt_labels)
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME
        if 'CrossEntropyLoss' in loss_names:
            loss_dict['loss_cls'] = cross_entropy_loss(cls_outputs, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE
        if 'TripletLoss' in loss_names:
            loss_dict['loss_triplet'] = triplet_loss(pred_features, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE
        if 'CircleLoss' in loss_names:
            loss_dict['loss_circle'] = circle_loss(pred_features, gt_labels, self._cfg.MODEL.LOSSES.CIRCLE.MARGIN, self._cfg.MODEL.LOSSES.CIRCLE.ALPHA) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE
        return loss_dict


class MGN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer('pixel_mean', torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer('pixel_std', torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        bn_norm = cfg.MODEL.BACKBONE.NORM
        with_se = cfg.MODEL.BACKBONE.WITH_SE
        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3[0])
        res_conv4 = nn.Sequential(*backbone.layer3[1:])
        res_g_conv5 = backbone.layer4
        res_p_conv5 = nn.Sequential(Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))), Bottleneck(2048, 512, bn_norm, False, with_se), Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(backbone.layer4.state_dict())
        self.b1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.b1_head = build_heads(cfg)
        self.b2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.b2_head = build_heads(cfg)
        self.b21_head = build_heads(cfg)
        self.b22_head = build_heads(cfg)
        self.b3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.b3_head = build_heads(cfg)
        self.b31_head = build_heads(cfg)
        self.b32_head = build_heads(cfg)
        self.b33_head = build_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        b1_feat = self.b1(features)
        b2_feat = self.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        b3_feat = self.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)
        if self.training:
            assert 'targets' in batched_inputs, 'Person ID annotation are missing in training!'
            targets = batched_inputs['targets'].long()
            if targets.sum() < 0:
                targets.zero_()
            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b21_outputs = self.b21_head(b21_feat, targets)
            b22_outputs = self.b22_head(b22_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            b31_outputs = self.b31_head(b31_feat, targets)
            b32_outputs = self.b32_head(b32_feat, targets)
            b33_outputs = self.b33_head(b33_feat, targets)
            return {'b1_outputs': b1_outputs, 'b2_outputs': b2_outputs, 'b21_outputs': b21_outputs, 'b22_outputs': b22_outputs, 'b3_outputs': b3_outputs, 'b31_outputs': b31_outputs, 'b32_outputs': b32_outputs, 'b33_outputs': b33_outputs, 'targets': targets}
        else:
            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b21_pool_feat = self.b21_head(b21_feat)
            b22_pool_feat = self.b22_head(b22_feat)
            b3_pool_feat = self.b3_head(b3_feat)
            b31_pool_feat = self.b31_head(b31_feat)
            b32_pool_feat = self.b32_head(b32_feat)
            b33_pool_feat = self.b33_head(b33_feat)
            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat, b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
            return pred_feat

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError('batched_inputs must be dict or torch.Tensor, but get {}'.format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outs):
        b1_outputs = outs['b1_outputs']
        b2_outputs = outs['b2_outputs']
        b21_outputs = outs['b21_outputs']
        b22_outputs = outs['b22_outputs']
        b3_outputs = outs['b3_outputs']
        b31_outputs = outs['b31_outputs']
        b32_outputs = outs['b32_outputs']
        b33_outputs = outs['b33_outputs']
        gt_labels = outs['targets']
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits = b1_outputs['cls_outputs']
        b2_logits = b2_outputs['cls_outputs']
        b21_logits = b21_outputs['cls_outputs']
        b22_logits = b22_outputs['cls_outputs']
        b3_logits = b3_outputs['cls_outputs']
        b31_logits = b31_outputs['cls_outputs']
        b32_logits = b32_outputs['cls_outputs']
        b33_logits = b33_outputs['cls_outputs']
        b1_pool_feat = b1_outputs['features']
        b2_pool_feat = b2_outputs['features']
        b3_pool_feat = b3_outputs['features']
        b21_pool_feat = b21_outputs['features']
        b22_pool_feat = b22_outputs['features']
        b31_pool_feat = b31_outputs['features']
        b32_pool_feat = b32_outputs['features']
        b33_pool_feat = b33_outputs['features']
        log_accuracy(pred_class_logits, gt_labels)
        b22_pool_feat = torch.cat((b21_pool_feat, b22_pool_feat), dim=1)
        b33_pool_feat = torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME
        if 'CrossEntropyLoss' in loss_names:
            loss_dict['loss_cls_b1'] = cross_entropy_loss(b1_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b2'] = cross_entropy_loss(b2_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b21'] = cross_entropy_loss(b21_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b22'] = cross_entropy_loss(b22_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b3'] = cross_entropy_loss(b3_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b31'] = cross_entropy_loss(b31_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b32'] = cross_entropy_loss(b32_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
            loss_dict['loss_cls_b33'] = cross_entropy_loss(b33_logits, gt_labels, self._cfg.MODEL.LOSSES.CE.EPSILON, self._cfg.MODEL.LOSSES.CE.ALPHA) * self._cfg.MODEL.LOSSES.CE.SCALE * 0.125
        if 'TripletLoss' in loss_names:
            loss_dict['loss_triplet_b1'] = triplet_loss(b1_pool_feat, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE * 0.2
            loss_dict['loss_triplet_b2'] = triplet_loss(b2_pool_feat, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE * 0.2
            loss_dict['loss_triplet_b3'] = triplet_loss(b3_pool_feat, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE * 0.2
            loss_dict['loss_triplet_b22'] = triplet_loss(b22_pool_feat, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE * 0.2
            loss_dict['loss_triplet_b33'] = triplet_loss(b33_pool_feat, gt_labels, self._cfg.MODEL.LOSSES.TRI.MARGIN, self._cfg.MODEL.LOSSES.TRI.NORM_FEAT, self._cfg.MODEL.LOSSES.TRI.HARD_MINING) * self._cfg.MODEL.LOSSES.TRI.SCALE * 0.2
        return loss_dict


class ShuffleV2Block(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)]
        self.branch_main = nn.Sequential(*branch_main)
        if stride == 2:
            branch_proj = [nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.Conv2d(inp, inp, 1, 1, 0, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True)]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """

    def __init__(self, input_size=224, n_class=1000, model_size='1.5x'):
        super(ShuffleNetV2, self).__init__()
        None
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.conv_last = nn.Sequential(nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False), nn.BatchNorm2d(self.stage_out_channels[-1]), nn.ReLU(inplace=True))
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ShuffleNetV2Backbone(nn.Module):

    def __init__(self, model_size, pretrained=False, pretrain_path=''):
        super(ShuffleNetV2Backbone, self).__init__()
        model = ShuffleNetV2(model_size=model_size)
        if pretrained:
            new_state_dict = OrderedDict()
            state_dict = torch.load(pretrain_path)['state_dict']
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    k = k[7:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)
        self.backbone = nn.Sequential(model.first_conv, model.maxpool, model.features, model.conv_last)

    def forward(self, x):
        return self.backbone(x)


class OcclusionUnit(nn.Module):

    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=False)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)
        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, 0])
        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)
        mask_score = mask_score.unsqueeze(1)
        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) * SpaFeat1.size(2), -1))
        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


class DSRHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        norm_type = cfg.MODEL.HEADS.NORM
        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == 'avgmaxpool':
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == 'identity':
            self.pool_layer = nn.Identity()
        elif pool_type == 'flatten':
            self.pool_layer = Flatten()
        else:
            raise KeyError(f'{pool_type} is not supported!')
        self.neck_feat = neck_feat
        self.occ_unit = OcclusionUnit(in_planes=feat_dim)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.bnneck = get_norm(norm_type, feat_dim, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck_occ = get_norm(norm_type, feat_dim, bias_freeze=True)
        self.bnneck_occ.apply(weights_init_kaiming)
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
            self.classifier_occ = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
            self.classifier_occ = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
            self.classifier_occ = CircleSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from 'linear', 'arcSoftmax' and 'circleSoftmax'.")
        self.classifier.apply(weights_init_classifier)
        self.classifier_occ.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        SpaFeat1 = self.MaxPool1(features)
        SpaFeat2 = self.MaxPool2(features)
        SpaFeat3 = self.MaxPool3(features)
        SpaFeat4 = self.MaxPool4(features)
        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), dim=2)
        foreground_feat, mask_weight, mask_weight_norm = self.occ_unit(features)
        bn_foreground_feat = self.bnneck_occ(foreground_feat)
        bn_foreground_feat = bn_foreground_feat[..., 0, 0]
        if not self.training:
            return bn_foreground_feat, SpatialFeatAll, mask_weight_norm
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat), F.normalize(self.classifier.weight))
        return {'cls_outputs': cls_outputs, 'fore_cls_outputs': fore_cls_outputs, 'pred_class_logits': pred_class_logits, 'global_features': global_feat[..., 0, 0], 'foreground_features': foreground_feat[..., 0, 0]}


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class TridentBottleneck(Bottleneck):
    """BottleBlock for TridentResNet.

    Args:
        trident_dilations (tuple[int, int, int]): Dilations of different
            trident branch.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        concat_output (bool): Whether to concat the output list to a Tensor.
            `True` only in the last Block.
    """

    def __init__(self, trident_dilations, test_branch_idx, concat_output, **kwargs):
        super(TridentBottleneck, self).__init__(**kwargs)
        self.trident_dilations = trident_dilations
        self.num_branch = len(trident_dilations)
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        self.conv2 = TridentConv(self.planes, self.planes, kernel_size=3, stride=self.conv2_stride, bias=False, trident_dilations=self.trident_dilations, test_branch_idx=test_branch_idx, init_cfg=dict(type='Kaiming', distribution='uniform', mode='fan_in', override=dict(name='conv2')))

    def forward(self, x):

        def _inner_forward(x):
            num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
            identity = x
            if not isinstance(x, list):
                x = (x,) * num_branch
                identity = x
                if self.downsample is not None:
                    identity = [self.downsample(b) for b in x]
            out = [self.conv1(b) for b in x]
            out = [self.norm1(b) for b in out]
            out = [self.relu(b) for b in out]
            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k], self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = [self.norm2(b) for b in out]
            out = [self.relu(b) for b in out]
            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k], self.after_conv2_plugin_names)
            out = [self.conv3(b) for b in out]
            out = [self.norm3(b) for b in out]
            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k], self.after_conv3_plugin_names)
            out = [(out_b + identity_b) for out_b, identity_b in zip(out, identity)]
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = [self.relu(b) for b in out]
        if self.concat_output:
            out = torch.cat(out, dim=0)
        return out


def make_trident_res_layer(block, inplanes, planes, num_blocks, stride=1, trident_dilations=(1, 2, 3), style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None, test_branch_idx=-1):
    """Build Trident Res Layers."""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
        downsample = nn.Sequential(*downsample)
    layers = []
    for i in range(num_blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride if i == 0 else 1, trident_dilations=trident_dilations, downsample=downsample if i == 0 else None, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=plugins, test_branch_idx=test_branch_idx, concat_output=True if i == num_blocks - 1 else False))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


class CenterPrior(nn.Module):
    """Center Weighting module to adjust the category-specific prior
    distributions.

    Args:
        force_topk (bool): When no point falls into gt_bbox, forcibly
            select the k points closest to the center to calculate
            the center prior. Defaults to False.
        topk (int): The number of points used to calculate the
            center prior when no point falls in gt_bbox. Only work when
            force_topk if True. Defaults to 9.
        num_classes (int): The class number of dataset. Defaults to 80.
        strides (tuple[int]): The stride of each input feature map. Defaults
            to (8, 16, 32, 64, 128).
    """

    def __init__(self, force_topk=False, topk=9, num_classes=80, strides=(8, 16, 32, 64, 128)):
        super(CenterPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list, gt_bboxes, labels, inside_gt_bbox_mask):
        """Get the center prior of each point on the feature map for each
        instance.

        Args:
            anchor_points_list (list[Tensor]): list of coordinate
                of points on feature map. Each with shape
                (num_points, 2).
            gt_bboxes (Tensor): The gt_bboxes with shape of
                (num_gt, 4).
            labels (Tensor): The gt_labels with shape of (num_gt).
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple(Tensor):

                - center_prior_weights(Tensor): Float tensor with shape                     of (num_points, num_gt). Each value represents                     the center weighting coefficient.
                - inside_gt_bbox_mask (Tensor): Tensor of bool type,                     with shape of (num_points, num_gt), each                     value is used to mark whether this point falls                     within a certain gt or is the topk nearest points for                     a specific gt_bbox.
        """
        inside_gt_bbox_mask = inside_gt_bbox_mask.clone()
        num_gts = len(labels)
        num_points = sum([len(item) for item in anchor_points_list])
        if num_gts == 0:
            return gt_bboxes.new_zeros(num_points, num_gts), inside_gt_bbox_mask
        center_prior_list = []
        for slvl_points, stride in zip(anchor_points_list, self.strides):
            single_level_points = slvl_points[:, None, :].expand((slvl_points.size(0), len(gt_bboxes), 2))
            gt_center_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
            gt_center_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
            gt_center = torch.stack((gt_center_x, gt_center_y), dim=1)
            gt_center = gt_center[None]
            instance_center = self.mean[labels][None]
            instance_sigma = self.sigma[labels][None]
            distance = ((single_level_points - gt_center) / float(stride) - instance_center) ** 2
            center_prior = torch.exp(-distance / (2 * instance_sigma ** 2)).prod(dim=-1)
            center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        if self.force_topk:
            gt_inds_no_points_inside = torch.nonzero(inside_gt_bbox_mask.sum(0) == 0).reshape(-1)
            if gt_inds_no_points_inside.numel():
                topk_center_index = center_prior_weights[:, gt_inds_no_points_inside].topk(self.topk, dim=0)[1]
                temp_mask = inside_gt_bbox_mask[:, gt_inds_no_points_inside]
                inside_gt_bbox_mask[:, gt_inds_no_points_inside] = torch.scatter(temp_mask, dim=0, index=topk_center_index, src=torch.ones_like(topk_center_index, dtype=torch.bool))
        center_prior_weights[~inside_gt_bbox_mask] = 0
        return center_prior_weights, inside_gt_bbox_mask


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1), nn.ReLU(inplace=True), nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0), nn.Sigmoid())
        self.reduction_conv = ConvModule(self.in_channels, self.feat_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)
        return feat


class Accuracy(nn.Module):

    def __init__(self, topk=(1,), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)


class AssociativeEmbeddingLoss(nn.Module):
    """Associative Embedding Loss.

    More details can be found in
    `Associative Embedding <https://arxiv.org/abs/1611.05424>`_ and
    `CornerNet <https://arxiv.org/abs/1808.01244>`_ .
    Code is modified from `kp_utils.py <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L180>`_  # noqa: E501

    Args:
        pull_weight (float): Loss weight for corners from same object.
        push_weight (float): Loss weight for corners from different object.
    """

    def __init__(self, pull_weight=0.25, push_weight=0.25):
        super(AssociativeEmbeddingLoss, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(self, pred, target, match):
        """Forward function."""
        batch = pred.size(0)
        pull_all, push_all = 0.0, 0.0
        for i in range(batch):
            pull, push = ae_loss_per_image(pred[i], target[i], match[i])
            pull_all += self.pull_weight * pull
            push_all += self.push_weight * push
        return pull_all, push_all


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Args:
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss. Defaults to 1.5.
        beta (float, optional): The loss is a piecewise function of prediction
            and target. ``beta`` serves as a threshold for the difference
            between the prediction and target. Defaults to 1.0.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, reduction='mean', loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 4).
            target (torch.Tensor): The learning target of the prediction with
                shape (N, 4).
            weight (torch.Tensor, optional): Sample-wise loss weight with
                shape (N, ).
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * balanced_l1_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None, ignore_index=None, **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, ignore_index=None, loss_weight=1.0, avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ignore_index is not None and not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn('Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.')
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=None, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, ignore_index=ignore_index, avg_non_ignore=self.avg_non_ignore, **kwargs)
        return loss_cls


def dice_loss(pred, target, weight=None, eps=0.001, reduction='mean', naive_dice=False, avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    input = pred.flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = 2 * a / (b + c)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, activate=True, reduction='mean', naive_dice=False, loss_weight=1.0, eps=0.001):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self, pred, target, weight=None, reduction_override=None, avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError
        loss = self.loss_weight * dice_loss(pred, target, weight, eps=self.eps, reduction=reduction, naive_dice=self.naive_dice, avg_factor=avg_factor)
        return loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_reg = self.loss_weight * gaussian_focal_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, reduction=reduction, avg_factor=avg_factor)
        return loss_reg


@weighted_loss
def quality_focal_loss_with_prob(pred, target, beta=2.0):
    """Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Different from `quality_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    label, score = target
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction='mean', loss_weight=1.0, activated=False):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                calculate_loss_func = quality_focal_loss
            loss_cls = self.loss_weight * calculate_loss_func(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0, reduction='mean'):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-06
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, label_weight, reduction_override=None, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = weight_reduce_loss(loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight


class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0, reduction='mean'):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] = 1000.0
        self.momentum = momentum
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, label_weight, avg_factor=None, reduction_override=None):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)
        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n
        loss = weight_reduce_loss(loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, linear=False, eps=1e-06, reduction='mean', loss_weight=1.0, mode='log'):
        super(IoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in IOULoss is deprecated, please use "mode=`linear`" instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and not torch.any(weight > 0) and reduction != 'none':
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(pred, target, weight, mode=self.mode, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=0.001, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(pred, target, weight, beta=self.beta, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class DIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * diou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class CIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * ciou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self, pred, soft_label, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor, T=self.T)
        return loss_kd


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


def seesaw_ce_loss(cls_score, labels, label_weights, cum_samples, num_classes, p, q, eps, reduction='mean', avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        label_weights (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes
    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[torch.arange(0, len(scores)).long(), labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor
    cls_score = cls_score + seesaw_weights.log() * (1 - onehot_labels)
    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')
    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv: https://arxiv.org/abs/2008.10032

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self, use_sigmoid=False, p=0.8, q=2.0, num_classes=1203, eps=0.01, reduction='mean', loss_weight=1.0, return_dict=True):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict
        self.cls_criterion = seesaw_ce_loss
        self.register_buffer('cum_samples', torch.zeros(self.num_classes + 1, dtype=torch.float))
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

    def _split_cls_score(self, cls_score):
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C + 1).
        """
        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        score_classes = F.softmax(cls_score_classes, dim=-1)
        score_objectness = F.softmax(cls_score_objectness, dim=-1)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = torch.cat([score_classes, score_neg], dim=-1)
        return scores

    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self, cls_score, labels, label_weights=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor | Dict [str, torch.Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()
        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)
        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(cls_score_classes[pos_inds], labels[pos_inds], label_weights[pos_inds], self.cum_samples[:self.num_classes], self.num_classes, self.p, self.q, self.eps, reduction, avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        loss_cls_objectness = self.loss_weight * cross_entropy(cls_score_objectness, obj_labels, label_weights, reduction, avg_factor)
        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_objectness'] = loss_cls_objectness
            loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes + loss_cls_objectness
        return loss_cls


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


class DilatedEncoder(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
    """

    def __init__(self, in_channels, out_channels, block_mid_channels, num_residual_blocks):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]
        self._init_layers()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.lateral_norm = BatchNorm2d(self.out_channels)
        self.fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.fpn_norm = BatchNorm2d(self.out_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(Bottleneck(self.out_channels, self.block_mid_channels, dilation=dilation))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def init_weights(self):
        caffe2_xavier_init(self.lateral_conv)
        caffe2_xavier_init(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            constant_init(m, 1)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    def forward(self, feature):
        out = self.lateral_norm(self.lateral_conv(feature[-1]))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out),


class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self, in_channels, out_channels, stride=1, norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self, in_channels, out_channels, zero_init_offset=True, act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3
        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1), nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                high_feat = F.interpolate(self.spatial_conv_high(x[level + 1], offset, mask), size=x[level].shape[-2:], mode='bilinear', align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) * x_float / norm).type_as(x)


eps = 1e-06


class DropBlock(nn.Module):
    """Randomly drop some regions of feature maps.

     Please refer to the method proposed in `DropBlock
     <https://arxiv.org/abs/1810.12890>`_ for details.

    Args:
        drop_prob (float): The probability of dropping each block.
        block_size (int): The size of dropped blocks.
        warmup_iters (int): The drop probability will linearly increase
            from `0` to `drop_prob` during the first `warmup_iters` iterations.
            Default: 2000.
    """

    def __init__(self, drop_prob, block_size, warmup_iters=2000, **kwargs):
        super(DropBlock, self).__init__()
        assert block_size % 2 == 1
        assert 0 < drop_prob <= 1
        assert warmup_iters >= 0
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.iter_cnt = 0

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map on which some areas will be randomly
                dropped.

        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return x
        self.iter_cnt += 1
        N, C, H, W = list(x.shape)
        gamma = self._compute_gamma((H, W))
        mask_shape = N, C, H - self.block_size + 1, W - self.block_size + 1
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(input=mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        x = x * mask * mask.numel() / (eps + mask.sum())
        return x

    def _compute_gamma(self, feat_size):
        """Compute the value of gamma according to paper. gamma is the
        parameter of bernoulli distribution, which controls the number of
        features to drop.

        gamma = (drop_prob * fm_area) / (drop_area * keep_area)

        Args:
            feat_size (tuple[int, int]): The height and width of feature map.

        Returns:
            float: The value of gamma.
        """
        gamma = self.drop_prob * feat_size[0] * feat_size[1]
        gamma /= (feat_size[0] - self.block_size + 1) * (feat_size[1] - self.block_size + 1)
        gamma /= self.block_size ** 2
        factor = 1.0 if self.iter_cnt > self.warmup_iters else self.iter_cnt / self.warmup_iters
        return gamma * factor

    def extra_repr(self):
        return f'drop_prob={self.drop_prob}, block_size={self.block_size}, warmup_iters={self.warmup_iters}'


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Handle empty batch dimension to AdaptiveAvgPool2d."""

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            output_size = self.output_size
            if isinstance(output_size, int):
                output_size = [output_size, output_size]
            else:
                output_size = [(v if v is not None else d) for v, d in zip(output_size, x.size()[-2:])]
            output_size = [*x.shape[:2], *output_size]
            empty = NewEmptyTensorOp.apply(x, output_size)
            return empty
        return super().forward(x)


class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
    """

    def __init__(self, *args, tempearture=20, power=1.0, eps=1e-06, **kwargs):
        super(NormedLinear, self).__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        weight_ = self.weight / (self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture
        return F.linear(x_, weight_, self.bias)


class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Default to False.
    """

    def __init__(self, *args, tempearture=20, power=1.0, eps=1e-06, norm_over_kernel=False, **kwargs):
        super(NormedConv2d, self).__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps

    def forward(self, x):
        if not self.norm_over_kernel:
            weight_ = self.weight / (self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        else:
            weight_ = self.weight / (self.weight.view(self.weight.size(0), -1).norm(dim=1, keepdim=True).pow(self.power)[..., None, None] + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture
        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        elif torch.__version__ >= '1.8':
            x_ = self._conv_forward(x_, weight_, self.bias)
        else:
            x_ = self._conv_forward(x_, weight_)
        return x_


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class WrapFunction(nn.Module):
    """Wrap the function to be tested for torch.onnx.export tracking."""

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, rescale=False, return_loss=False):
        return imgs

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {'loss': 0.5, 'log_vars': {'accuracy': 0.98}, 'num_samples': 1}
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AconC,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaptiveAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaptiveAvgPool2d,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaptivePadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AnyHead,
     lambda: ([], {'w_in': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'bn_norm': torch.nn.ReLU, 'd': 4, 'block_fun': torch.nn.ReLU, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BNAndPadLayer,
     lambda: ([], {'pad_pixels': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BRLayer,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BaseConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchDrop,
     lambda: ([], {'h_ratio': 4, 'w_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNormXd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (C1,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBAM,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelAttention,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Chuncat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Classify,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClipGlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Contract,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1x1Linear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvSig,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvSqu,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvTranspose,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DBBBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConvTranspose2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeformConv2d,
     lambda: ([], {'inc': 4, 'outc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiverseBranchBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EffHead,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ExampleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Expand,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FRN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FReLU,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FastGlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeatureConcat,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {})),
    (FeatureConcat2,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Foldcut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GAP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeneralizedMeanPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeneralizedMeanPoolingP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Ghost,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostSPPCSPC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostStem,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HarDBlock,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4, 'grmul': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HarDBlock2,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4, 'grmul': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Hardswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IBN,
     lambda: ([], {'planes': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IOUloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (IdentityBasedConv1x1,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Implicit2DA,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {})),
    (Implicit2DC,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {})),
    (Implicit2DM,
     lambda: ([], {'atom': 4, 'channel': 4}),
     lambda: ([], {})),
    (ImplicitA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImplicitC,
     lambda: ([], {'channel': 4}),
     lambda: ([], {})),
    (ImplicitM,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Integral,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 17])], {})),
    (L2Norm,
     lambda: ([], {'n_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LightConv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPoolStride1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MemoryEfficientMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MetaAconC,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MixConv2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp_v2,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PPYOLOE,
     lambda: ([], {'backbone': torch.nn.ReLU(), 'neck': torch.nn.ReLU(), 'head': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Proto,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QFocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RSoftmax,
     lambda: ([], {'radix': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReOrg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reorg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepConv_OREPA,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepRes,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepVGGBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Res,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RobustConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RobustConv2,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPCSPC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (STCSPA,
     lambda: ([], {'c1': 4, 'c2': 64}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (STCSPB,
     lambda: ([], {'c1': 4, 'c2': 32}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (STCSPC,
     lambda: ([], {'c1': 4, 'c2': 64}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Shortcut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ShuffleNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Silence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimSPPF,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SmoothAP_old,
     lambda: ([], {'anneal': 4, 'batch_size': 4, 'num_id': 4, 'feat_dims': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SplAtConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Stem,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemIN,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'bn_norm': torch.nn.ReLU}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sum,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwinTransformer2Block,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwinTransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwinTransformerLayer,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwinTransformerLayer_v2,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TLU,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VoVCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WeightedFeatureFusion,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {})),
    (WrapFunction,
     lambda: ([], {'wrapped_function': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (rSoftMax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

