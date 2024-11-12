
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


import torch.utils.data as data


import numpy as np


import itertools


from collections import OrderedDict


import random


import math


from torch.autograd import Function


from math import sqrt as sqrt


from itertools import product as product


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import torch.backends.cudnn as cudnn


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import logging


import collections


import copy


from typing import Any


from collections import defaultdict


from typing import List


from typing import Tuple


from typing import Optional


from torch.utils.data.sampler import Sampler


from typing import Dict


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, obj_t, idx, overlap=None):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    if overlap is not None:
        overlap[idx] = best_truth_overlap
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold, 0] = 0
    conf[best_truth_overlap < threshold, 1] = 1
    loc = encode(matches, priors, variances)
    obj = conf[:, 0] != 0
    loc_t[idx] = loc
    conf_t[idx] = conf
    obj_t[idx] = obj


class MultiBoxLoss_combined(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss_combined, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, obj_data = predictions
        device = loc_data.device
        targets = [anno for anno in targets]
        num = loc_data.size(0)
        num_priors = priors.size(0)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.Tensor(num, num_priors, 2)
        obj_t = torch.BoolTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-2].data
            labels = targets[idx][:, -2:].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, obj_t, idx)
        pos = (conf_t[:, :, 0] > 0).bool()
        num_pos = (conf_t[:, :, 1] * pos.float()).sum(1, keepdim=True).long()
        loc_p = loc_data[pos]
        loc_t = loc_t[pos]
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='none')
        weight_pos = conf_t[pos][:, 1]
        loss_l = torch.sum(torch.sum(loss_l, dim=1) * weight_pos)
        with torch.no_grad():
            loss_obj = F.cross_entropy(obj_data.view(-1, 2), obj_t.long().view(-1), reduction='none')
            loss_obj[obj_t.view(-1)] = 0
            loss_obj = loss_obj.view(num, -1)
            _, loss_idx = loss_obj.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
        mask = pos | neg
        weight = conf_t[mask][:, 1]
        loss_obj = torch.sum(F.cross_entropy(obj_data[mask], obj_t[mask].long(), reduction='none') * weight)
        batch_conf = conf_data.view(-1, self.num_classes - 1)
        batch_obj = obj_data.view(-1, 2)
        logit_0 = batch_obj[:, 0].unsqueeze(1) + torch.log(torch.exp(batch_conf).sum(dim=1, keepdim=True))
        logit_k = batch_obj[:, 1].unsqueeze(1).expand_as(batch_conf) + batch_conf
        logit = torch.cat((logit_0, logit_k), 1)
        logit = logit.view(num, -1, self.num_classes)
        loss_c = torch.sum(F.cross_entropy(logit[mask], conf_t[mask][:, 0].long(), reduction='none') * weight)
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        loss_obj /= N
        return {'loss_box_reg': loss_l, 'loss_cls': loss_c, 'loss_obj': loss_obj}


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False))
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1), BasicConv(inter_planes // 2, inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, args, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.method = args.method
        self.phase = args.phase
        self.setting = args.setting
        self.num_classes = num_classes
        self.size = size
        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            None
            return
        self.base = nn.ModuleList(base)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.obj = nn.ModuleList(head[2])
        self.init_weight()
        if args.method == 'ours' and args.phase == 2:
            if args.setting == 'transfer':
                self.theta = nn.Linear(60, 60)
                self.phi = nn.Linear(60, 60)
                self.g = nn.Linear(60, 60)
                self.Wz = nn.Parameter(torch.FloatTensor(60))
                self.OBJ_Target = nn.Linear(60, 20, bias=False)
                self.scale = nn.Parameter(torch.FloatTensor([5]), requires_grad=False)
                init.kaiming_normal_(self.theta.weight, mode='fan_out')
                init.kaiming_normal_(self.phi.weight, mode='fan_out')
                init.kaiming_normal_(self.g.weight, mode='fan_out')
                self.theta.bias.data.fill_(0)
                self.phi.bias.data.fill_(0)
                self.g.bias.data.fill_(0)
                self.Wz.data.fill_(0)
            elif args.setting == 'incre':
                self.fc_base = nn.Linear(15, 15)
                self.theta = nn.Linear(15, 15)
                self.phi = nn.Linear(15, 15)
                self.g = nn.Linear(15, 15)
                self.Wz = nn.Parameter(torch.FloatTensor(15))
                self.OBJ_Target = nn.Linear(15, 5, bias=False)
                self.scale = nn.Parameter(torch.FloatTensor([5]), requires_grad=False)
                self.fc_base.weight.data.fill_(0)
                init.kaiming_normal_(self.theta.weight, mode='fan_out')
                init.kaiming_normal_(self.phi.weight, mode='fan_out')
                init.kaiming_normal_(self.g.weight, mode='fan_out')
                self.fc_base.bias.data.fill_(0)
                self.theta.bias.data.fill_(0)
                self.phi.bias.data.fill_(0)
                self.g.bias.data.fill_(0)
                self.Wz.data.fill_(0)

    def forward(self, x, init=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x = x
        num = x.size(0)
        sources = list()
        loc = list()
        conf = list()
        obj = list()
        conf_pool = list()
        for k in range(23):
            x = self.base[k](x)
        s = self.Norm(x)
        sources.append(s)
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)
        kernel_size = [3, 2, 2, 2, 1, 1]
        stride = [3, 2, 2, 2, 1, 1]
        for i, (x, l, c, o) in enumerate(zip(sources, self.loc, self.conf, self.obj)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            obj.append(o(x).permute(0, 2, 3, 1).contiguous())
            if self.method == 'ours' and self.phase == 2:
                conf_pool.append(nn.functional.max_pool2d(c(x), kernel_size=kernel_size[i], stride=stride[i], ceil_mode=True).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        obj = torch.cat([o.view(o.size(0), -1) for o in obj], 1)
        if init:
            return conf.view(num, -1, self.num_classes)
        if self.method == 'ours' and self.phase == 2:
            conf_pool = torch.cat([o.view(o.size(0), -1) for o in conf_pool], 1)
            conf = conf.view(num, -1, self.num_classes)
            conf_pool = conf_pool.view(num, -1, self.num_classes)
            if self.setting == 'incre':
                conf_base = self.fc_base(conf) + conf
            conf_theta = self.theta(conf) + conf
            conf_phi = self.phi(conf_pool) + conf_pool
            conf_g = self.g(conf_pool) + conf_pool
            weight = torch.matmul(conf_theta, conf_phi.transpose(1, 2))
            weight = nn.functional.softmax(weight, dim=2)
            delta_conf = torch.matmul(weight, conf_g) * self.Wz
            conf_novel = conf + delta_conf
            conf_novel = conf_novel / conf_novel.norm(dim=2, keepdim=True)
            conf_novel = self.OBJ_Target(conf_novel) * self.scale
            if self.setting == 'transfer':
                conf = conf_novel
            elif self.setting == 'incre':
                conf = torch.cat((conf_base, conf_novel), dim=2)
        if self.training:
            output = loc.view(num, -1, 4), conf if self.phase == 2 and self.method == 'ours' else conf.view(num, -1, self.num_classes), obj.view(num, -1, 2)
        else:
            output = loc.view(num, -1, 4), nn.functional.softmax(conf if self.phase == 2 and self.method == 'ours' else conf.view(num, -1, self.num_classes), dim=-1), nn.functional.softmax(obj.view(num, -1, 2), dim=-1)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file))
            None
        else:
            None

    def init_weight(self):

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        self.base.apply(weights_init)
        self.Norm.apply(weights_init)
        self.extras.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)
        self.obj.apply(weights_init)

    def normalize(self):
        self.OBJ_Target.weight.data = self.OBJ_Target.weight / self.OBJ_Target.weight.norm(dim=1, keepdim=True)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

