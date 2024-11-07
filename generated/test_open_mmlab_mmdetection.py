
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


import re


from collections import OrderedDict


from functools import partial


import numpy as np


import pandas as pd


import torch


import math


import time


from typing import Tuple


import torch.nn as nn


import copy


import warnings


from typing import Dict


from typing import Iterable


from typing import List


from typing import Optional


from typing import Sequence


from typing import Union


from torch.nn import BatchNorm2d


from torch.optim.sgd import SGD


from torch.optim import SGD


from torch.optim.adamw import AdamW


from torch.nn.modules.activation import ReLU


from torch.nn.modules.batchnorm import BatchNorm2d


from torch.nn.modules.normalization import GroupNorm


from torch.nn import GroupNorm


from torch.nn.modules.activation import SiLU


from torch.nn import SyncBatchNorm


import itertools


from collections import defaultdict


import torch.multiprocessing as mp


import collections


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


from typing import Iterator


from typing import Sized


import random


import inspect


from numpy import random


from typing import Callable


from torch import nn


from torch.optim import Optimizer


from torch.nn.modules.batchnorm import _BatchNorm


from torch import Tensor


import torch.utils.checkpoint as cp


import torch.nn.functional as F


from torch.nn.modules.utils import _pair as to_2tuple


from copy import deepcopy


from torch.nn.modules.utils import _pair


from numbers import Number


from abc import abstractmethod


from typing import Any


from numpy import ndarray


from abc import ABCMeta


from inspect import signature


from logging import warning


from math import ceil


from math import log


from torch.nn.init import normal_


from torch import nn as nn


from torch.nn import ModuleList


import torch.distributed as dist


import functools


from torch.utils.checkpoint import checkpoint


from scipy.optimize import linear_sum_assignment


from math import sqrt


from torch.autograd import Function


from torch.nn import functional as F


import torch.utils.checkpoint as checkpoint


from abc import abstractproperty


from abc import abstractstaticmethod


from typing import Type


from typing import TypeVar


from torch import BoolTensor


from torch.nn.parallel import DistributedDataParallel


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from collections import abc


from functools import wraps


from tensorflow.python.training import py_checkpoint_reader


from torch.nn.init import _calculate_fan_in_and_fan_out


from torch.nn.init import trunc_normal_


from torch.nn import init


from torch.cuda.amp import autocast


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


import torch.optim as optim


from torch.nn.modules import GroupNorm


from torch.nn.modules import AvgPool2d


from torch.autograd import gradcheck


from scipy.optimize import differential_evolution


from itertools import repeat


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


class BatchFixedSizePad(nn.Module):
    """Fixed size padding for batch images.

    Args:
        size (Tuple[int, int]): Fixed padding size. Expected padding
            shape (h, w). Defaults to None.
        img_pad_value (int): The padded pixel value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
    """

    def __init__(self, size: 'Tuple[int, int]', img_pad_value: 'int'=0, pad_mask: 'bool'=False, mask_pad_value: 'int'=0, pad_seg: 'bool'=False, seg_pad_value: 'int'=255) ->None:
        super().__init__()
        self.size = size
        self.pad_mask = pad_mask
        self.pad_seg = pad_seg
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.seg_pad_value = seg_pad_value

    def forward(self, inputs: 'Tensor', data_samples: 'Optional[List[dict]]'=None) ->Tuple[Tensor, Optional[List[dict]]]:
        """Pad image, instance masks, segmantic segmentation maps."""
        src_h, src_w = inputs.shape[-2:]
        dst_h, dst_w = self.size
        if src_h >= dst_h and src_w >= dst_w:
            return inputs, data_samples
        inputs = F.pad(inputs, pad=(0, max(0, dst_w - src_w), 0, max(0, dst_h - src_h)), mode='constant', value=self.img_pad_value)
        if data_samples is not None:
            for data_sample in data_samples:
                data_sample.set_metainfo({'batch_input_shape': (dst_h, dst_w), 'pad_shape': (dst_h, dst_w)})
            if self.pad_mask:
                for data_sample in data_samples:
                    masks = data_sample.gt_instances.masks
                    data_sample.gt_instances.masks = masks.pad((dst_h, dst_w), pad_val=self.mask_pad_value)
            if self.pad_seg:
                for data_sample in data_samples:
                    gt_sem_seg = data_sample.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    gt_sem_seg = F.pad(gt_sem_seg, pad=(0, max(0, dst_w - w), 0, max(0, dst_h - h)), mode='constant', value=self.seg_pad_value)
                    data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)
        return inputs, data_samples


class Conv3x3Norm(nn.Module):
    """Conv3x3 and norm."""

    def __init__(self, in_channels: 'int', out_channels: 'int', stride: 'int', groups: 'int'=1, use_dcn: 'bool'=False, norm_type: 'Optional[Union[Sequence, str]]'=None):
        super().__init__()
        if use_dcn:
            self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
        if isinstance(norm_type, Sequence):
            assert len(norm_type) == 2
            assert norm_type[0] == 'gn'
            gn_group = norm_type[1]
            norm_type = norm_type[0]
        if norm_type == 'bn':
            bn_op = nn.BatchNorm2d(out_channels)
        elif norm_type == 'gn':
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        if norm_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(nn.Module):
    """Dynamic Convolution."""

    def __init__(self, conv_func: 'Callable', in_channels: 'int', out_channels: 'int', use_dyfuse: 'bool'=True, use_dyrelu: 'bool'=False, use_dcn: 'bool'=False):
        super().__init__()
        self.dyconvs = nn.ModuleList()
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 2))
        if use_dyfuse:
            self.attnconv = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, kernel_size=1), nn.ReLU(inplace=True))
            self.h_sigmoid = nn.Hardsigmoid(inplace=True)
        else:
            self.attnconv = None
        if use_dyrelu:
            self.relu = DyReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()
        if use_dcn:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None
        self.init_weights()

    def init_weights(self):
        for m in self.dyconvs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.attnconv is not None:
            for m in self.attnconv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs: 'dict') ->dict:
        visual_feats = inputs['visual']
        out_vis_feats = []
        for level, feature in enumerate(visual_feats):
            offset_conv_args = {}
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                offset_conv_args = dict(offset=offset, mask=mask)
            temp_feats = [self.dyconvs[1](feature, **offset_conv_args)]
            if level > 0:
                temp_feats.append(self.dyconvs[2](visual_feats[level - 1], **offset_conv_args))
            if level < len(visual_feats) - 1:
                temp_feats.append(F.upsample_bilinear(self.dyconvs[0](visual_feats[level + 1], **offset_conv_args), size=[feature.size(2), feature.size(3)]))
            mean_feats = torch.mean(torch.stack(temp_feats), dim=0, keepdim=False)
            if self.attnconv is not None:
                attn_feat = []
                res_feat = []
                for feat in temp_feats:
                    res_feat.append(feat)
                    attn_feat.append(self.attnconv(feat))
                res_feat = torch.stack(res_feat)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_feat))
                mean_feats = torch.mean(res_feat * spa_pyr_attn, dim=0, keepdim=False)
            out_vis_feats.append(mean_feats)
        out_vis_feats = [self.relu(item) for item in out_vis_feats]
        features_dict = {'visual': out_vis_feats, 'lang': inputs['lang']}
        return features_dict


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
        strides (Sequence[int]): The stride of each input feature map.
            Defaults to (8, 16, 32, 64, 128).
    """

    def __init__(self, force_topk: 'bool'=False, topk: 'int'=9, num_classes: 'int'=80, strides: 'Sequence[int]'=(8, 16, 32, 64, 128)) ->None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
        self.strides = strides
        self.force_topk = force_topk
        self.topk = topk

    def forward(self, anchor_points_list: 'List[Tensor]', gt_instances: 'InstanceData', inside_gt_bbox_mask: 'Tensor') ->Tuple[Tensor, Tensor]:
        """Get the center prior of each point on the feature map for each
        instance.

        Args:
            anchor_points_list (list[Tensor]): list of coordinate
                of points on feature map. Each with shape
                (num_points, 2).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            inside_gt_bbox_mask (Tensor): Tensor of bool type,
                with shape of (num_points, num_gt), each
                value is used to mark whether this point falls
                within a certain gt.

        Returns:
            tuple[Tensor, Tensor]:

            - center_prior_weights(Tensor): Float tensor with shape  of             (num_points, num_gt). Each value represents the center             weighting coefficient.
            - inside_gt_bbox_mask (Tensor): Tensor of bool type, with shape             of (num_points, num_gt), each value is used to mark whether this             point falls within a certain gt or is the topk nearest points for             a specific gt_bbox.
        """
        gt_bboxes = gt_instances.bboxes
        labels = gt_instances.labels
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

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: 'int'=16) ->None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: 'Tensor') ->Tensor:
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


class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self, max_text_len: 'int'=256, log_scale: 'Optional[Union[str, float]]'=None, bias: 'bool'=False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of "auto", "none", None, but got {log_scale}')
        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: 'Tensor', text_feat: 'Tensor', text_token_mask: 'Tensor') ->Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float('-inf'), device=res.device)
        new_res[..., :res.shape[-1]] = res
        return new_res


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
            Defaults to 8.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional):  Config dict for
        normalization layer. Defaults to None.
    """

    def __init__(self, feat_channels: 'int', stacked_convs: 'int', la_down_rate: 'int'=8, conv_cfg: 'OptConfigType'=None, norm_cfg: 'OptConfigType'=None) ->None:
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1), nn.ReLU(inplace=True), nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0), nn.Sigmoid())
        self.reduction_conv = ConvModule(self.in_channels, self.feat_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=norm_cfg is None)

    def init_weights(self) ->None:
        """Initialize the parameters."""
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat: 'Tensor', avg_feat: 'Optional[Tensor]'=None) ->Tensor:
        """Forward function of task decomposition module."""
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


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self, name: 'str', add_pooling_layer: 'bool'=False, num_layers_of_embedded: 'int'=1, use_checkpoint: 'bool'=False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError('transformers is not installed, please install it by: pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        self.model = HFBertModel.from_pretrained(name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) ->dict:
        mask = x['attention_mask']
        outputs = self.model(input_ids=x['input_ids'], attention_mask=mask, position_ids=x['position_ids'], token_type_ids=x['token_type_ids'], output_hidden_states=True)
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:], 1).mean(1)
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features
        results = {'embedded': embedded, 'masks': mask, 'hidden': encoded_layers[-1]}
        return results


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


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are
    fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    Args:
       num_features (int):  :math:`C` from an expected input of size
            :math:`(N, C, H, W)`.
       eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
    """

    def __init__(self, num_features, eps=1e-05, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale + bias
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

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


class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
    """

    def __init__(self, *args, tempearture: float=20, power: int=1.0, eps: float=1e-06, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self) ->None:
        """Initialize the weights."""
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward function for `NormedLinear`."""
        weight_ = self.weight / (self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture
        return F.linear(x_, weight_, self.bias)


class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Defaults to False.
    """

    def __init__(self, *args, tempearture: float=20, power: int=1.0, eps: float=1e-06, norm_over_kernel: bool=False, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward function for `NormedConv2d`."""
        if not self.norm_over_kernel:
            weight_ = self.weight / (self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        else:
            weight_ = self.weight / (self.weight.view(self.weight.size(0), -1).norm(dim=1, keepdim=True).pow(self.power)[..., None, None] + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture
        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        elif digit_version(torch.__version__) >= digit_version('1.8'):
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


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


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


def ae_loss_per_image(tl_preds, br_preds, match):
    """Associative Embedding Loss in one image.

    Associative Embedding Loss including two parts: pull loss and push loss.
    Pull loss makes embedding vectors from same object closer to each other.
    Push loss distinguish embedding vector from different objects, and makes
        the gap between them is large enough.

    During computing, usually there are 3 cases:
        - no object in image: both pull loss and push loss will be 0.
        - one object in image: push loss will be 0 and pull loss is computed
            by the two corner of the only object.
        - more than one objects in image: pull loss is computed by corner pairs
            from each object, push loss is computed by each object with all
            other objects. We use confusion matrix with 0 in diagonal to
            compute the push loss.

    Args:
        tl_preds (tensor): Embedding feature map of left-top corner.
        br_preds (tensor): Embedding feature map of bottim-right corner.
        match (list): Downsampled coordinates pair of each ground truth box.
    """
    tl_list, br_list, me_list = [], [], []
    if len(match) == 0:
        pull_loss = tl_preds.sum() * 0.0
        push_loss = tl_preds.sum() * 0.0
    else:
        for m in match:
            [tl_y, tl_x], [br_y, br_x] = m
            tl_e = tl_preds[:, tl_y, tl_x].view(-1, 1)
            br_e = br_preds[:, br_y, br_x].view(-1, 1)
            tl_list.append(tl_e)
            br_list.append(br_e)
            me_list.append((tl_e + br_e) / 2.0)
        tl_list = torch.cat(tl_list)
        br_list = torch.cat(br_list)
        me_list = torch.cat(me_list)
        assert tl_list.size() == br_list.size()
        N, M = tl_list.size()
        pull_loss = (tl_list - me_list).pow(2) + (br_list - me_list).pow(2)
        pull_loss = pull_loss.sum() / N
        margin = 1
        conf_mat = me_list.expand((N, N, M)).permute(1, 0, 2) - me_list
        conf_weight = 1 - torch.eye(N).type_as(me_list)
        conf_mat = conf_weight * (margin - conf_mat.sum(-1).abs())
        if N > 1:
            push_loss = F.relu(conf_mat).sum() / (N * (N - 1))
        else:
            push_loss = tl_preds.sum() * 0.0
    return pull_loss, push_loss


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


def reduce_loss(loss: 'Tensor', reduction: 'str') ->Tensor:
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


def weight_reduce_loss(loss: 'Tensor', weight: 'Optional[Tensor]'=None, reduction: 'str'='mean', avg_factor: 'Optional[float]'=None) ->Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func: 'Callable') ->Callable:
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
    def wrapper(pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, reduction: 'str'='mean', avg_factor: 'Optional[int]'=None, **kwargs) ->Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction='mean'):
    """Calculate balanced L1 loss.

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, 4).
        target (torch.Tensor): The learning target of the prediction with
            shape (N, 4).
        beta (float): The loss is a piecewise function of prediction and target
            and ``beta`` serves as a threshold for the difference between the
            prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss.
            Defaults to 1.5.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".

    Returns:
        torch.Tensor: The calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - alpha * beta)
    return loss


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


class CrossEntropyCustomLoss(CrossEntropyLoss):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', num_classes=-1, class_weight=None, ignore_index=None, loss_weight=1.0, avg_non_ignore=False):
        """CrossEntropyCustomLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            num_classes (int): Number of classes to classify.
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyCustomLoss, self).__init__()
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
        self.num_classes = num_classes
        assert self.num_classes != -1
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        if not self.use_sigmoid:
            return num_classes + 1
        else:
            return num_classes

    def get_activation(self, cls_score):
        fine_cls_score = cls_score[:, :self.num_classes]
        if not self.use_sigmoid:
            bg_score = cls_score[:, [-1]]
            new_score = torch.cat([fine_cls_score, bg_score], dim=-1)
            scores = F.softmax(new_score, dim=-1)
        else:
            score_classes = fine_cls_score.sigmoid()
            score_neg = 1 - score_classes.sum(dim=1, keepdim=True)
            score_neg = score_neg.clamp(min=0, max=1)
            scores = torch.cat([score_classes, score_neg], dim=1)
        return scores

    def get_accuracy(self, cls_score, labels):
        fine_cls_score = cls_score[:, :self.num_classes]
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc


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


class DDQAuxLoss(nn.Module):
    """DDQ auxiliary branches loss for dense queries.

    Args:
        loss_cls (dict):
            Configuration of classification loss function.
        loss_bbox (dict):
            Configuration of bbox regression loss function.
        train_cfg (dict):
            Configuration of gt targets assigner for each predicted bbox.
    """

    def __init__(self, loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, activated=True, beta=2.0, loss_weight=1.0), loss_bbox=dict(type='GIoULoss', loss_weight=2.0), train_cfg=dict(assigner=dict(type='TopkHungarianAssigner', topk=8), alpha=1, beta=6)):
        super(DDQAuxLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, alignment_metrics):
        """Calculate auxiliary branches loss for dense queries for one image.

        Args:
            cls_score (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_pred (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            labels (Tensor): Labels for one image.
            label_weights (Tensor): Label weights for one image.
            bbox_targets (Tensor): Bbox targets for one image.
            alignment_metrics (Tensor): Normalized alignment metrics for one
                image.

        Returns:
            tuple: A tuple of loss components and loss weights.
        """
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = labels, alignment_metrics
        cls_loss_func = self.loss_cls
        loss_cls = cls_loss_func(cls_score, targets, label_weights, avg_factor=1.0)
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            pos_bbox_weight = alignment_metrics[pos_inds]
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=pos_bbox_weight, avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.0)
        return loss_cls, loss_bbox, alignment_metrics.sum(), pos_bbox_weight.sum()

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, **kwargs):
        """Calculate auxiliary branches loss for dense queries.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (list[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (list[Tensor]): List of ground truth classification
                index for each image, each has shape (num_gt,).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict: A dictionary of loss components.
        """
        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds
        cls_reg_targets = self.get_targets(flatten_cls_scores, flatten_bbox_preds, gt_bboxes, img_metas, gt_labels_list=gt_labels)
        labels_list, label_weights_list, bbox_targets_list, alignment_metrics_list = cls_reg_targets
        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(self.loss_single, flatten_cls_scores, flatten_bbox_preds, labels_list, label_weights_list, bbox_targets_list, alignment_metrics_list)
        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))
        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(aux_loss_cls=losses_cls, aux_loss_bbox=losses_bbox)

    def get_targets(self, cls_scores, bbox_preds, gt_bboxes_list, img_metas, gt_labels_list=None, **kwargs):
        """Compute regression and classification targets for a batch images.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes_list (List[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.
            gt_labels_list (list[Tensor]): List of ground truth classification
                    index for each image, each has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
                    Default: None.

        Returns:
            tuple: a tuple containing the following targets.

            - all_labels (list[Tensor]): Labels for all images.
            - all_label_weights (list[Tensor]): Label weights for all images.
            - all_bbox_targets (list[Tensor]): Bbox targets for all images.
            - all_assign_metrics (list[Tensor]): Normalized alignment metrics
                for all images.
        """
        all_labels, all_label_weights, all_bbox_targets, all_assign_metrics = multi_apply(self._get_target_single, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, img_metas)
        return all_labels, all_label_weights, all_bbox_targets, all_assign_metrics

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_meta, **kwargs):
        """Compute regression and classification targets for one image.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (Tensor): Unnormalized ground truth
                bboxes for one image, has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (Tensor): Ground truth classification
                    index for the image, has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels for one image.
            - label_weights (Tensor): Label weights for one image.
            - bbox_targets (Tensor): Bbox targets for one image.
            - norm_alignment_metrics (Tensor): Normalized alignment
                metrics for one image.
        """
        if len(gt_labels) == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors,), cls_scores.size(-1), dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors, dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors, dtype=torch.float)
            return labels, label_weights, bbox_targets, norm_alignment_metrics
        assign_result = self.assigner.assign(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_meta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics
        pred_instances = BaseDataElement()
        gt_instances = BaseDataElement()
        pred_instances.bboxes = bbox_preds
        gt_instances.bboxes = gt_bboxes
        pred_instances.priors = cls_scores
        gt_instances.labels = gt_labels
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors,), cls_scores.size(-1), dtype=torch.long)
        label_weights = bbox_preds.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        class_assigned_gt_inds = torch.unique(sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (pos_alignment_metrics.max() + 1e-07) * pos_ious.max()
            norm_alignment_metrics[pos_inds[gt_class_inds]] = pos_norm_alignment_metrics
        return labels, label_weights, bbox_targets, norm_alignment_metrics


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


def print_log(msg: 'str', logger: 'Optional[MMLogger]'=None) ->None:
    """Print a log message."""
    if logger is None:
        None
    else:
        logger.info(msg)


class EQLV2Loss(nn.Module):

    def __init__(self, use_sigmoid: 'bool'=True, reduction: 'str'='mean', class_weight: 'Optional[Tensor]'=None, loss_weight: 'float'=1.0, num_classes: 'int'=1203, use_distributed: 'bool'=False, mu: 'float'=0.8, alpha: 'float'=4.0, gamma: 'int'=12, vis_grad: 'bool'=False, test_with_obj: 'bool'=True) ->None:
        """`Equalization Loss v2 <https://arxiv.org/abs/2012.08548>`_

        Args:
            use_sigmoid (bool): EQLv2 uses the sigmoid function to transform
                the predicted logits to an estimated probability distribution.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            class_weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            loss_weight (float, optional): The weight of the total EQLv2 loss.
                Defaults to 1.0.
            num_classes (int): 1203 for lvis v1.0, 1230 for lvis v0.5.
            use_distributed (bool, float): EQLv2 will calculate the gradients
                on all GPUs if there is any. Change to True if you are using
                distributed training. Default to False.
            mu (float, optional): Defaults to 0.8
            alpha (float, optional): A balance factor for the negative part of
                EQLV2 Loss. Defaults to 4.0.
            gamma (int, optional): The gamma for calculating the modulating
                factor. Defaults to 12.
            vis_grad (bool, optional): Default to False.
            test_with_obj (bool, optional): Default to True.

        Returns:
            None.
        """
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.vis_grad = vis_grad
        self.mu = mu
        self.alpha = alpha
        self.gamma = gamma
        self.use_distributed = use_distributed
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        print_log(f'build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}', logger='current', level=logging.DEBUG)

    def forward(self, cls_score: 'Tensor', label: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[Tensor]'=None) ->Tensor:
        """`Equalization Loss v2 <https://arxiv.org/abs/2012.08548>`_

        Args:
            cls_score (Tensor): The prediction with shape (N, C), C is the
                number of classes.
            label (Tensor): The ground truth label of the predicted target with
                shape (N, C), C is the number of classes.
            weight (Tensor, optional): The weight of loss for each prediction.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
           Tensor: The calculated loss
        """
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target
        target = expand_label(cls_score, label)
        pos_w, neg_w = self.get_weight(cls_score)
        weight = pos_w * target + neg_w * (1 - target)
        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i
        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, pred):
        pred = torch.sigmoid(pred)
        n_i, n_c = pred.size()
        bg_score = pred[:, -1].view(n_i, 1)
        if self.test_with_obj:
            pred[:, :-1] *= 1 - bg_score
        return pred

    def collect_grad(self, pred, target, weight):
        prob = torch.sigmoid(pred)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]
        if self.use_distributed:
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, pred):
        neg_w = torch.cat([self.map_func(self.pos_neg), pred.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


def py_focal_loss_with_prob(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
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
    if pred.dim() != target.dim():
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
    """A wrapper of cuda version `Focal Loss
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
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
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
            elif pred.dim() == target.dim():
                calculate_loss_func = py_sigmoid_focal_loss
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


class FocalCustomLoss(nn.Module):

    def __init__(self, use_sigmoid=True, num_classes=-1, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False):
        """`Focal Loss for V3Det <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            num_classes (int): Number of classes to classify.
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
        super(FocalCustomLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        assert self.num_classes != -1
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        return num_classes

    def get_activation(self, cls_score):
        fine_cls_score = cls_score[:, :self.num_classes]
        score_classes = fine_cls_score.sigmoid()
        return score_classes

    def get_accuracy(self, cls_score, labels):
        fine_cls_score = cls_score[:, :self.num_classes]
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc

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
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss
            loss_cls = self.loss_weight * calculate_loss_func(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@weighted_loss
def gaussian_focal_loss(pred: 'Tensor', gaussian_target: 'Tensor', alpha: 'float'=2.0, gamma: 'float'=4.0, pos_weight: 'float'=1.0, neg_weight: 'float'=1.0) ->Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_weight * pos_loss + neg_weight * neg_loss


def gaussian_focal_loss_with_pos_inds(pred: 'Tensor', gaussian_target: 'Tensor', pos_inds: 'Tensor', pos_labels: 'Tensor', alpha: 'float'=2.0, gamma: 'float'=4.0, pos_weight: 'float'=1.0, neg_weight: 'float'=1.0, reduction: 'str'='mean', avg_factor: 'Optional[Union[int, float]]'=None) ->Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Note: The index with a value of 1 in ``gaussian_target`` in the
    ``gaussian_focal_loss`` function is a positive sample, but in
    ``gaussian_focal_loss_with_pos_inds`` the positive sample is passed
    in through the ``pos_inds`` parameter.

    Args:
        pred (torch.Tensor): The prediction. The shape is (N, num_classes).
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution. The shape is (N, num_classes).
        pos_inds (torch.Tensor): The positive sample index.
            The shape is (M, ).
        pos_labels (torch.Tensor): The label corresponding to the positive
            sample index. The shape is (M, ).
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to 'mean`.
        avg_factor (int, float, optional): Average factor that is used to
            average the loss. Defaults to None.
    """
    eps = 1e-12
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_pred_pix = pred[pos_inds]
    pos_pred = pos_pred_pix.gather(1, pos_labels.unsqueeze(1))
    pos_loss = -(pos_pred + eps).log() * (1 - pos_pred).pow(alpha)
    pos_loss = weight_reduce_loss(pos_loss, None, reduction, avg_factor)
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    neg_loss = weight_reduce_loss(neg_loss, None, reduction, avg_factor)
    return pos_weight * pos_loss + neg_weight * neg_loss


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
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self, alpha: 'float'=2.0, gamma: 'float'=4.0, reduction: 'str'='mean', loss_weight: 'float'=1.0, pos_weight: 'float'=1.0, neg_weight: 'float'=1.0) ->None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', pos_inds: 'Optional[Tensor]'=None, pos_labels: 'Optional[Tensor]'=None, weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[Union[int, float]]'=None, reduction_override: 'Optional[str]'=None) ->Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if pos_inds is not None:
            assert pos_labels is not None
            loss_reg = self.loss_weight * gaussian_focal_loss_with_pos_inds(pred, target, pos_inds, pos_labels, alpha=self.alpha, gamma=self.gamma, pos_weight=self.pos_weight, neg_weight=self.neg_weight, reduction=reduction, avg_factor=avg_factor)
        else:
            loss_reg = self.loss_weight * gaussian_focal_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, pos_weight=self.pos_weight, neg_weight=self.neg_weight, reduction=reduction, avg_factor=avg_factor)
        return loss_reg


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    """Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

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
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def quality_focal_loss_tensor_target(pred, target, beta=2.0, activated=False):
    """`QualityFocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        activated (bool): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """
    assert pred.size() == target.size()
    if activated:
        pred_sigmoid = pred
        loss_function = F.binary_cross_entropy
    else:
        pred_sigmoid = pred.sigmoid()
        loss_function = F.binary_cross_entropy_with_logits
    scale_factor = pred_sigmoid
    target = target.type_as(pred)
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = loss_function(pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    pos = target != 0
    scale_factor = target[pos] - pred_sigmoid[pos]
    loss[pos] = loss_function(pred[pos], target[pos], reduction='none') * scale_factor.abs().pow(beta)
    loss = loss.sum(dim=1, keepdim=False)
    return loss


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
            target (Union(tuple([torch.Tensor]),Torch.Tensor)): The type is
                tuple, it should be included Target category label with
                shape (N,) and target quality label with shape (N,).The type
                is torch.Tensor, the target should be one-hot form with
                soft weights.
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
            if isinstance(target, torch.Tensor):
                calculate_loss_func = partial(quality_focal_loss_tensor_target, activated=self.activated)
            loss_cls = self.loss_weight * calculate_loss_func(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@weighted_loss
def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


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


@weighted_loss
def iou_loss(pred: 'Tensor', target: 'Tensor', linear: 'bool'=False, mode: 'str'='log', eps: 'float'=1e-06) ->Tensor:
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in iou_loss is deprecated, please use "mode=`linear`" instead.')
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred
    else:
        fp16 = False
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if fp16:
        ious = ious
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious ** 2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, linear: 'bool'=False, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0, mode: 'str'='log') ->None:
        super().__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in IOULoss is deprecated, please use "mode=`linear`" instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Return:
            Tensor: Loss tensor.
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


@weighted_loss
def bounded_iou_loss(pred: 'Tensor', target: 'Tensor', beta: 'float'=0.2, eps: 'float'=0.001) ->Tensor:
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.

    Return:
        Tensor: Loss tensor.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry
    loss_dx = 1 - torch.max((target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).flatten(1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss


class BoundedIoULoss(nn.Module):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        beta (float, optional): Beta parameter in smoothl1.
        eps (float, optional): Epsilon to avoid NaN values.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, beta: 'float'=0.2, eps: 'float'=0.001, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(pred, target, weight, beta=self.beta, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


def giou_loss(boxes1: 'torch.Tensor', boxes2: 'torch.Tensor', reduction: 'str'='none', eps: 'float'=1e-07) ->torch.Tensor:
    """Generalized Intersection over Union Loss (Hamid Rezatofighi et.

    al)
    https://arxiv.org/abs/1902.09630
    Gradient-friendly IoU loss with an additional penalty that is
    non-zero when the boxes do not overlap and scales with the size
    of their smallest enclosing box. This loss is symmetric, so the
    boxes1 and boxes2 arguments are interchangeable.
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape
        (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    assert (x2 >= x1).all(), 'bad box: x1 larger than x2'
    assert (y2 >= y1).all(), 'bad box: y1 larger than y2'
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - (area_c - unionk) / (area_c + eps)
    loss = 1 - miouk
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class GIoULoss(nn.Module):
    """`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
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


@weighted_loss
def diou_loss(pred: 'Tensor', target: 'Tensor', eps: 'float'=1e-07) ->Tensor:
    """Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw ** 2 + ch ** 2 + eps
    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]
    left = (b2_x1 + b2_x2 - (b1_x1 + b1_x2)) ** 2 / 4
    right = (b2_y1 + b2_y2 - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


class DIoULoss(nn.Module):
    """Implementation of `Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
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


@weighted_loss
def ciou_loss(pred: 'Tensor', target: 'Tensor', eps: 'float'=1e-07) ->Tensor:
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw ** 2 + ch ** 2 + eps
    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    left = (b2_x1 + b2_x2 - (b1_x1 + b1_x2)) ** 2 / 4
    right = (b2_y1 + b2_y2 - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right
    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss


class CIoULoss(nn.Module):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
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


@weighted_loss
def eiou_loss(pred: 'Tensor', target: 'Tensor', smooth_point: 'float'=0.1, eps: 'float'=1e-07) ->Tensor:
    """Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1.
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (iy1 - ey1)
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - intersection + eps
    ious = 1 - intersection / union
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * ious ** 2 / smooth_point + (1 - smooth_sign) * (ious - 0.5 * smooth_point)
    return loss


class EIoULoss(nn.Module):
    """Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0, smooth_point: 'float'=0.1) ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * eiou_loss(pred, target, weight, smooth_point=self.smooth_point, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def siou_loss(pred, target, eps=1e-07, neg_gamma=False):
    """`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
    sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss


class SIoULoss(nn.Module):
    """`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0, neg_gamma: 'bool'=False) ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_gamma = neg_gamma

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * siou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, neg_gamma=self.neg_gamma, **kwargs)
        return loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred: 'Tensor', soft_label: 'Tensor', T: 'int', detach_target: 'bool'=True) ->Tensor:
    """Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()
    kd_loss = F.kl_div(F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (T * T)
    return kd_loss


class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction: 'str'='mean', loss_weight: 'float'=1.0, T: 'int'=10) ->None:
        super().__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self, pred: 'Tensor', soft_label: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor, T=self.T)
        return loss_kd


@weighted_loss
def mse_loss(pred: 'Tensor', target: 'Tensor') ->Tensor:
    """A Wrapper of MSE loss.
    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: loss Tensor
    """
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None) ->Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


def seesaw_ce_loss(cls_score: 'Tensor', labels: 'Tensor', label_weights: 'Tensor', cum_samples: 'Tensor', num_classes: 'int', p: 'float', q: 'float', eps: 'float', reduction: 'str'='mean', avg_factor: 'Optional[int]'=None) ->Tensor:
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (Tensor): The learning label of the prediction.
        label_weights (Tensor): Sample-wise loss weight.
        cum_samples (Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: The calculated loss
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

    def __init__(self, use_sigmoid: 'bool'=False, p: 'float'=0.8, q: 'float'=2.0, num_classes: 'int'=1203, eps: 'float'=0.01, reduction: 'str'='mean', loss_weight: 'float'=1.0, return_dict: 'bool'=True) ->None:
        super().__init__()
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

    def _split_cls_score(self, cls_score: 'Tensor') ->Tuple[Tensor, Tensor]:
        """split cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).

        Returns:
            Tuple[Tensor, Tensor]: The score for classes and objectness,
                 respectively
        """
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes: 'int') ->int:
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score: 'Tensor') ->Tensor:
        """Get custom activation of cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).

        Returns:
            Tensor: The custom activation of cls_score with shape
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

    def get_accuracy(self, cls_score: 'Tensor', labels: 'Tensor') ->Dict[str, Tensor]:
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).
            labels (Tensor): The learning label of the prediction.

        Returns:
            Dict [str, Tensor]: The accuracy for objectness and classes,
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

    def forward(self, cls_score: 'Tensor', labels: 'Tensor', label_weights: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None) ->Union[Tensor, Dict[str, Tensor]]:
        """Forward function.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).
            labels (Tensor): The learning label of the prediction.
            label_weights (Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".

        Returns:
            Tensor | Dict [str, Tensor]:
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


@weighted_loss
def smooth_l1_loss(pred: 'Tensor', target: 'Tensor', beta: 'float'=1.0) ->Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta: 'float'=1.0, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None, **kwargs) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


@weighted_loss
def l1_loss(pred: 'Tensor', target: 'Tensor') ->Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


def varifocal_loss(pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, alpha: 'float'=0.75, gamma: 'float'=2.0, iou_weighted: 'bool'=True, reduction: 'str'='mean', avg_factor: 'Optional[int]'=None) ->Tensor:
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes.
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: Loss tensor.
    """
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class VarifocalLoss(nn.Module):

    def __init__(self, use_sigmoid: 'bool'=True, alpha: 'float'=0.75, gamma: 'float'=2.0, iou_weighted: 'bool'=True, reduction: 'str'='mean', loss_weight: 'float'=1.0) ->None:
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
        super().__init__()
        assert use_sigmoid is True, 'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', weight: 'Optional[Tensor]'=None, avg_factor: 'Optional[int]'=None, reduction_override: 'Optional[str]'=None) ->Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction with shape (N, C), C is the
                number of classes.
            target (Tensor): The learning target of the iou-aware
                classification score with shape (N, C), C is
                the number of classes.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * varifocal_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, iou_weighted=self.iou_weighted, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


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
        block_dilations (list): The list of residual blocks dilation.
    """

    def __init__(self, in_channels, out_channels, block_mid_channels, num_residual_blocks, block_dilations):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
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


MAX_CLAMP_VALUE = 50000


class BiMultiHeadAttention(nn.Module):
    """Bidirectional fusion Multi-Head Attention layer.

    Args:
        v_dim (int): The dimension of the vision input.
        l_dim (int): The dimension of the language input.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self, v_dim: 'int', l_dim: 'int', embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.1):
        super(BiMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim
        assert self.head_dim * self.num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)
        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True
        self._reset_parameters()

    def _shape(self, tensor: 'Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, vision: 'Tensor', lang: 'Tensor', attention_mask_v: 'Optional[Tensor]'=None, attention_mask_l: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        bsz, tgt_len, _ = vision.size()
        query_states = self.v_proj(vision) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(vision), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=MAX_CLAMP_VALUE)
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=MAX_CLAMP_VALUE)
        if attention_mask_v is not None:
            attention_mask_v = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights_l.masked_fill_(attention_mask_v, float('-inf'))
        attn_weights_l = attn_weights_l.softmax(dim=-1)
        if attention_mask_l is not None:
            assert attention_mask_l.dim() == 2
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9000000000000000.0)
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)
        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)
        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output_v` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output_v.size()}')
        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(f'`attn_output_l` should be of size {bsz, self.num_heads, src_len, self.head_dim}, but is {attn_output_l.size()}')
        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)
        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)
        return attn_output_v, attn_output_l


def permute_and_flatten(layer: 'Tensor', N: 'int', A: 'int', C: 'int', H: 'int', W: 'int') ->Tensor:
    """Permute and then flatten a tensor,

       from size (N, A, C, H, W) to (N, H * W * A, C).

    Args:
        layer (Tensor): Tensor of shape (N, C, H, W).
        N (int): Batch size.
        A (int): Number of attention heads.
        C (int): Number of channels.
        H (int): Height of feature map.
        W (int): Width of feature map.

    Returns:
        Tensor: A Tensor of shape (N, H * W * A, C).
    """
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class BiAttentionBlock(nn.Module):
    """BiAttentionBlock Module:

    First, multi-level visual features are concat; Then the concat visual
    feature and lang feature are fused by attention; Finally the newly visual
    feature are split into multi levels.

    Args:
        v_dim (int): The dimension of the visual features.
        l_dim (int): The dimension of the language feature.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        drop_path (float, optional): The drop path probability.
            Defaults to 0.0.
        init_values (float, optional):
            The initial value for the scaling parameter.
            Defaults to 1e-4.
    """

    def __init__(self, v_dim: 'int', l_dim: 'int', embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.1, drop_path: 'float'=0.0, init_values: 'float'=0.0001):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones(v_dim), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones(l_dim), requires_grad=True)

    def forward(self, vf0: 'Tensor', vf1: 'Tensor', vf2: 'Tensor', vf3: 'Tensor', vf4: 'Tensor', lang_feature: 'Tensor', attention_mask_l=None):
        visual_features = [vf0, vf1, vf2, vf3, vf4]
        size_per_level, visual_features_flatten = [], []
        for i, feat_per_level in enumerate(visual_features):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, -1, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = torch.cat(visual_features_flatten, dim=1)
        new_v, new_lang_feature = self.single_attention_call(visual_features_flatten, lang_feature, attention_mask_l=attention_mask_l)
        new_v = new_v.transpose(1, 2).contiguous()
        start = 0
        fvfs = []
        for h, w in size_per_level:
            new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
            fvfs.append(new_v_per_level)
            start += h * w
        return fvfs[0], fvfs[1], fvfs[2], fvfs[3], fvfs[4], new_lang_feature

    def single_attention_call(self, visual: 'Tensor', lang: 'Tensor', attention_mask_v: 'Optional[Tensor]'=None, attention_mask_l: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        """Perform a single attention call between the visual and language
        inputs.

        Args:
        visual (Tensor): The visual input tensor.
        lang (Tensor): The language input tensor.
        attention_mask_v (Optional[Tensor]):
            An optional attention mask tensor for the visual input.
        attention_mask_l (Optional[Tensor]):
            An optional attention mask tensor for the language input.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the updated
                visual and language tensors after the attention call.
        """
        visual = self.layer_norm_v(visual)
        lang = self.layer_norm_l(lang)
        delta_v, delta_l = self.attn(visual, lang, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        visual = visual + self.drop_path(self.gamma_v * delta_v)
        lang = lang + self.drop_path(self.gamma_l * delta_l)
        return visual, lang


class SingleScaleBiAttentionBlock(BiAttentionBlock):
    """This is a single-scale implementation of `BiAttentionBlock`.

    The only differenece between it and `BiAttentionBlock` is that the
    `forward` function of `SingleScaleBiAttentionBlock` only accepts a single
    flatten visual feature map, while the `forward` function in
    `BiAttentionBlock` accepts multiple visual feature maps.
    """

    def forward(self, visual_feature: 'Tensor', lang_feature: 'Tensor', attention_mask_v=None, attention_mask_l=None):
        """Single-scale forward pass.

        Args:
            visual_feature (Tensor): The visual input tensor. Tensor of
                shape (bs, patch_len, ch).
            lang_feature (Tensor): The language input tensor. Tensor of
                shape (bs, text_len, ch).
            attention_mask_v (_type_, optional): Visual feature attention
                mask. Defaults to None.
            attention_mask_l (_type_, optional): Language feature attention
                mask.Defaults to None.
        """
        new_v, new_lang_feature = self.single_attention_call(visual_feature, lang_feature, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        return new_v, new_lang_feature


class VLFuse(nn.Module):
    """Early Fusion Module.

    Args:
        v_dim (int): Dimension of visual features.
        l_dim (int): Dimension of language features.
        embed_dim (int): The embedding dimension for the attention operation.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        drop_path (float): Drop path probability.
        use_checkpoint (bool): Whether to use PyTorch's checkpoint function.
    """

    def __init__(self, v_dim: 'int'=256, l_dim: 'int'=768, embed_dim: 'int'=2048, num_heads: 'int'=8, dropout: 'float'=0.1, drop_path: 'float'=0.0, use_checkpoint: 'bool'=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.b_attn = BiAttentionBlock(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, drop_path=drop_path, init_values=1.0 / 6.0)

    def forward(self, x: 'dict') ->dict:
        """Forward pass of the VLFuse module."""
        visual_features = x['visual']
        language_dict_features = x['lang']
        if self.use_checkpoint:
            vf0, vf1, vf2, vf3, vf4, language_features = checkpoint.checkpoint(self.b_attn, *visual_features, language_dict_features['hidden'], language_dict_features['masks'])
        else:
            vf0, vf1, vf2, vf3, vf4, language_features = self.b_attn(*visual_features, language_dict_features['hidden'], language_dict_features['masks'])
        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features
        features_dict = {'visual': [vf0, vf1, vf2, vf3, vf4], 'lang': fused_language_dict_features}
        return features_dict


class BertSelfAttention(nn.Module):
    """BERT self-attention layer from Huggingface transformers.

    Compared to the BertSelfAttention of Huggingface, only add the clamp.

    Args:
        config (:class:`~transformers.BertConfig`):
            The configuration object that
            contains various parameters for the model.
        clamp_min_for_underflow (bool, optional):
            Whether to clamp the minimum value of the hidden states
             to prevent underflow. Defaults to `False`.
        clamp_max_for_overflow (bool, optional):
            Whether to clamp the maximum value of the hidden states
            to prevent overflow. Defaults to `False`.
    """

    def __init__(self, config: 'BertConfig', clamp_min_for_underflow: 'bool'=False, clamp_max_for_overflow: 'bool'=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: 'Tensor') ->Tensor:
        """Transpose the dimensions of `x`."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None, encoder_attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, output_attentions: 'bool'=False) ->Tuple[Tensor, ...]:
        """Perform a forward pass through the BERT self-attention layer."""
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if self.clamp_min_for_underflow:
            attention_scores = torch.clamp(attention_scores, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            attention_scores = torch.clamp(attention_scores, max=MAX_CLAMP_VALUE)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        pretrained_model, _ = clip.load(model_name, device='cpu')
        self.clip = pretrained_model

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def tokenize(self, texts: 'Union[str, List[str]]', context_length: 'int'=77) ->torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [([sot_token] + self.tokenizer.encode(text) + [eot_token]) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                st = torch.randint(len(tokens) - context_length + 1, (1,))[0].item()
                tokens = tokens[st:st + context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def forward(self, text):
        text = self.tokenize(text)
        text_features = self.clip.encode_text(text)
        return text_features


class ZeroShotClassifier(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', zs_weight_path: 'str', zs_weight_dim: 'int'=512, use_bias: 'float'=0.0, norm_weight: 'bool'=True, norm_temperature: 'float'=50.0):
        super().__init__()
        num_classes = out_features
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)
        self.linear = nn.Linear(in_features, zs_weight_dim)
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(np.load(zs_weight_path), dtype=torch.float32).permute(1, 0).contiguous()
        zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], dim=1)
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    def forward(self, x, classifier=None):
        """
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        """
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()
            zs_weight = F.normalize(zs_weight, p=2, dim=0) if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x


def heatmap_focal_loss_with_pos_inds(pred: 'Tensor', targets: 'Tensor', pos_inds: 'Tensor', alpha: 'float'=2.0, beta: 'float'=4.0, gamma: 'float'=4.0, sigmoid_clamp: 'float'=0.0001, ignore_high_fp: 'float'=-1.0, pos_weight: 'float'=1.0, neg_weight: 'float'=1.0, avg_factor: 'Optional[Union[int, float]]'=None) ->Tensor:
    pred = torch.clamp(pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred = pred[pos_inds]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss
    pos_loss = -pos_loss.sum()
    neg_loss = -neg_loss.sum()
    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss
    pos_loss = pos_weight * pos_loss / avg_factor
    neg_loss = neg_weight * neg_loss / avg_factor
    return pos_loss, neg_loss


class HeatmapFocalLoss(nn.Module):
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
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self, alpha: 'float'=2.0, beta: 'float'=4.0, gamma: 'float'=4.0, sigmoid_clamp: 'float'=0.0001, ignore_high_fp: 'float'=-1.0, loss_weight: 'float'=1.0, pos_weight: 'float'=1.0, neg_weight: 'float'=1.0) ->None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred: 'Tensor', target: 'Tensor', pos_inds: 'Optional[Tensor]'=None, avg_factor: 'Optional[Union[int, float]]'=None) ->Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        pos_loss, neg_loss = heatmap_focal_loss_with_pos_inds(pred, target, pos_inds, alpha=self.alpha, beta=self.beta, gamma=self.gamma, sigmoid_clamp=self.sigmoid_clamp, ignore_high_fp=self.ignore_high_fp, pos_weight=self.pos_weight, neg_weight=self.neg_weight, avg_factor=avg_factor)
        return pos_loss, neg_loss


class IOULoss(nn.Module):

    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None, reduction='sum'):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_aera = (target_left + target_right) * (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None:
            losses = losses * weight
        else:
            losses = losses
        if reduction == 'sum':
            return losses.sum()
        elif reduction == 'batch':
            return losses.sum(dim=[1])
        elif reduction == 'none':
            return losses
        else:
            raise NotImplementedError


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def bbox_cxcywh_to_xyxy(bbox: 'Tensor') ->Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: 'Tensor') ->Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return torch.cat(bbox_new, dim=-1)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


T = TypeVar('T')

