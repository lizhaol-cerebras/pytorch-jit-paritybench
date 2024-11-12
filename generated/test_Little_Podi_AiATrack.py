
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


import torch.autograd as ag


import torch.nn as nn


import torch


import torch.nn.functional as F


from torch import nn


import math


from typing import List


from typing import Optional


from typing import Tuple


from torch import Tensor


from torch.nn.functional import *


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.modules.linear import Linear


from torch.nn.modules.module import Module


from torch.nn.parameter import Parameter


from typing import Dict


from torchvision.models._utils import IntermediateLayerGetter


from torch.utils.model_zoo import load_url as load_state_dict_from_url


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import conv1x1


import copy


import numpy as np


import matplotlib


import matplotlib.pyplot as plt


from collections import OrderedDict


from itertools import product


from torch.utils.data.distributed import DistributedSampler


import collections


import torch.utils.data.dataloader


import torchvision.transforms as transforms


import random


import torch.utils.data


import torchvision.transforms.functional as tvisf


import pandas


import warnings


import torch.backends.cudnn


import torch.distributed as dist


from torch.nn.functional import l1_loss


from torch.nn.parallel import DistributedDataParallel as DDP


import time


from torchvision.ops.boxes import box_area


from collections import defaultdict


from collections import deque


import torchvision


import functools


def _import_prroi_pooling():
    global _prroi_pooling
    if _prroi_pooling is None:
        try:
            from torch.utils.cpp_extension import load as load_extension
            root_dir = pjoin(dirname(__file__), 'src')
            _prroi_pooling = load_extension('_prroi_pooling', [pjoin(root_dir, 'prroi_pooling_gpu.c'), pjoin(root_dir, 'prroi_pooling_gpu_impl.cu')], verbose=True)
        except ImportError:
            raise ImportError('Can not compile Precise RoI Pooling library.')
    return _prroi_pooling


class PrRoIPool2DFunction(ag.Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        _prroi_pooling = _import_prroi_pooling()
        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), 'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())
        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)
        features = features.contiguous()
        rois = rois.contiguous()
        params = pooled_height, pooled_width, spatial_scale
        if features.is_cuda:
            output = _prroi_pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _prroi_pooling = _import_prroi_pooling()
        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None
        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = _prroi_pooling.prroi_pooling_backward_cuda(features, rois, output, grad_output, *ctx.params)
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = _prroi_pooling.prroi_pooling_coor_backward_cuda(features, rois, output, grad_output, *ctx.params)
        return grad_input, grad_coor, None, None, None


prroi_pool2d = PrRoIPool2DFunction.apply


class PrRoIPool2D(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)

    def extra_repr(self):
        return 'kernel_size=({pooled_height}, {pooled_width}), spatial_scale={spatial_scale}'.format(**self.__dict__)


class BASIC(nn.Module):
    """
    This is the base class for Transformer Tracking.
    """

    def __init__(self, backbone, transformer, box_head, num_queries, aux_loss=False, head_type='CORNER'):
        """
        Initializes the model.

        Args:
            backbone: Torch module of the backbone to be used. See backbone.py
            transformer: Torch module of the transformer architecture. See transformer.py
            num_queries: Number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        self.hidden_dim = transformer.d_model
        self.foreground_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.background_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.bottleneck = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=(1, 1))
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == 'CORNER':
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.pool_sz = 4
        self.pool_len = self.pool_sz ** 2

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        """
        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [{'pred_boxes': b} for b in outputs_coord[:-1]]


class BaseTracker:
    """
    Base class for all trackers.
    """

    def __init__(self, params):
        self.params = params
        self.visdom = None

    def predicts_segmentation_mask(self):
        return False

    def initialize(self, image, info: 'dict') ->dict:
        """
        Overload this function in your tracker. This should initialize the model.
        """
        raise NotImplementedError

    def track(self, image, info: 'dict'=None) ->dict:
        """
        Overload this function in your tracker. This should track in the frame and update the model.
        """
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = box,
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')


class NestedTensor(object):

    def __init__(self, tensors, mask: 'Optional[Tensor]'):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class Preprocessor(object):

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))

    def process(self, img_arr: 'np.ndarray', amask_arr: 'np.ndarray'):
        img_tensor = torch.tensor(img_arr).float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = (img_tensor / 255.0 - self.mean) / self.std
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).unsqueeze(dim=0)
        return NestedTensor(img_tensor_norm, amask_tensor)

