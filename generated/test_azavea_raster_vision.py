
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


from typing import TYPE_CHECKING


import numpy as np


import warnings


import torch.distributed as dist


from typing import Iterable


from collections.abc import Callable


from enum import Enum


from torch import nn


from typing import Any


from typing import Literal


import torch


from torch.utils.data import Dataset


from collections import defaultdict


from typing import overload


from typing import Sequence


import matplotlib.pyplot as plt


import matplotlib.colors as mcolors


import matplotlib.patches as mpatches


from abc import ABC


from abc import abstractmethod


from torch import Tensor


from torch.utils.data import DataLoader


from typing import Iterator


from time import perf_counter


import numbers


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.multiprocessing as mp


import random


import uuid


from torch import optim


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import ConcatDataset


from torch.utils.data import Subset


from functools import reduce


from torchvision.ops import box_area


from torchvision.ops import box_convert


from torchvision.ops import batched_nms


from torchvision.ops import clip_boxes_to_image


from torchvision.utils import draw_bounding_boxes


import matplotlib.gridspec as gridspec


import torch.nn.functional as F


from torchvision import models


from torch.nn import functional as F


import torch.hub


from typing import Container


from torch.hub import _import_module


import pandas as pd


from uuid import uuid4


from matplotlib import pyplot as plt


class BoxList:

    def __init__(self, boxes: 'torch.Tensor', format: 'str'='xyxy', **extras) ->None:
        """Representation of a list of bounding boxes and associated data.

        Internally, boxes are always stored in the xyxy format.

        Args:
            boxes: tensor<n, 4>
            format: format of input boxes.
            extras: dict with values that are tensors with first dimension corresponding
                to boxes first dimension
        """
        self.extras = extras
        if format == 'xyxy':
            self.boxes = boxes
        elif format == 'yxyx':
            self.boxes = boxes[:, [1, 0, 3, 2]]
        else:
            self.boxes = box_convert(boxes, format, 'xyxy')

    def __contains__(self, key: 'str') ->bool:
        return key == 'boxes' or key in self.extras

    def get_field(self, name: 'str') ->Any:
        if name == 'boxes':
            return self.boxes
        else:
            return self.extras.get(name)

    def _map_extras(self, func: 'Callable[[str, Any], Any]', cond: 'Callable[[str, Any], bool]'=lambda k, v: True) ->dict:
        new_extras = {}
        for k, v in self.extras.items():
            if cond(k, v):
                new_extras[k] = func(k, v)
            else:
                new_extras[k] = v
        return new_extras

    def copy(self) ->'Self':
        return BoxList(self.boxes.copy(), **self._map_extras(lambda k, v: v.copy()), cond=lambda k, v: torch.is_tensor(v))

    def to(self, *args, **kwargs) ->'Self':
        """Recursively apply :meth:`torch.Tensor.to` to Tensors.

        Args:
            *args: Args for :meth:`torch.Tensor.to`.
            **kwargs: Keyword args for :meth:`torch.Tensor.to`.

        Returns:
            BoxList: New BoxList with to'd Tensors.
        """
        boxes = self.boxes
        extras = self._map_extras(func=lambda k, v: v, cond=lambda k, v: torch.is_tensor(v))
        return BoxList(boxes, **extras)

    def convert_boxes(self, out_fmt: 'str') ->torch.Tensor:
        if out_fmt == 'yxyx':
            boxes = self.boxes[:, [1, 0, 3, 2]]
        else:
            boxes = box_convert(self.boxes, 'xyxy', out_fmt)
        return boxes

    def __len__(self) ->int:
        return len(self.boxes)

    @staticmethod
    def cat(box_lists: "Iterable['Self']") ->'Self':
        boxes = []
        extras = defaultdict(list)
        for bl in box_lists:
            boxes.append(bl.boxes)
            for k, v in bl.extras.items():
                extras[k].append(v)
        boxes = torch.cat(boxes)
        for k, v in extras.items():
            extras[k] = torch.cat(v)
        return BoxList(boxes, **extras)

    def equal(self, other: "'Self'") ->bool:
        if len(other) != len(self):
            return False
        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float()) for v in self.extras.values()]
        cat_arr = torch.cat([self.boxes] + extras, 1)
        self_tups = set([tuple([x.item() for x in row]) for row in cat_arr])
        extras = [(v.float().unsqueeze(1) if v.ndim == 1 else v.float()) for v in other.extras.values()]
        cat_arr = torch.cat([other.boxes] + extras, 1)
        other_tups = set([tuple([x.item() for x in row]) for row in cat_arr])
        return self_tups == other_tups

    def ind_filter(self, inds: 'Sequence[int]') ->'Self':
        boxes = self.boxes[inds]
        extras = self._map_extras(func=lambda k, v: v[inds], cond=lambda k, v: torch.is_tensor(v))
        return BoxList(boxes, **extras)

    def score_filter(self, score_thresh: 'float'=0.25) ->'Self':
        scores = self.extras.get('scores')
        if scores is not None:
            return self.ind_filter(scores > score_thresh)
        else:
            raise ValueError('must have scores as key in extras')

    def clip_boxes(self, img_height: 'int', img_width: 'int') ->'Self':
        boxes = clip_boxes_to_image(self.boxes, (img_height, img_width))
        return BoxList(boxes, **self.extras)

    def nms(self, iou_thresh: 'float'=0.5) ->torch.Tensor:
        if len(self) == 0:
            return self
        good_inds = batched_nms(self.boxes, self.get_field('scores'), self.get_field('class_ids'), iou_thresh)
        return self.ind_filter(good_inds)

    def scale(self, yscale: 'float', xscale: 'float') ->'Self':
        """Scale box coords by the given scaling factors."""
        dtype = self.boxes.dtype
        boxes = self.boxes.float()
        boxes[:, [0, 2]] *= xscale
        boxes[:, [1, 3]] *= yscale
        self.boxes = boxes
        return self

    def pin_memory(self) ->'Self':
        self.boxes = self.boxes.pin_memory()
        for k, v in self.extras.items():
            if torch.is_tensor(v):
                self.extras[k] = v.pin_memory()
        return self

    def __repr__(self) ->str:
        return pformat(dict(boxes=self.boxes, **self.extras))


class RegressionModel(nn.Module):

    def __init__(self, backbone: 'nn.Module', out_features: 'int', pos_out_inds: 'Sequence[int] | None'=None, prob_out_inds: 'Sequence[int] | None'=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_features)
        self.pos_out_inds = pos_out_inds
        self.prob_out_inds = prob_out_inds

    def forward(self, x: "'torch.Tensor'") ->'torch.Tensor':
        out: "'torch.Tensor'" = self.backbone(x)
        if self.pos_out_inds:
            for ind in self.pos_out_inds:
                out[:, ind] = out[:, ind].exp()
        if self.prob_out_inds:
            for ind in self.prob_out_inds:
                out[:, ind] = out[:, ind].sigmoid()
        return out


class SplitTensor(nn.Module):
    """ Wrapper around `torch.split` """

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, X):
        return X.split(self.size_or_sizes, dim=self.dim)


class Parallel(nn.ModuleList):
    """ Passes inputs through multiple `nn.Module`s in parallel.
        Returns a tuple of outputs.
    """

    def __init__(self, *args):
        super().__init__(args)

    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            return tuple(m(xs) for m in self)
        assert len(xs) == len(self)
        return tuple(m(x) for m, x in zip(self, xs))


class AddTensors(nn.Module):
    """ Adds all its inputs together. """

    def forward(self, xs):
        return sum(xs)


class MockModel(nn.Module):

    def __init__(self, num_classes: 'int') ->None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, y=None):
        if self.training:
            assert y is not None
            return {'loss1': 0, 'loss2': 0}
        else:
            N = len(x)
            nboxes = np.random.randint(0, 10)
            outs = [{'boxes': torch.rand((nboxes, 4)), 'labels': torch.randint(0, self.num_classes, (nboxes,)), 'scores': torch.rand((nboxes,))} for _ in range(N)]
            return outs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AddTensors,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MockModel,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Parallel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SplitTensor,
     lambda: ([], {'size_or_sizes': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

