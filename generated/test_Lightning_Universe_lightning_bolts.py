
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


import inspect


import math


from typing import Sequence


from typing import Union


import torch.nn as nn


from torch import Tensor


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


import numpy as np


import torch


from torch import nn


from torch.nn import Module


from torch.utils.hooks import RemovableHandle


from typing import Tuple


from torch.nn import functional as F


from torch.optim import Optimizer


from abc import abstractmethod


from copy import deepcopy


from typing import Callable


from typing import Iterable


from typing import Type


import torch.nn.functional as F


import collections.abc as container_abcs


import re


from queue import Queue


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from abc import ABC


from collections import deque


from collections import namedtuple


from typing import Iterator


from torch.utils.data import IterableDataset


from torch.utils.data.dataset import random_split


from torch.utils.data import random_split


import logging


from warnings import warn


from torch.nn.functional import binary_cross_entropy


from torch.nn.functional import binary_cross_entropy_with_logits


from torch.nn.functional import one_hot


from collections import OrderedDict


from copy import copy


from torch import optim


from torch.optim import Adam


from torch.optim.optimizer import Optimizer


import collections


from torch import FloatTensor


from torch.distributions import Categorical


from torch.distributions import Normal


from torch.nn.functional import log_softmax


from torch.nn.functional import softmax


from torch.utils.model_zoo import load_url as load_state_dict_from_url


from torch import distributed as dist


from torch.optim.optimizer import required


import warnings


from torch.optim.lr_scheduler import _LRScheduler


from torchvision.transforms import InterpolationMode


import random


import uuid


from torchvision import transforms as transform_lib


import torch.testing


from torch.utils.data import Subset


from torch.utils.data.dataloader import DataLoader


import torch.cuda


from torch.optim import SGD


from collections import Counter


class UnderReviewWarning(Warning):
    pass


def _create_full_message(message: 'str') ->str:
    return f'{message} The compatibility with other Lightning projects is not guaranteed and API may change at any time. The API and functionality may change without warning in future releases. More details: https://lightning-bolts.readthedocs.io/en/latest/stability.html'


def _create_docstring_message(docstring: 'str', message: 'str') ->str:
    rst_warning = '.. warning:: ' + _create_full_message(message)
    if docstring is None:
        return rst_warning
    return rst_warning + '\n\n    ' + docstring


def _raise_review_warning(message: 'str', stacklevel: 'int'=6) ->None:
    rank_zero_warn(_create_full_message(message), stacklevel=stacklevel, category=UnderReviewWarning)


def under_review():
    """The under_review decorator is used to indicate that a particular feature is not properly reviewed and tested yet.

    A callable or type that has been marked as under_review will give a ``UnderReviewWarning`` when it is called or
    instantiated. This designation should be used following the description given in :ref:`stability`.
    Args:
        message: The message to include in the warning.
    Examples
    ________
    >>> from pytest import warns
    >>> from pl_bolts.utils.stability import under_review, UnderReviewWarning
    >>> @under_review()
    ... class MyExperimentalFeature:
    ...     pass
    ...
    >>> with warns(UnderReviewWarning, match="The feature MyExperimentalFeature is currently marked under review."):
    ...     MyExperimentalFeature()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ...
    <...>

    """

    def decorator(cls_or_callable: 'Union[Callable, Type]', feature_name: 'Optional[str]'=None, was_class: 'bool'=False):
        if feature_name is None:
            feature_name = cls_or_callable.__qualname__
        message = f'The feature {feature_name} is currently marked under review.'
        filterwarnings('once', message, UnderReviewWarning)
        if inspect.isclass(cls_or_callable):
            cls_or_callable.__init__ = decorator(cls_or_callable.__init__, feature_name=cls_or_callable.__qualname__, was_class=True)
            cls_or_callable.__doc__ = _create_docstring_message(cls_or_callable.__doc__, message)
            return cls_or_callable

        @functools.wraps(cls_or_callable)
        def wrapper(*args, **kwargs):
            _raise_review_warning(message)
            return cls_or_callable(*args, **kwargs)
        if not was_class:
            wrapper.__doc__ = _create_docstring_message(cls_or_callable.__doc__, message)
        return wrapper
    return decorator


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _get_padding(kernel_size: 'int', stride: 'int') ->Tuple[int, nn.Module]:
    """Returns the amount of padding needed by convolutional and max pooling layers.

    Determines the amount of padding needed to make the output size of the layer the input size divided by the stride.
    The first value that the function returns is the amount of padding to be added to all sides of the input matrix
    (``padding`` argument of the operation). If an uneven amount of padding is needed in different sides of the input,
    the second variable that is returned is an ``nn.ZeroPad2d`` operation that adds an additional column and row of
    padding. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        kernel_size: Size of the kernel.
        stride: Stride of the operation.

    Returns:
        padding, pad_op: The amount of padding to be added to all sides of the input and an ``nn.Identity`` or
        ``nn.ZeroPad2d`` operation to add one more column and row of padding if necessary.

    """
    padding, remainder = divmod(max(kernel_size, stride) - stride, 2)
    pad_op: 'nn.Module' = nn.Identity() if remainder == 0 else nn.ZeroPad2d((0, 1, 0, 1))
    return padding, pad_op


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x: 'Tensor') ->Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


def create_activation_module(name: 'Optional[str]') ->nn.Module:
    """Creates a layer activation module given its type as a string.

    Args:
        name: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic", "linear",
            or "none".

    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky':
        return nn.LeakyReLU(0.1, inplace=True)
    if name == 'mish':
        return Mish()
    if name == 'silu' or name == 'swish':
        return nn.SiLU(inplace=True)
    if name == 'logistic':
        return nn.Sigmoid()
    if name == 'linear' or name == 'none' or name is None:
        return nn.Identity()
    raise ValueError(f'Activation type `{name}´ is unknown.')


def create_normalization_module(name: 'Optional[str]', num_channels: 'int') ->nn.Module:
    """Creates a layer normalization module given its type as a string.

    Group normalization uses always 8 channels. The most common network widths are divisible by this number.

    Args:
        name: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        num_channels: The number of input channels that the module expects.

    """
    if name == 'batchnorm':
        return nn.BatchNorm2d(num_channels, eps=0.001)
    if name == 'groupnorm':
        return nn.GroupNorm(8, num_channels, eps=0.001)
    if name == 'none' or name is None:
        return nn.Identity()
    raise ValueError(f'Normalization layer type `{name}´ is unknown.')


class Conv(nn.Module):
    """A convolutional layer with optional layer normalization and activation.

    If ``padding`` is ``None``, the module tries to add padding so much that the output size will be the input size
    divided by the stride. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        in_channels: Number of input channels that the layer expects.
        out_channels: Number of output channels that the convolution produces.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        bias: If ``True``, adds a learnable bias to the output.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1, stride: 'int'=1, padding: 'Optional[int]'=None, bias: 'bool'=False, activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm'):
        super().__init__()
        if padding is None:
            padding, self.pad = _get_padding(kernel_size, stride)
        else:
            self.pad = nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = create_normalization_module(norm, out_channels)
        self.act = create_activation_module(activation)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


PRED = Dict[str, Any]


PREDS = Union[Tuple[PRED, ...], List[PRED]]


TARGET = Dict[str, Any]


def grid_offsets(grid_size: 'Tensor') ->Tensor:
    """Given a grid size, returns a tensor containing offsets to the grid cells.

    Args:
        The width and height of the grid in a tensor.

    Returns:
        A ``[height, width, 2]`` tensor containing the grid cell `(x, y)` offsets.

    """
    x_range = torch.arange(grid_size[0].item(), device=grid_size.device)
    y_range = torch.arange(grid_size[1].item(), device=grid_size.device)
    grid_y, grid_x = meshgrid(y_range, x_range)
    return torch.stack((grid_x, grid_y), -1)


class DetectionLayer(nn.Module):
    """A YOLO detection layer.

    A YOLO model has usually 1 - 3 detection layers at different resolutions. The loss is summed from all of them.

    Args:
        num_classes: Number of different classes that this layer predicts.
        prior_shapes: A list of prior box dimensions for this layer, used for scaling the predicted dimensions. The list
            should contain (width, height) tuples in the network input resolution.
        matching_func: The matching algorithm to be used for assigning targets to anchors.
        loss_func: ``YOLOLoss`` object for calculating the losses.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.

    """

    def __init__(self, num_classes: 'int', prior_shapes: 'List[Tuple[int, int]]', matching_func: 'Callable', loss_func: 'YOLOLoss', xy_scale: 'float'=1.0, input_is_normalized: 'bool'=False) ->None:
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError('YOLO model uses `torchvision`, which is not installed yet.')
        self.num_classes = num_classes
        self.prior_shapes = prior_shapes
        self.matching_func = matching_func
        self.loss_func = loss_func
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized

    def forward(self, x: 'Tensor', image_size: 'Tensor') ->Tuple[Tensor, PREDS]:
        """Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the image space, scales the bounding boxes with the
        anchors, converts the center coordinates to corner coordinates, and maps probabilities to the `]0, 1[` range
        using sigmoid.

        If targets are given, computes also losses from the predictions and the targets. This layer is responsible only
        for the targets that best match one of the anchors assigned to this layer. Training losses will be saved to the
        ``losses`` attribute. ``hits`` attribute will be set to the number of targets that this layer was responsible
        for. ``losses`` is a tensor of three elements: the overlap, confidence, and classification loss.

        Args:
            x: The output from the previous layer. The size of this tensor has to be
                ``[batch_size, anchors_per_cell * (num_classes + 5), height, width]``.
            image_size: Image width and height in a vector (defines the scale of the predicted and target coordinates).

        Returns:
            The layer output, with normalized probabilities, in a tensor sized
            ``[batch_size, anchors_per_cell * height * width, num_classes + 5]`` and a list of dictionaries, containing
            the same predictions, but with unnormalized probabilities (for loss calculation).

        """
        batch_size, num_features, height, width = x.shape
        num_attrs = self.num_classes + 5
        anchors_per_cell = int(torch.div(num_features, num_attrs, rounding_mode='floor'))
        if anchors_per_cell != len(self.prior_shapes):
            raise ValueError('The model predicts {} bounding boxes per spatial location, but {} prior box dimensions are defined for this layer.'.format(anchors_per_cell, len(self.prior_shapes)))
        x = x.permute(0, 2, 3, 1)
        x = x.view(batch_size, height, width, anchors_per_cell, num_attrs)
        norm_x = x if self.input_is_normalized else torch.sigmoid(x)
        xy = norm_x[..., :2]
        wh = x[..., 2:4]
        confidence = x[..., 4]
        classprob = x[..., 5:]
        norm_confidence = norm_x[..., 4]
        norm_classprob = norm_x[..., 5:]
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)
        image_xy = global_xy(xy, image_size)
        prior_shapes = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        image_wh = 4 * torch.square(wh) * prior_shapes if self.input_is_normalized else torch.exp(wh) * prior_shapes
        box = torch.cat((image_xy, image_wh), -1)
        box = box_convert(box, in_fmt='cxcywh', out_fmt='xyxy')
        output = torch.cat((box, norm_confidence.unsqueeze(-1), norm_classprob), -1)
        output = output.reshape(batch_size, height * width * anchors_per_cell, num_attrs)
        preds = [{'boxes': b, 'confidences': c, 'classprobs': p} for b, c, p in zip(box, confidence, classprob)]
        return output, preds

    def match_targets(self, preds: 'PREDS', return_preds: 'PREDS', targets: 'TARGETS', image_size: 'Tensor') ->Tuple[PRED, TARGET]:
        """Matches the predictions to targets.

        Args:
            preds: List of predictions for each image, as returned by the ``forward()`` method of this layer. These will
                be matched to the training targets.
            return_preds: List of predictions for each image. The matched predictions will be returned from this list.
                When calculating the auxiliary loss for deep supervision, predictions from a different layer are used
                for loss computation.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.

        Returns:
            Two dictionaries, the matched predictions and targets.

        """
        batch_size = len(preds)
        if len(targets) != batch_size or len(return_preds) != batch_size:
            raise ValueError('Different batch size for predictions and targets.')
        matches = []
        for image_preds, image_return_preds, image_targets in zip(preds, return_preds, targets):
            if image_targets['boxes'].shape[0] > 0:
                pred_selector, background_selector, target_selector = self.matching_func(image_preds, image_targets, image_size)
                matched_preds = {'boxes': image_return_preds['boxes'][pred_selector], 'confidences': image_return_preds['confidences'][pred_selector], 'bg_confidences': image_return_preds['confidences'][background_selector], 'classprobs': image_return_preds['classprobs'][pred_selector]}
                matched_targets = {'boxes': image_targets['boxes'][target_selector], 'labels': image_targets['labels'][target_selector]}
            else:
                matched_preds = {'boxes': torch.empty((0, 4), device=image_return_preds['boxes'].device), 'confidences': torch.empty(0, device=image_return_preds['confidences'].device), 'bg_confidences': image_return_preds['confidences'].flatten(), 'classprobs': torch.empty((0, self.num_classes), device=image_return_preds['classprobs'].device)}
                matched_targets = {'boxes': torch.empty((0, 4), device=image_targets['boxes'].device), 'labels': torch.empty(0, dtype=torch.int64, device=image_targets['labels'].device)}
            matches.append((matched_preds, matched_targets))
        matched_preds = {'boxes': torch.cat(tuple(m[0]['boxes'] for m in matches)), 'confidences': torch.cat(tuple(m[0]['confidences'] for m in matches)), 'bg_confidences': torch.cat(tuple(m[0]['bg_confidences'] for m in matches)), 'classprobs': torch.cat(tuple(m[0]['classprobs'] for m in matches))}
        matched_targets = {'boxes': torch.cat(tuple(m[1]['boxes'] for m in matches)), 'labels': torch.cat(tuple(m[1]['labels'] for m in matches))}
        return matched_preds, matched_targets

    def calculate_losses(self, preds: 'PREDS', targets: 'TARGETS', image_size: 'Tensor', loss_preds: 'Optional[PREDS]'=None) ->Tuple[Tensor, int]:
        """Matches the predictions to targets and computes the losses.

        Args:
            preds: List of predictions for each image, as returned by ``forward()``. These will be matched to the
                training targets and used to compute the losses (unless another set of predictions for loss computation
                is given in ``loss_preds``).
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
            loss_preds: List of predictions for each image. If given, these will be used for loss computation, instead
                of the same predictions that were used for matching. This is needed for deep supervision in YOLOv7.

        Returns:
            A vector of the overlap, confidence, and classification loss, normalized by batch size, and the number of
            targets that were matched to this layer.

        """
        if loss_preds is None:
            loss_preds = preds
        matched_preds, matched_targets = self.match_targets(preds, loss_preds, targets, image_size)
        losses = self.loss_func.elementwise_sums(matched_preds, matched_targets, self.input_is_normalized, image_size)
        losses = torch.stack((losses.overlap, losses.confidence, losses.classification)) / len(preds)
        hits = len(matched_targets['boxes'])
        return losses, hits


NETWORK_OUTPUT = Tuple[List[Tensor], List[Tensor], List[int]]


class RouteLayer(nn.Module):
    """A routing layer concatenates the output (or part of it) from given layers.

    Args:
        source_layers: Indices of the layers whose output will be concatenated.
        num_chunks: Layer outputs will be split into this number of chunks.
        chunk_idx: Only the chunks with this index will be concatenated.

    """

    def __init__(self, source_layers: 'List[int]', num_chunks: 'int', chunk_idx: 'int') ->None:
        super().__init__()
        self.source_layers = source_layers
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

    def forward(self, outputs: 'List[Tensor]') ->Tensor:
        chunks = [torch.chunk(outputs[layer], self.num_chunks, dim=1)[self.chunk_idx] for layer in self.source_layers]
        return torch.cat(chunks, dim=1)


class ShortcutLayer(nn.Module):
    """A shortcut layer adds a residual connection from the source layer.

    Args:
        source_layer: Index of the layer whose output will be added to the output of the previous layer.

    """

    def __init__(self, source_layer: 'int') ->None:
        super().__init__()
        self.source_layer = source_layer

    def forward(self, outputs: 'List[Tensor]') ->Tensor:
        return outputs[-1] + outputs[self.source_layer]


CREATE_LAYER_OUTPUT = Tuple[nn.Module, int]


def _create_convolutional(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a convolutional layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    batch_normalize = config.get('batch_normalize', False)
    padding = (config['size'] - 1) // 2 if config['pad'] else 0
    layer = Conv(num_inputs[-1], config['filters'], kernel_size=config['size'], stride=config['stride'], padding=padding, bias=not batch_normalize, activation=config['activation'], norm='batchnorm' if batch_normalize else None)
    return layer, config['filters']


class MaxPool(nn.Module):
    """A max pooling layer with padding.

    The module tries to add padding so much that the output size will be the input size divided by the stride. If the
    input size is not divisible by the stride, the output size will be rounded upwards.

    """

    def __init__(self, kernel_size: 'int', stride: 'int'):
        super().__init__()
        padding, self.pad = _get_padding(kernel_size, stride)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.pad(x)
        return self.maxpool(x)


def _create_maxpool(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = MaxPool(config['size'], config['stride'])
    return layer, num_inputs[-1]


def _create_route(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a routing layer.

    A routing layer concatenates the output (or part of it) from the layers specified by the "layers" configuration
    option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    num_chunks = config.get('groups', 1)
    chunk_idx = config.get('group_id', 0)
    last = len(num_inputs) - 1
    source_layers = [(layer if layer >= 0 else last + layer) for layer in config['layers']]
    layer = RouteLayer(source_layers, num_chunks, chunk_idx)
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)
    return layer, num_outputs


def _create_shortcut(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a shortcut layer.

    A shortcut layer adds a residual connection from the layer specified by the "from" configuration option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = ShortcutLayer(config['from'])
    return layer, num_inputs[-1]


def _create_upsample(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a layer that upsamples the data.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = nn.Upsample(scale_factor=config['stride'], mode='nearest')
    return layer, num_inputs[-1]


def iou_below(pred_boxes: 'Tensor', target_boxes: 'Tensor', threshold: 'float') ->Tensor:
    """Creates a binary mask whose value will be ``True``, unless the predicted box overlaps any target
    significantly (IoU greater than ``threshold``).

    Args:
        pred_boxes: The predicted corner coordinates. Tensor of size ``[height, width, boxes_per_cell, 4]``.
        target_boxes: Corner coordinates of the target boxes. Tensor of size ``[height, width, boxes_per_cell, 4]``.

    Returns:
        A boolean tensor sized ``[height, width, boxes_per_cell]``, with ``False`` where the predicted box overlaps a
        target significantly and ``True`` elsewhere.
    """
    shape = pred_boxes.shape[:-1]
    pred_boxes = pred_boxes.view(-1, 4)
    ious = box_iou(pred_boxes, target_boxes)
    best_iou = ious.max(-1).values
    below_threshold = best_iou <= threshold
    return below_threshold.view(shape)


class ShapeMatching(ABC):
    """Selects which anchors are used to predict each target, by comparing the shape of the target box to a set of prior
    shapes.

    Most YOLO variants match targets to anchors based on prior shapes that are assigned to the anchors in the model
    configuration. The subclasses of ``ShapeMatching`` implement matching rules that compare the width and height of
    the targets to each prior shape (regardless of the location where the target is). When the model includes multiple
    detection layers, different shapes are defined for each layer. Usually there are three detection layers and three
    prior shapes per layer.

    Args:
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(self, ignore_bg_threshold: 'float'=0.7) ->None:
        self.ignore_bg_threshold = ignore_bg_threshold

    def __call__(self, preds: 'Dict[str, Tensor]', targets: 'Dict[str, Tensor]', image_size: 'Tensor') ->Tuple[List[Tensor], Tensor, Tensor]:
        """For each target, selects predictions from the same grid cell, where the center of the target box is.

        Typically there are three predictions per grid cell. Subclasses implement ``match()``, which selects the
        predictions within the grid cell.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.

        Returns:
            The indices of the matched predictions, background mask, and a mask for selecting the matched targets.

        """
        height, width = preds['boxes'].shape[:2]
        device = preds['boxes'].device
        grid_size = torch.tensor([width, height], device=device)
        image_to_grid = torch.true_divide(grid_size, image_size)
        xywh = box_convert(targets['boxes'], in_fmt='xyxy', out_fmt='cxcywh')
        grid_xy = xywh[:, :2] * image_to_grid
        cell_i = grid_xy[:, 0].clamp(0, width - 1)
        cell_j = grid_xy[:, 1].clamp(0, height - 1)
        target_selector, anchor_selector = self.match(xywh[:, 2:])
        cell_i = cell_i[target_selector]
        cell_j = cell_j[target_selector]
        background_mask = iou_below(preds['boxes'], targets['boxes'], self.ignore_bg_threshold)
        background_mask[cell_j, cell_i, anchor_selector] = False
        pred_selector = [cell_j, cell_i, anchor_selector]
        return pred_selector, background_mask, target_selector

    @abstractmethod
    def match(self, wh: 'Tensor') ->Union[Tuple[Tensor, Tensor], Tensor]:
        """Selects anchors for each target based on the predicted shapes. The subclasses implement this method.

        Args:
            wh: A matrix of predicted width and height values.

        Returns:
            matched_targets, matched_anchors: Two vectors or a `2xN` matrix. The first vector is used to select the
            targets that this layer matched and the second one lists the matching anchors within the grid cell.

        """
        pass


def aligned_iou(wh1: 'Tensor', wh2: 'Tensor') ->Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at the
    same coordinates.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of pairwise IoU values for every element in ``wh1`` and ``wh2``

    """
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    inter_wh = torch.min(wh1[:, None, :], wh2)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union


class HighestIoUMatching(ShapeMatching):
    """For each target, select the prior shape that gives the highest IoU.

    This is the original YOLO matching rule.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.

    """

    def __init__(self, prior_shapes: 'Sequence[Tuple[int, int]]', prior_shape_idxs: 'Sequence[int]', ignore_bg_threshold: 'float'=0.7) ->None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = prior_shapes
        self.anchor_map = [(prior_shape_idxs.index(idx) if idx in prior_shape_idxs else -1) for idx in range(len(prior_shapes))]

    def match(self, wh: 'Tensor') ->Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=wh.device)
        ious = aligned_iou(wh, prior_wh)
        highest_iou_anchors = ious.max(1).indices
        highest_iou_anchors = anchor_map[highest_iou_anchors]
        matched_targets = highest_iou_anchors >= 0
        matched_anchors = highest_iou_anchors[matched_targets]
        return matched_targets, matched_anchors


class IoUThresholdMatching(ShapeMatching):
    """For each target, select all prior shapes that give a high enough IoU.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: IoU treshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.

    """

    def __init__(self, prior_shapes: 'Sequence[Tuple[int, int]]', prior_shape_idxs: 'Sequence[int]', threshold: 'float', ignore_bg_threshold: 'float'=0.7) ->None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: 'Tensor') ->Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        ious = aligned_iou(wh, prior_wh)
        above_threshold = (ious > self.threshold).nonzero()
        return above_threshold.T


def _sim_ota_match(costs: 'Tensor', ious: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Implements the SimOTA matching rule.

    The number of units supplied by each supplier (training target) needs to be decided in the Optimal Transport
    problem. "Dynamic k Estimation" uses the sum of the top 10 IoU values (casted to int) between the target and the
    predicted boxes.

    Args:
        costs: A ``[predictions, targets]`` matrix of losses.
        ious: A ``[predictions, targets]`` matrix of IoUs.

    Returns:
        A mask of predictions that were matched, and the indices of the matched targets. The latter contains as many
        elements as there are ``True`` values in the mask.

    """
    num_preds, num_targets = ious.shape
    matching_matrix = torch.zeros_like(costs, dtype=torch.bool)
    if ious.numel() > 0:
        top10_iou = torch.topk(ious, min(10, num_preds), dim=0).values.sum(0)
        ks = torch.clip(top10_iou.int(), min=1)
        assert len(ks) == num_targets
        for target_idx, (target_costs, k) in enumerate(zip(costs.T, ks)):
            pred_idx = torch.topk(target_costs, k, largest=False).indices
            matching_matrix[pred_idx, target_idx] = True
        more_than_one_match = matching_matrix.sum(1) > 1
        best_targets = costs[more_than_one_match, :].argmin(1)
        matching_matrix[more_than_one_match, :] = False
        matching_matrix[more_than_one_match, best_targets] = True
    pred_mask = matching_matrix.sum(1) > 0
    target_selector = matching_matrix[pred_mask, :].int().argmax(1)
    return pred_mask, target_selector


def box_size_ratio(wh1: 'Tensor', wh2: 'Tensor') ->Tensor:
    """Compares the dimensions of the boxes pairwise.

    For each pair of boxes, calculates the largest ratio that can be obtained by dividing the widths with each other or
    dividing the heights with each other.

    Args:
        wh1: An ``[N, 2]`` matrix of box shapes (width and height).
        wh2: An ``[M, 2]`` matrix of box shapes (width and height).

    Returns:
        An ``[N, M]`` matrix of ratios of width or height dimensions, whichever is larger.

    """
    wh_ratio = wh1[:, None, :] / wh2[None, :, :]
    wh_ratio = torch.max(wh_ratio, 1.0 / wh_ratio)
    return wh_ratio.max(2).values


def grid_centers(grid_size: 'Tensor') ->Tensor:
    """Given a grid size, returns a tensor containing coordinates to the centers of the grid cells.

    Returns:
        A ``[height, width, 2]`` tensor containing coordinates to the centers of the grid cells.

    """
    return grid_offsets(grid_size) + 0.5


def is_inside_box(points: 'Tensor', boxes: 'Tensor') ->Tensor:
    """Get pairwise truth values of whether the point is inside the box.

    Args:
        points: Point (x, y) coordinates, a tensor shaped ``[points, 2]``.
        boxes: Box (x1, y1, x2, y2) coordinates, a tensor shaped ``[boxes, 4]``.

    Returns:
        A tensor shaped ``[points, boxes]`` containing pairwise truth values of whether the points are inside the boxes.

    """
    lt = points[:, None, :] - boxes[None, :, :2]
    rb = boxes[None, :, 2:] - points[:, None, :]
    deltas = torch.cat((lt, rb), -1)
    return deltas.min(-1).values > 0.0


class SimOTAMatching:
    """Selects which anchors are used to predict each target using the SimOTA matching rule.

    This is the matching rule used by YOLOX.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        loss_func: A ``YOLOLoss`` object that can be used to calculate the pairwise costs.
        spatial_range: For each target, restrict to the anchors that are within an `N × N` grid cell are centered at the
            target, where `N` is the value of this parameter.
        size_range: For each target, restrict to the anchors whose prior dimensions are not larger than the target
            dimensions multiplied by this value and not smaller than the target dimensions divided by this value.

    """

    def __init__(self, prior_shapes: 'Sequence[Tuple[int, int]]', prior_shape_idxs: 'Sequence[int]', loss_func: 'YOLOLoss', spatial_range: 'float', size_range: 'float') ->None:
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.loss_func = loss_func
        self.spatial_range = spatial_range
        self.size_range = size_range

    def __call__(self, preds: 'Dict[str, Tensor]', targets: 'Dict[str, Tensor]', image_size: 'Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        """For each target, selects predictions using the SimOTA matching rule.

        Args:
            preds: Predictions for a single image.
            targets: Training targets for a single image.
            image_size: Input image width and height.

        Returns:
            A mask of predictions that were matched, background mask (inverse of the first mask), and the indices of the
            matched targets. The last tensor contains as many elements as there are ``True`` values in the first mask.

        """
        height, width, boxes_per_cell, _ = preds['boxes'].shape
        prior_mask, anchor_inside_target = self._get_prior_mask(targets, image_size, width, height, boxes_per_cell)
        prior_preds = {'boxes': preds['boxes'][prior_mask], 'confidences': preds['confidences'][prior_mask], 'classprobs': preds['classprobs'][prior_mask]}
        losses, ious = self.loss_func.pairwise(prior_preds, targets, input_is_normalized=False)
        costs = losses.overlap + losses.confidence + losses.classification
        costs += 100000.0 * ~anchor_inside_target
        pred_mask, target_selector = _sim_ota_match(costs, ious)
        prior_mask[prior_mask.nonzero().T.tolist()] = pred_mask
        background_mask = torch.logical_not(prior_mask)
        return prior_mask, background_mask, target_selector

    def _get_prior_mask(self, targets: 'Dict[str, Tensor]', image_size: 'Tensor', grid_width: 'int', grid_height: 'int', boxes_per_cell: 'int') ->Tuple[Tensor, Tensor]:
        """Creates a mask for selecting the "center prior" anchors.

        In the first step we restrict ourselves to the grid cells whose center is inside or close enough to one or more
        targets.

        Args:
            targets: Training targets for a single image.
            image_size: Input image width and height.
            grid_width: Width of the feature grid.
            grid_height: Height of the feature grid.
            boxes_per_cell: Number of boxes that will be predicted per feature grid cell.

        Returns:
            Two masks, a ``[grid_height, grid_width, boxes_per_cell]`` mask for selecting anchors that are close and
            similar in shape to a target, and an ``[anchors, targets]`` matrix that indicates which targets are inside
            those anchors.

        """
        grid_size = torch.tensor([grid_width, grid_height], device=targets['boxes'].device)
        grid_to_image = torch.true_divide(image_size, grid_size)
        xywh = box_convert(targets['boxes'], in_fmt='xyxy', out_fmt='cxcywh')
        xy = xywh[:, :2]
        wh = xywh[:, 2:]
        prior_wh = torch.tensor(self.prior_shapes, device=targets['boxes'].device)
        shape_selector = box_size_ratio(prior_wh, wh) < self.size_range
        centers = grid_centers(grid_size).view(-1, 2) * grid_to_image
        inside_selector = is_inside_box(centers, targets['boxes'])
        inside_selector = inside_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        inside_selector = torch.logical_and(inside_selector, shape_selector)
        wh = self.spatial_range * grid_to_image * torch.ones_like(xy)
        xywh = torch.cat((xy, wh), -1)
        boxes = box_convert(xywh, in_fmt='cxcywh', out_fmt='xyxy')
        close_selector = is_inside_box(centers, boxes)
        close_selector = close_selector[:, None, :].repeat(1, boxes_per_cell, 1)
        close_selector = torch.logical_and(close_selector, shape_selector)
        mask = torch.logical_or(inside_selector, close_selector).sum(-1) > 0
        mask = mask.view(grid_height, grid_width, boxes_per_cell)
        inside_selector = inside_selector.view(grid_height, grid_width, boxes_per_cell, -1)
        return mask, inside_selector[mask]


class SizeRatioMatching(ShapeMatching):
    """For each target, select those prior shapes, whose width and height relative to the target is below given ratio.

    This is the matching rule used by Ultralytics YOLOv5 implementation.

    Args:
        prior_shapes: A list of all the prior box dimensions. The list should contain (width, height) tuples in the
            network input resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        threshold: Size ratio threshold for matching.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.

    """

    def __init__(self, prior_shapes: 'Sequence[Tuple[int, int]]', prior_shape_idxs: 'Sequence[int]', threshold: 'float', ignore_bg_threshold: 'float'=0.7) ->None:
        super().__init__(ignore_bg_threshold)
        self.prior_shapes = [prior_shapes[idx] for idx in prior_shape_idxs]
        self.threshold = threshold

    def match(self, wh: 'Tensor') ->Union[Tuple[Tensor, Tensor], Tensor]:
        prior_wh = torch.tensor(self.prior_shapes, dtype=wh.dtype, device=wh.device)
        return (box_size_ratio(wh, prior_wh) < self.threshold).nonzero().T


def _background_confidence_loss(preds: 'Tensor', bce_func: 'Callable') ->Tensor:
    """Calculates the sum of the confidence losses for background anchors.

    Args:
        preds: A vector of predicted confidences for background anchors.
        bce_func: A function for calculating binary cross entropy.

    Returns:
        The sum of the background confidence losses.

    """
    targets = torch.zeros_like(preds)
    return bce_func(preds, targets, reduction='sum')


def _foreground_confidence_loss(preds: 'Tensor', overlap: 'Tensor', bce_func: 'Callable', predict_overlap: 'Optional[float]') ->Tensor:
    """Calculates the sum of the confidence losses for foreground anchors and their matched targets.

    If ``predict_overlap`` is ``None``, the target confidence will be 1. If ``predict_overlap`` is 1.0, ``overlap`` will
    be used as the target confidence. Otherwise this parameter defines a balance between these two targets. The method
    returns a vector of losses for each foreground anchor.

    Args:
        preds: A vector of predicted confidences.
        overlap: A vector of overlaps between matched target and predicted bounding boxes.
        bce_func: A function for calculating binary cross entropy.
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1, and 1.0 means that the target confidence is the overlap.

    Returns:
        The sum of the confidence losses for foreground anchors.

    """
    targets = torch.ones_like(preds)
    if predict_overlap is not None:
        targets -= predict_overlap
        targets += predict_overlap * overlap.detach().clamp(min=0)
    return bce_func(preds, targets, reduction='sum')


def box_iou_loss(boxes1: 'Tensor', boxes2: 'Tensor') ->Tensor:
    return 1.0 - box_iou(boxes1, boxes2).diagonal()


def _get_iou_and_loss_functions(name: 'str') ->Tuple[Callable, Callable]:
    """Returns functions for calculating the IoU and the IoU loss, given the IoU variant name.

    Args:
        name: Name of the IoU variant. Either "iou", "giou", "diou", or "ciou".

    Returns:
        A tuple of two functions. The first function calculates the pairwise IoU and the second function calculates the
        elementwise loss.

    """
    if name not in _iou_and_loss_functions:
        raise ValueError(f"Unknown IoU function '{name}'.")
    iou_func, loss_func = _iou_and_loss_functions[name]
    if not callable(iou_func):
        raise ValueError(f"The IoU function '{name}' is not supported by the installed version of Torchvision.")
    assert callable(loss_func)
    return iou_func, loss_func


def _pairwise_confidence_loss(preds: 'Tensor', overlap: 'Tensor', bce_func: 'Callable', predict_overlap: 'Optional[float]') ->Tensor:
    """Calculates the confidence loss for every pair of a foreground anchor and a target.

    If ``predict_overlap`` is ``None``, the target confidence will be 1. If ``predict_overlap`` is 1.0, ``overlap`` will
    be used as the target confidence. Otherwise this parameter defines a balance between these two targets. The method
    returns a vector of losses for each foreground anchor.

    Args:
        preds: An ``[N]`` vector of predicted confidences.
        overlap: An ``[N, M]`` matrix of overlaps between all predicted and target bounding boxes.
        bce_func: A function for calculating binary cross entropy.
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the overlap.

    Returns:
        An ``[N, M]`` matrix of confidence losses between all predictions and targets.

    """
    if predict_overlap is not None:
        preds = preds.unsqueeze(1).expand(overlap.shape)
        targets = torch.ones_like(preds) - predict_overlap
        targets += predict_overlap * overlap.detach().clamp(min=0)
        return bce_func(preds, targets, reduction='none')
    targets = torch.ones_like(preds)
    return bce_func(preds, targets, reduction='none').unsqueeze(1).expand(overlap.shape)


def _size_compensation(targets: 'Tensor', image_size: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Calcuates the size compensation factor for the overlap loss.

    The overlap losses for each target should be multiplied by the returned weight. The returned value is
    `2 - (unit_width * unit_height)`, which is large for small boxes (the maximum value is 2) and small for large boxes
    (the minimum value is 1).

    Args:
        targets: An ``[N, 4]`` matrix of target `(x1, y1, x2, y2)` coordinates.
        image_size: Image size, which is used to scale the target boxes to unit coordinates.

    Returns:
        The size compensation factor.

    """
    unit_wh = targets[:, 2:] / image_size
    return 2 - unit_wh[:, 0] * unit_wh[:, 1]


def _target_labels_to_probs(targets: 'Tensor', num_classes: 'int', dtype: 'torch.dtype', label_smoothing: 'Optional[float]'=None) ->Tensor:
    """If ``targets`` is a vector of class labels, converts it to a matrix of one-hot class probabilities.

    If label smoothing is disabled, the returned target probabilities will be binary. If label smoothing is enabled, the
    target probabilities will be, ``(label_smoothing / 2)`` or ``(label_smoothing / 2) + (1.0 - label_smoothing)``. That
    corresponds to label smoothing with two categories, since the YOLO model does multi-label classification.

    Args:
        targets: An ``[M, C]`` matrix of target class probabilities or an ``[M]`` vector of class labels.
        num_classes: The number of classes (C dimension) for the new targets. If ``targets`` is already two-dimensional,
            checks that the length of the second dimension matches this number.
        dtype: Floating-point data type to be used for the one-hot targets.
        label_smoothing: The epsilon parameter (weight) for label smoothing. 0.0 means no smoothing (binary targets),
            and 1.0 means that the target probabilities are always 0.5.

    Returns:
        An ``[M, C]`` matrix of target class probabilities.
    """
    if targets.ndim == 1:
        last_class = torch.tensor(num_classes - 1, device=targets.device)
        targets = torch.min(targets, last_class)
        targets = one_hot(targets, num_classes)
    elif targets.shape[-1] != num_classes:
        raise ValueError(f"The number of classes in the data ({targets.shape[-1]}) doesn't match the number of classes predicted by the model ({num_classes}).")
    targets = targets
    if label_smoothing is not None:
        targets = label_smoothing / 2 + targets * (1.0 - label_smoothing)
    return targets


def create_detection_layer(prior_shapes: 'Sequence[Tuple[int, int]]', prior_shape_idxs: 'Sequence[int]', matching_algorithm: 'Optional[str]'=None, matching_threshold: 'Optional[float]'=None, spatial_range: 'float'=5.0, size_range: 'float'=4.0, ignore_bg_threshold: 'float'=0.7, overlap_func: 'Union[str, Callable]'='ciou', predict_overlap: 'Optional[float]'=None, label_smoothing: 'Optional[float]'=None, overlap_loss_multiplier: 'float'=5.0, confidence_loss_multiplier: 'float'=1.0, class_loss_multiplier: 'float'=1.0, **kwargs: Any) ->DetectionLayer:
    """Creates a detection layer module and the required loss function and target matching objects.

    Args:
        prior_shapes: A list of all the prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution.
        prior_shape_idxs: List of indices to ``prior_shapes`` that is used to select the (usually 3) prior shapes that
            this layer uses.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        num_classes: Number of different classes that this layer predicts.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
        input_is_normalized: The input is normalized by logistic activation in the previous layer. In this case the
            detection layer will not take the sigmoid of the coordinate and probability predictions, and the width and
            height are scaled up so that the maximum value is four times the anchor dimension. This is used by the
            Darknet configurations of Scaled-YOLOv4.

    """
    matching_func: 'Union[ShapeMatching, SimOTAMatching]'
    if matching_algorithm == 'simota':
        loss_func = YOLOLoss(overlap_func, None, None, overlap_loss_multiplier, confidence_loss_multiplier, class_loss_multiplier)
        matching_func = SimOTAMatching(prior_shapes, prior_shape_idxs, loss_func, spatial_range, size_range)
    elif matching_algorithm == 'size':
        if matching_threshold is None:
            raise ValueError('matching_threshold is required with size ratio matching.')
        matching_func = SizeRatioMatching(prior_shapes, prior_shape_idxs, matching_threshold, ignore_bg_threshold)
    elif matching_algorithm == 'iou':
        if matching_threshold is None:
            raise ValueError('matching_threshold is required with IoU threshold matching.')
        matching_func = IoUThresholdMatching(prior_shapes, prior_shape_idxs, matching_threshold, ignore_bg_threshold)
    elif matching_algorithm == 'maxiou' or matching_algorithm is None:
        matching_func = HighestIoUMatching(prior_shapes, prior_shape_idxs, ignore_bg_threshold)
    else:
        raise ValueError(f'Matching algorithm `{matching_algorithm}´ is unknown.')
    loss_func = YOLOLoss(overlap_func, predict_overlap, label_smoothing, overlap_loss_multiplier, confidence_loss_multiplier, class_loss_multiplier)
    layer_shapes = [prior_shapes[i] for i in prior_shape_idxs]
    return DetectionLayer(prior_shapes=layer_shapes, matching_func=matching_func, loss_func=loss_func, **kwargs)


def _create_yolo(config: 'CONFIG', num_inputs: 'List[int]', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, matching_algorithm: 'Optional[str]'=None, matching_threshold: 'Optional[float]'=None, spatial_range: 'float'=5.0, size_range: 'float'=4.0, ignore_bg_threshold: 'Optional[float]'=None, overlap_func: 'Optional[Union[str, Callable]]'=None, predict_overlap: 'Optional[float]'=None, label_smoothing: 'Optional[float]'=None, overlap_loss_multiplier: 'Optional[float]'=None, confidence_loss_multiplier: 'Optional[float]'=None, class_loss_multiplier: 'Optional[float]'=None, **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Creates a YOLO detection layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer. Not used by the detection layer.
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output (always 0 for a detection layer).

    """
    if prior_shapes is None:
        dims = config['anchors']
        prior_shapes = [(dims[i], dims[i + 1]) for i in range(0, len(dims), 2)]
    if ignore_bg_threshold is None:
        ignore_bg_threshold = config.get('ignore_thresh', 1.0)
        assert isinstance(ignore_bg_threshold, float)
    if overlap_func is None:
        overlap_func = config.get('iou_loss', 'iou')
        assert isinstance(overlap_func, str)
    if overlap_loss_multiplier is None:
        overlap_loss_multiplier = config.get('iou_normalizer', 1.0)
        assert isinstance(overlap_loss_multiplier, float)
    if confidence_loss_multiplier is None:
        confidence_loss_multiplier = config.get('obj_normalizer', 1.0)
        assert isinstance(confidence_loss_multiplier, float)
    if class_loss_multiplier is None:
        class_loss_multiplier = config.get('cls_normalizer', 1.0)
        assert isinstance(class_loss_multiplier, float)
    layer = create_detection_layer(num_classes=config['classes'], prior_shapes=prior_shapes, prior_shape_idxs=config['mask'], matching_algorithm=matching_algorithm, matching_threshold=matching_threshold, spatial_range=spatial_range, size_range=size_range, ignore_bg_threshold=ignore_bg_threshold, overlap_func=overlap_func, predict_overlap=predict_overlap, label_smoothing=label_smoothing, overlap_loss_multiplier=overlap_loss_multiplier, confidence_loss_multiplier=confidence_loss_multiplier, class_loss_multiplier=class_loss_multiplier, xy_scale=config.get('scale_x_y', 1.0), input_is_normalized=config.get('new_coords', 0) > 0)
    return layer, 0


def _create_layer(config: 'CONFIG', num_inputs: 'List[int]', **kwargs: Any) ->CREATE_LAYER_OUTPUT:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the
    layer config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    create_func: 'Dict[str, Callable[..., CREATE_LAYER_OUTPUT]]' = {'convolutional': _create_convolutional, 'maxpool': _create_maxpool, 'route': _create_route, 'shortcut': _create_shortcut, 'upsample': _create_upsample, 'yolo': _create_yolo}
    return create_func[config['type']](config, num_inputs, **kwargs)


@torch.jit.script
def get_image_size(images: 'Tensor') ->Tensor:
    """Get the image size from an input tensor.

    The function needs the ``@torch.jit.script`` decorator in order for ONNX generation to work. The tracing based
    generator will loose track of e.g. ``images.shape[1]`` and treat it as a Python variable and not a tensor. This will
    cause the dimension to be treated as a constant in the model, which prevents dynamic input sizes.

    Args:
        images: An image batch to take the width and height from.

    Returns:
        A tensor that contains the image width and height.

    """
    height = images.shape[2]
    width = images.shape[3]
    return torch.tensor([width, height], device=images.device)


class DarknetNetwork(nn.Module):
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    Iterates through the layers from the configuration and creates corresponding PyTorch modules. If ``weights_path`` is
    given and points to a Darknet model file, loads the convolutional layer weights from the file.

    Args:
        config_path: Path to a Darknet configuration file that defines the network architecture.
        weights_path: Path to a Darknet model file. If given, the model weights will be read from this file.
        in_channels: Number of channels in the input image.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    """

    def __init__(self, config_path: 'str', weights_path: 'Optional[str]'=None, in_channels: 'Optional[int]'=None, **kwargs: Any) ->None:
        super().__init__()
        with open(config_path) as config_file:
            sections = self._read_config(config_file)
        if len(sections) < 2:
            raise MisconfigurationException('The model configuration file should include at least two sections.')
        self.__dict__.update(sections[0])
        global_config = sections[0]
        layer_configs = sections[1:]
        if in_channels is None:
            in_channels = global_config.get('channels', 3)
            assert isinstance(in_channels, int)
        self.layers = nn.ModuleList()
        num_inputs = [in_channels]
        for layer_config in layer_configs:
            config = {**global_config, **layer_config}
            layer, num_outputs = _create_layer(config, num_inputs, **kwargs)
            self.layers.append(layer)
            num_inputs.append(num_outputs)
        if weights_path is not None:
            with open(weights_path) as weight_file:
                self.load_weights(weight_file)

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        outputs: 'List[Tensor]' = []
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        for layer in self.layers:
            if isinstance(layer, (RouteLayer, ShortcutLayer)):
                x = layer(outputs)
            elif isinstance(layer, DetectionLayer):
                x, preds = layer(x, image_size)
                detections.append(x)
                if targets is not None:
                    layer_losses, layer_hits = layer.calculate_losses(preds, targets, image_size)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = layer(x)
            outputs.append(x)
        return detections, losses, hits

    def load_weights(self, weight_file: 'io.IOBase') ->None:
        """Loads weights to layer modules from a pretrained Darknet model.

        One may want to continue training from pretrained weights, on a dataset with a different number of object
        categories. The number of kernels in the convolutional layers just before each detection layer depends on the
        number of output classes. The Darknet solution is to truncate the weight file and stop reading weights at the
        first incompatible layer. For this reason the function silently leaves the rest of the layers unchanged, when
        the weight file ends.

        Args:
            weight_file: A file-like object containing model weights in the Darknet binary format.

        """
        if not isinstance(weight_file, io.IOBase):
            raise ValueError('weight_file must be a file-like object.')
        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        rank_zero_info(f'Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} that has been trained on {images_seen[0]} images.')

        def read(tensor: 'Tensor') ->int:
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            Returns the number of elements read. If there's no more data in ``weight_file``, returns 0.
            """
            np_array = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            num_elements = np_array.size
            if num_elements > 0:
                source = torch.from_numpy(np_array).view_as(tensor)
                with torch.no_grad():
                    tensor.copy_(source)
            return num_elements
        for layer in self.layers:
            if not isinstance(layer, Conv):
                continue
            if isinstance(layer.norm, nn.Identity):
                assert layer.conv.bias is not None
                read(layer.conv.bias)
            else:
                assert isinstance(layer.norm, nn.BatchNorm2d)
                assert layer.norm.running_mean is not None
                assert layer.norm.running_var is not None
                read(layer.norm.bias)
                read(layer.norm.weight)
                read(layer.norm.running_mean)
                read(layer.norm.running_var)
            read_count = read(layer.conv.weight)
            if read_count == 0:
                return

    def _read_config(self, config_file: 'Iterable[str]') ->List[Dict[str, Any]]:
        """Reads a Darnet network configuration file and returns a list of configuration sections.

        Args:
            config_file: The configuration file to read.

        Returns:
            A list of configuration sections.

        """
        section_re = re.compile('\\[([^]]+)\\]')
        list_variables = 'layers', 'anchors', 'mask', 'scales'
        variable_types = {'activation': str, 'anchors': int, 'angle': float, 'batch': int, 'batch_normalize': bool, 'beta_nms': float, 'burn_in': int, 'channels': int, 'classes': int, 'cls_normalizer': float, 'decay': float, 'exposure': float, 'filters': int, 'from': int, 'groups': int, 'group_id': int, 'height': int, 'hue': float, 'ignore_thresh': float, 'iou_loss': str, 'iou_normalizer': float, 'iou_thresh': float, 'jitter': float, 'layers': int, 'learning_rate': float, 'mask': int, 'max_batches': int, 'max_delta': float, 'momentum': float, 'mosaic': bool, 'new_coords': int, 'nms_kind': str, 'num': int, 'obj_normalizer': float, 'pad': bool, 'policy': str, 'random': bool, 'resize': float, 'saturation': float, 'scales': float, 'scale_x_y': float, 'size': int, 'steps': str, 'stride': int, 'subdivisions': int, 'truth_thresh': float, 'width': int}
        section = None
        sections = []

        def convert(key: 'str', value: 'str') ->Union[str, int, float, List[Union[str, int, float]]]:
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn(f'Unknown YOLO configuration variable: {key}')
                return value
            if key in list_variables:
                return [variable_types[key](v) for v in value.split(',')]
            return variable_types[key](value)
        for line in config_file:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {'type': section_match.group(1)}
            else:
                if section is None:
                    raise RuntimeError('Darknet network configuration file does not start with a section header.')
                key, value = line.split('=')
                key = key.rstrip()
                value = value.lstrip()
                section[key] = convert(key, value)
        if section is not None:
            sections.append(section)
        return sections


class ReOrg(nn.Module):
    """Re-organizes the tensor so that every square region of four cells is placed into four different channels.

    The result is a tensor with half the width and height, and four times as many channels.

    """

    def forward(self, x: 'Tensor') ->Tensor:
        tl = x[..., ::2, ::2]
        bl = x[..., 1::2, ::2]
        tr = x[..., ::2, 1::2]
        br = x[..., 1::2, 1::2]
        return torch.cat((tl, bl, tr, br), dim=1)


class BottleneckBlock(nn.Module):
    """A residual block with a bottleneck layer.

    Args:
        in_channels: Number of input channels that the block expects.
        out_channels: Number of output channels that the block produces.
        hidden_channels: Number of output channels the (hidden) bottleneck layer produces. By default the number of
            output channels of the block.
        shortcut: Whether the block should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', hidden_channels: 'Optional[int]'=None, shortcut: 'bool'=True, activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm') ->None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.convs = nn.Sequential(Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm), Conv(hidden_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm))
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: 'Tensor') ->Tensor:
        y = self.convs(x)
        return x + y if self.shortcut else y


class TinyStage(nn.Module):
    """One stage of the "tiny" network architecture from YOLOv4.

    Args:
        num_channels: Number of channels in the input of the stage. Partial output will have as many channels and full
            output will have twice as many channels.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, num_channels: 'int', activation: 'Optional[str]'='leaky', norm: 'Optional[str]'='batchnorm') ->None:
        super().__init__()
        hidden_channels = num_channels // 2
        self.conv1 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.mix = Conv(num_channels, num_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: 'Tensor') ->Tuple[Tensor, Tensor]:
        partial = torch.chunk(x, 2, dim=1)[1]
        y1 = self.conv1(partial)
        y2 = self.conv2(y1)
        partial_output = self.mix(torch.cat((y2, y1), dim=1))
        full_output = torch.cat((x, partial_output), dim=1)
        return partial_output, full_output


class CSPStage(nn.Module):
    """One stage of a Cross Stage Partial Network (CSPNet).

    Encapsulates a number of bottleneck blocks in the "fusion first" CSP structure.

    `Chien-Yao Wang et al. <https://arxiv.org/abs/1911.11929>`_

    Args:
        in_channels: Number of input channels that the CSP stage expects.
        out_channels: Number of output channels that the CSP stage produces.
        depth: Number of bottleneck blocks that the CSP stage contains.
        shortcut: Whether the bottleneck blocks should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', depth: 'int'=1, shortcut: 'bool'=True, activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm') ->None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.split1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        bottlenecks: 'List[nn.Module]' = [BottleneckBlock(hidden_channels, hidden_channels, shortcut=shortcut, norm=norm, activation=activation) for _ in range(depth)]
        self.bottlenecks = nn.Sequential(*bottlenecks)
        self.mix = Conv(hidden_channels * 2, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: 'Tensor') ->Tensor:
        y1 = self.bottlenecks(self.split1(x))
        y2 = self.split2(x)
        return self.mix(torch.cat((y1, y2), dim=1))


class ELANStage(nn.Module):
    """One stage of an Efficient Layer Aggregation Network (ELAN).

    `Chien-Yao Wang et al. <https://arxiv.org/abs/2211.04800>`_

    Args:
        in_channels: Number of input channels that the ELAN stage expects.
        out_channels: Number of output channels that the ELAN stage produces.
        hidden_channels: Number of output channels that the computational blocks produce. The default value is half the
            number of output channels of the block, as in YOLOv7-W6, but the value varies between the variants.
        split_channels: Number of channels in each part after splitting the input to the cross stage connection and the
            computational blocks. The default value is the number of hidden channels, as in all YOLOv7 backbones. Most
            YOLOv7 heads use twice the number of hidden channels.
        depth: Number of computational blocks that the ELAN stage contains. The default value is 2. YOLOv7 backbones use
            2 to 4 blocks per stage.
        block_depth: Number of convolutional layers in one computational block. The default value is 2. YOLOv7 backbones
            have two convolutions per block. YOLOv7 heads (except YOLOv7-X) have 2 to 8 blocks with only one convolution
            in each.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', hidden_channels: 'Optional[int]'=None, split_channels: 'Optional[int]'=None, depth: 'int'=2, block_depth: 'int'=2, activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm') ->None:
        super().__init__()

        def conv3x3(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm)

        def block(in_channels: 'int', out_channels: 'int') ->nn.Module:
            convs = [conv3x3(in_channels, out_channels)]
            for _ in range(block_depth - 1):
                convs.append(conv3x3(out_channels, out_channels))
            return nn.Sequential(*convs)
        if hidden_channels is None:
            hidden_channels = out_channels // 2
        if split_channels is None:
            split_channels = hidden_channels
        self.split1 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        blocks = [block(split_channels, hidden_channels)]
        for _ in range(depth - 1):
            blocks.append(block(hidden_channels, hidden_channels))
        self.blocks = nn.ModuleList(blocks)
        total_channels = split_channels * 2 + hidden_channels * depth
        self.mix = Conv(total_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: 'Tensor') ->Tensor:
        outputs = [self.split1(x), self.split2(x)]
        x = outputs[-1]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return self.mix(torch.cat(outputs, dim=1))


class CSPSPP(nn.Module):
    """Spatial pyramid pooling module from the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm'):
        super().__init__()

        def conv(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1) ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, activation=activation, norm=norm)
        self.conv1 = nn.Sequential(conv(in_channels, out_channels), conv(out_channels, out_channels, kernel_size=3), conv(out_channels, out_channels))
        self.conv2 = conv(in_channels, out_channels)
        self.maxpool1 = MaxPool(kernel_size=5, stride=1)
        self.maxpool2 = MaxPool(kernel_size=9, stride=1)
        self.maxpool3 = MaxPool(kernel_size=13, stride=1)
        self.mix1 = nn.Sequential(conv(4 * out_channels, out_channels), conv(out_channels, out_channels, kernel_size=3))
        self.mix2 = Conv(2 * out_channels, out_channels)

    def forward(self, x: 'Tensor') ->Tensor:
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.maxpool2(x1)
        x4 = self.maxpool3(x1)
        y1 = self.mix1(torch.cat((x1, x2, x3, x4), dim=1))
        y2 = self.conv2(x)
        return self.mix2(torch.cat((y1, y2), dim=1))


class FastSPP(nn.Module):
    """Fast spatial pyramid pooling module from YOLOv5.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.maxpool = MaxPool(kernel_size=5, stride=1)
        self.mix = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: 'Tensor') ->Tensor:
        y1 = self.conv(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y4 = self.maxpool(y3)
        return self.mix(torch.cat((y1, y2, y3, y4), dim=1))


class YOLOV4TinyBackbone(nn.Module):
    """Backbone of the "tiny" network architecture from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int'=3, width: 'int'=32, activation: 'Optional[str]'='leaky', normalization: 'Optional[str]'='batchnorm'):
        super().__init__()

        def smooth(num_channels: 'int') ->nn.Module:
            return Conv(num_channels, num_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            conv_module = Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([('downsample', conv_module), ('smooth', smooth(out_channels))]))

        def maxpool(out_channels: 'int') ->nn.Module:
            return nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((0, 1, 0, 1))), ('maxpool', MaxPool(kernel_size=2, stride=2)), ('smooth', smooth(out_channels))]))

        def stage(out_channels: 'int', use_maxpool: 'bool') ->nn.Module:
            downsample_module = maxpool(out_channels) if use_maxpool else downsample(out_channels // 2, out_channels)
            stage_module = TinyStage(out_channels, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([('downsample', downsample_module), ('stage', stage_module)]))
        stages = [Conv(in_channels, width, kernel_size=3, stride=2, activation=activation, norm=normalization), stage(width * 2, False), stage(width * 4, True), stage(width * 8, True), maxpool(width * 16)]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: 'Tensor') ->List[Tensor]:
        c1 = self.stages[0](x)
        c2, x = self.stages[1](c1)
        c3, x = self.stages[2](x)
        c4, x = self.stages[3](x)
        c5 = self.stages[4](x)
        return [c1, c2, c3, c4, c5]


class YOLOV4Backbone(nn.Module):
    """A backbone that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Typically ``(32, 64, 128, 256, 512, 1024)``. The P6 variant
            adds one more stage with 1024 channels.
        depths: Number of bottleneck layers at each network stage. Typically ``(1, 1, 2, 8, 8, 4)``. The P6 variant uses
            ``(1, 1, 3, 15, 15, 7, 7)``.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int'=3, widths: 'Sequence[int]'=(32, 64, 128, 256, 512, 1024), depths: 'Sequence[int]'=(1, 1, 2, 8, 8, 4), activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm') ->None:
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError('Width and depth has to be given for an equal number of stages.')

        def conv3x3(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: 'int', out_channels: 'int', depth: 'int') ->nn.Module:
            csp = CSPStage(out_channels, out_channels, depth=depth, shortcut=True, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([('downsample', downsample(in_channels, out_channels)), ('csp', csp)]))
        convs = [conv3x3(in_channels, widths[0])] + [conv3x3(widths[0], widths[0]) for _ in range(depths[0] - 1)]
        self.stem = nn.Sequential(*convs)
        self.stages = nn.ModuleList(stage(in_channels, out_channels, depth) for in_channels, out_channels, depth in zip(widths[:-1], widths[1:], depths[1:]))

    def forward(self, x: 'Tensor') ->List[Tensor]:
        x = self.stem(x)
        outputs: 'List[Tensor]' = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class YOLOV5Backbone(nn.Module):
    """The Cross Stage Partial Network backbone from YOLOv5.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int'=3, width: 'int'=64, depth: 'int'=3, activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm') ->None:
        super().__init__()

        def downsample(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3) ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: 'int', out_channels: 'int', depth: 'int') ->nn.Module:
            csp = CSPStage(out_channels, out_channels, depth=depth, shortcut=True, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([('downsample', downsample(in_channels, out_channels)), ('csp', csp)]))
        stages = [downsample(in_channels, width, kernel_size=6), stage(width, width * 2, depth), stage(width * 2, width * 4, depth * 2), stage(width * 4, width * 8, depth * 3), stage(width * 8, width * 16, depth)]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: 'Tensor') ->List[Tensor]:
        c1 = self.stages[0](x)
        c2 = self.stages[1](c1)
        c3 = self.stages[2](c2)
        c4 = self.stages[3](c3)
        c5 = self.stages[4](c4)
        return [c1, c2, c3, c4, c5]


class YOLOV7Backbone(nn.Module):
    """A backbone that corresponds to the W6 variant of the Efficient Layer Aggregation Network from YOLOv7.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Before the first stage there will be one extra split of
            spatial resolution by a ``ReOrg`` layer, producing ``in_channels * 4`` channels.
        depth: Number of computational blocks at each network stage. YOLOv7-W6 backbone uses 2.
        block_depth: Number of convolutional layers in one computational block. YOLOv7-W6 backbone uses 2.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int'=3, widths: 'Sequence[int]'=(64, 128, 256, 512, 768, 1024), depth: 'int'=2, block_depth: 'int'=2, activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm') ->None:
        super().__init__()

        def conv3x3(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: 'int', out_channels: 'int') ->nn.Module:
            elan = ELANStage(out_channels, out_channels, depth=depth, block_depth=block_depth, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([('downsample', downsample(in_channels, out_channels)), ('elan', elan)]))
        self.stem = nn.Sequential(*[ReOrg(), conv3x3(in_channels * 4, widths[0])])
        self.stages = nn.ModuleList(stage(in_channels, out_channels) for in_channels, out_channels in zip(widths[:-1], widths[1:]))

    def forward(self, x: 'Tensor') ->List[Tensor]:
        x = self.stem(x)
        outputs: 'List[Tensor]' = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def run_detection(detection_layer: 'DetectionLayer', layer_input: 'Tensor', targets: 'Optional[TARGETS]', image_size: 'Tensor', detections: 'List[Tensor]', losses: 'List[Tensor]', hits: 'List[int]') ->None:
    """Runs the detection layer on the inputs and appends the output to the ``detections`` list.

    If ``targets`` is given, also calculates the losses and appends to the ``losses`` list.

    Args:
        detection_layer: The detection layer.
        layer_input: Input to the detection layer.
        targets: List of training targets for each image.
        image_size: Width and height in a vector that defines the scale of the target coordinates.
        detections: A list where a tensor containing the detections will be appended to.
        losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
        hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is given.
    """
    output, preds = detection_layer(layer_input, image_size)
    detections.append(output)
    if targets is not None:
        layer_losses, layer_hits = detection_layer.calculate_losses(preds, targets, image_size)
        losses.append(layer_losses)
        hits.append(layer_hits)


class YOLOV4TinyNetwork(nn.Module):
    """The "tiny" network architecture from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, width: 'int'=32, activation: 'Optional[str]'='leaky', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if prior_shapes is None:
            prior_shapes = [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 3.')
        num_outputs = (5 + num_classes) * anchors_per_cell

        def conv(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1) ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def upsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            return nn.Sequential(OrderedDict([('channels', channels), ('upsample', upsample)]))

        def outputs(in_channels: 'int') ->nn.Module:
            return nn.Conv2d(in_channels, num_outputs, kernel_size=1, stride=1, bias=True)

        def detect(prior_shape_idxs: 'Sequence[int]') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs)
        self.backbone = backbone or YOLOV4TinyBackbone(width=width, activation=activation, normalization=normalization)
        self.fpn5 = conv(width * 16, width * 8)
        self.out5 = nn.Sequential(OrderedDict([('channels', conv(width * 8, width * 16)), (f'outputs_{num_outputs}', outputs(width * 16))]))
        self.upsample5 = upsample(width * 8, width * 4)
        self.fpn4 = conv(width * 12, width * 8, kernel_size=3)
        self.out4 = nn.Sequential(OrderedDict([(f'outputs_{num_outputs}', outputs(width * 8))]))
        self.upsample4 = upsample(width * 8, width * 2)
        self.fpn3 = conv(width * 6, width * 4, kernel_size=3)
        self.out3 = nn.Sequential(OrderedDict([(f'outputs_{num_outputs}', outputs(width * 4))]))
        self.detect3 = detect([0, 1, 2])
        self.detect4 = detect([3, 4, 5])
        self.detect5 = detect([6, 7, 8])

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, c5 = self.backbone(x)[-3:]
        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample5(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), c3), dim=1)
        p3 = self.fpn3(x)
        run_detection(self.detect5, self.out5(p5), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(p4), targets, image_size, detections, losses, hits)
        run_detection(self.detect3, self.out3(p3), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4Network(nn.Module):
    """Network architecture that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, widths: 'Sequence[int]'=(32, 64, 128, 256, 512, 1024), activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if prior_shapes is None:
            prior_shapes = [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 3.')
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPStage(in_channels, out_channels, depth=2, shortcut=False, norm=normalization, activation=activation)

        def out(in_channels: 'int') ->nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([('conv', conv), (f'outputs_{num_outputs}', outputs)]))

        def upsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            return nn.Sequential(OrderedDict([('channels', channels), ('upsample', upsample)]))

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: 'Sequence[int]') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(widths=widths, activation=activation, normalization=normalization)
        w3 = widths[-3]
        w4 = widths[-2]
        w5 = widths[-1]
        self.spp = spp(w5, w5)
        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)
        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)
        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)
        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)
        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)
        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)
        x = torch.cat((self.upsample5(c5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), c5), dim=1)
        n5 = self.pan5(x)
        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4P6Network(nn.Module):
    """Network architecture that corresponds approximately to the variant of YOLOv4 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, widths: 'Sequence[int]'=(32, 64, 128, 256, 512, 1024, 1024), activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if prior_shapes is None:
            prior_shapes = [(13, 17), (31, 25), (24, 51), (61, 45), (61, 45), (48, 102), (119, 96), (97, 189), (97, 189), (217, 184), (171, 384), (324, 451), (324, 451), (545, 357), (616, 618), (1024, 1024)]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 4.')
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPStage(in_channels, out_channels, depth=2, shortcut=False, norm=normalization, activation=activation)

        def out(in_channels: 'int') ->nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([('conv', conv), (f'outputs_{num_outputs}', outputs)]))

        def upsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            return nn.Sequential(OrderedDict([('channels', channels), ('upsample', upsample)]))

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: 'Sequence[int]') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(widths=widths, depths=(1, 1, 3, 15, 15, 7, 7), activation=activation, normalization=normalization)
        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]
        self.spp = spp(w6, w6)
        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6, w5 // 2)
        self.fpn5 = csp(w5, w5)
        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)
        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)
        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)
        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)
        self.downsample5 = downsample(w5, w5)
        self.pan6 = csp(w5 + w6, w6)
        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)
        self.out6 = out(w6)
        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4))

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)
        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)
        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        run_detection(self.detect6, self.out6(n6), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV5Network(nn.Module):
    """The YOLOv5 network architecture. Different variants (n/s/m/l/x) can be achieved by adjusting the ``depth``
    and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, width: 'int'=64, depth: 'int'=3, activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if prior_shapes is None:
            prior_shapes = [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 3.')
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def out(in_channels: 'int') ->nn.Module:
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([(f'outputs_{num_outputs}', outputs)]))

        def csp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPStage(in_channels, out_channels, depth=depth, shortcut=False, norm=normalization, activation=activation)

        def detect(prior_shape_idxs: 'Sequence[int]') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs)
        self.backbone = backbone or YOLOV5Backbone(depth=depth, width=width, activation=activation, normalization=normalization)
        self.spp = spp(width * 16, width * 16)
        self.pan3 = csp(width * 8, width * 4)
        self.out3 = out(width * 4)
        self.fpn4 = nn.Sequential(OrderedDict([('csp', csp(width * 16, width * 8)), ('conv', conv(width * 8, width * 4))]))
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = out(width * 8)
        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = out(width * 16)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)
        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)
        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)
        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


def run_detection_with_aux_head(detection_layer: 'DetectionLayer', aux_detection_layer: 'DetectionLayer', layer_input: 'Tensor', aux_input: 'Tensor', targets: 'Optional[TARGETS]', image_size: 'Tensor', aux_weight: 'float', detections: 'List[Tensor]', losses: 'List[Tensor]', hits: 'List[int]') ->None:
    """Runs the detection layer on the inputs and appends the output to the ``detections`` list.

    If ``targets`` is given, also runs the auxiliary detection layer on the auxiliary inputs, calculates the losses, and
    appends the losses to the ``losses`` list.

    Args:
        detection_layer: The lead detection layer.
        aux_detection_layer: The auxiliary detection layer.
        layer_input: Input to the lead detection layer.
        aux_input: Input to the auxiliary detection layer.
        targets: List of training targets for each image.
        image_size: Width and height in a vector that defines the scale of the target coordinates.
        aux_weight: Weight of the auxiliary loss.
        detections: A list where a tensor containing the detections will be appended to.
        losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
        hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is given.
    """
    output, preds = detection_layer(layer_input, image_size)
    detections.append(output)
    if targets is not None:
        layer_losses, layer_hits = detection_layer.calculate_losses(preds, targets, image_size)
        losses.append(layer_losses)
        hits.append(layer_hits)
        _, aux_preds = aux_detection_layer(aux_input, image_size)
        layer_losses, layer_hits = aux_detection_layer.calculate_losses(preds, targets, image_size, loss_preds=aux_preds)
        losses.append(layer_losses * aux_weight)
        hits.append(layer_hits)


class YOLOV7Network(nn.Module):
    """Network architecture that corresponds to the W6 variant of YOLOv7 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        aux_weight: Weight for the loss from the auxiliary heads.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, widths: 'Sequence[int]'=(64, 128, 256, 512, 768, 1024), activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, aux_weight: 'float'=0.25, **kwargs: Any) ->None:
        super().__init__()
        self.aux_weight = aux_weight
        if prior_shapes is None:
            prior_shapes = [(13, 17), (31, 25), (24, 51), (61, 45), (61, 45), (48, 102), (119, 96), (97, 189), (97, 189), (217, 184), (171, 384), (324, 451), (324, 451), (545, 357), (616, 618), (1024, 1024)]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 4.')
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def elan(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return ELANStage(in_channels, out_channels, split_channels=out_channels, depth=4, block_depth=1, norm=normalization, activation=activation)

        def out(in_channels: 'int', hidden_channels: 'int') ->nn.Module:
            conv = Conv(in_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(hidden_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([('conv', conv), (f'outputs_{num_outputs}', outputs)]))

        def upsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            return nn.Sequential(OrderedDict([('channels', channels), ('upsample', upsample)]))

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: 'Sequence[int]', range: 'float') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, spatial_range=range, num_classes=num_classes, input_is_normalized=False, **kwargs)
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV7Backbone(widths=widths, depth=2, block_depth=2, activation=activation, normalization=normalization)
        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]
        self.spp = spp(w6, w6 // 2)
        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6 // 2, w5 // 2)
        self.fpn5 = elan(w5, w5 // 2)
        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5 // 2, w4 // 2)
        self.fpn4 = elan(w4, w4 // 2)
        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4 // 2, w3 // 2)
        self.fpn3 = elan(w3, w3 // 2)
        self.downsample3 = downsample(w3 // 2, w4 // 2)
        self.pan4 = elan(w4, w4 // 2)
        self.downsample4 = downsample(w4 // 2, w5 // 2)
        self.pan5 = elan(w5, w5 // 2)
        self.downsample5 = downsample(w5 // 2, w6 // 2)
        self.pan6 = elan(w6, w6 // 2)
        self.out3 = out(w3 // 2, w3)
        self.aux_out3 = out(w3 // 2, w3 + w3 // 4)
        self.out4 = out(w4 // 2, w4)
        self.aux_out4 = out(w4 // 2, w4 + w4 // 4)
        self.out5 = out(w5 // 2, w5)
        self.aux_out5 = out(w5 // 2, w5 + w5 // 4)
        self.out6 = out(w6 // 2, w6)
        self.aux_out6 = out(w6 // 2, w6 + w6 // 4)
        self.detect3 = detect(range(0, anchors_per_cell), 5.0)
        self.aux_detect3 = detect(range(0, anchors_per_cell), 3.0)
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2), 5.0)
        self.aux_detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2), 3.0)
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3), 5.0)
        self.aux_detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3), 3.0)
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4), 5.0)
        self.aux_detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4), 3.0)

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)
        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)
        run_detection_with_aux_head(self.detect3, self.aux_detect3, self.out3(n3), self.aux_out3(n3), targets, image_size, self.aux_weight, detections, losses, hits)
        run_detection_with_aux_head(self.detect4, self.aux_detect4, self.out4(n4), self.aux_out4(p4), targets, image_size, self.aux_weight, detections, losses, hits)
        run_detection_with_aux_head(self.detect5, self.aux_detect5, self.out5(n5), self.aux_out5(p5), targets, image_size, self.aux_weight, detections, losses, hits)
        run_detection_with_aux_head(self.detect6, self.aux_detect6, self.out6(n6), self.aux_out6(c6), targets, image_size, self.aux_weight, detections, losses, hits)
        return detections, losses, hits


class YOLOXHead(nn.Module):
    """A module that produces features for YOLO detection layer, decoupling the classification and localization
    features.

    Args:
        in_channels: Number of input channels that the module expects.
        hidden_channels: Number of output channels in the hidden layers.
        anchors_per_cell: Number of detections made at each spatial location of the feature map.
        num_classes: Number of different classes that this model predicts.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(self, in_channels: 'int', hidden_channels: 'int', anchors_per_cell: 'int', num_classes: 'int', activation: 'Optional[str]'='silu', norm: 'Optional[str]'='batchnorm') ->None:
        super().__init__()

        def conv(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1) ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=norm)

        def linear(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def features(num_channels: 'int') ->nn.Module:
            return nn.Sequential(conv(num_channels, num_channels, kernel_size=3), conv(num_channels, num_channels, kernel_size=3))

        def classprob(num_channels: 'int') ->nn.Module:
            num_outputs = anchors_per_cell * num_classes
            outputs = linear(num_channels, num_outputs)
            return nn.Sequential(OrderedDict([('convs', features(num_channels)), (f'outputs_{num_outputs}', outputs)]))
        self.stem = conv(in_channels, hidden_channels)
        self.feat = features(hidden_channels)
        self.box = linear(hidden_channels, anchors_per_cell * 4)
        self.confidence = linear(hidden_channels, anchors_per_cell)
        self.classprob = classprob(hidden_channels)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.stem(x)
        features = self.feat(x)
        box = self.box(features)
        confidence = self.confidence(features)
        classprob = self.classprob(x)
        return torch.cat((box, confidence, classprob), dim=1)


class YOLOXNetwork(nn.Module):
    """The YOLOX network architecture. Different variants (nano/tiny/s/m/l/x) can be achieved by adjusting the
    ``depth`` and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 24 (yolox-tiny),
            32 (yolox-s), 48 (yolox-m), and 64 (yolox-l).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolox-tiny, yolox-s), 2 (yolox-m), and 3 (yolox-l).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(self, num_classes: 'int', backbone: 'Optional[nn.Module]'=None, width: 'int'=64, depth: 'int'=3, activation: 'Optional[str]'='silu', normalization: 'Optional[str]'='batchnorm', prior_shapes: 'Optional[List[Tuple[int, int]]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if prior_shapes is None:
            prior_shapes = [(8, 8), (16, 16), (32, 32)]
            anchors_per_cell = 1
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError('The number of provided prior shapes needs to be divisible by 3.')

        def spp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1) ->nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: 'int', out_channels: 'int') ->nn.Module:
            return CSPStage(in_channels, out_channels, depth=depth, shortcut=False, norm=normalization, activation=activation)

        def head(in_channels: 'int', hidden_channels: 'int') ->YOLOXHead:
            return YOLOXHead(in_channels, hidden_channels, anchors_per_cell, num_classes, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: 'Sequence[int]') ->DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs)
        self.backbone = backbone or YOLOV5Backbone(depth=depth, width=width, activation=activation, normalization=normalization)
        self.spp = spp(width * 16, width * 16)
        self.pan3 = csp(width * 8, width * 4)
        self.out3 = head(width * 4, width * 4)
        self.fpn4 = nn.Sequential(OrderedDict([('csp', csp(width * 16, width * 8)), ('conv', conv(width * 8, width * 4))]))
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = head(width * 8, width * 4)
        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = head(width * 16, width * 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)
        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: 'Tensor', targets: 'Optional[TARGETS]'=None) ->NETWORK_OUTPUT:
        detections: 'List[Tensor]' = []
        losses: 'List[Tensor]' = []
        hits: 'List[int]' = []
        image_size = get_image_size(x)
        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)
        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)
        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class Discriminator(nn.Module):

    def __init__(self, img_shape, hidden_dim=1024) ->None:
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim: 'int', feature_maps: 'int', image_channels: 'int') ->None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.gen = nn.Sequential(self._make_gen_block(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0), self._make_gen_block(feature_maps * 8, feature_maps * 4), self._make_gen_block(feature_maps * 4, feature_maps * 2), self._make_gen_block(feature_maps * 2, feature_maps), self._make_gen_block(feature_maps, image_channels, last_block=True))

    @staticmethod
    def _make_gen_block(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=4, stride: 'int'=2, padding: 'int'=1, bias: 'bool'=False, last_block: 'bool'=False) ->nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        else:
            gen_block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.Tanh())
        return gen_block

    def forward(self, noise: 'Tensor') ->Tensor:
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):

    def __init__(self, feature_maps: 'int', image_channels: 'int') ->None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = nn.Sequential(self._make_disc_block(image_channels, feature_maps, batch_norm=False), self._make_disc_block(feature_maps, feature_maps * 2), self._make_disc_block(feature_maps * 2, feature_maps * 4), self._make_disc_block(feature_maps * 4, feature_maps * 8), self._make_disc_block(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True))

    @staticmethod
    def _make_disc_block(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=4, stride: 'int'=2, padding: 'int'=1, bias: 'bool'=False, batch_norm: 'bool'=True, last_block: 'bool'=False) ->nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(), nn.LeakyReLU(0.2, inplace=True))
        else:
            disc_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), nn.Sigmoid())
        return disc_block

    def forward(self, x: 'Tensor') ->Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)


class MLP(nn.Module):
    """MLP architecture used as projectors in online and target networks and predictors in the online network.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 2048.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4096.
        output_dim (int, optional): Output dimension. Defaults to 256.

    Note:
        Default values for input, hidden, and output dimensions are based on values used in BYOL.

    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=4096, output_dim: 'int'=256) ->None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim, bias=True))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.model(x)


class SiameseArm(nn.Module):
    """SiameseArm consolidates the encoder and projector networks of BYOL's symmetric architecture into a single class.

    Args:
        encoder (Union[str, nn.Module], optional): Online and target network encoder architecture.
            Defaults to "resnet50".
        encoder_out_dim (int, optional): Output dimension of encoder. Defaults to 2048.
        projector_hidden_dim (int, optional): Online and target network projector network hidden dimension.
            Defaults to 4096.
        projector_out_dim (int, optional): Online and target network projector network output dimension.
            Defaults to 256.

    """

    def __init__(self, encoder: 'Union[str, nn.Module]'='resnet50', encoder_out_dim: 'int'=2048, projector_hidden_dim: 'int'=4096, projector_out_dim: 'int'=256) ->None:
        super().__init__()
        if isinstance(encoder, str):
            self.encoder = torchvision_ssl_encoder(encoder)
        else:
            self.encoder = encoder
        self.projector = MLP(encoder_out_dim, projector_hidden_dim, projector_out_dim)

    def forward(self, x: 'Tensor') ->Tuple[Tensor, Tensor]:
        y = self.encoder(x)[0]
        z = self.projector(y)
        return y, z

    def encode(self, x: 'Tensor') ->Tensor:
        """Returns the encoded representation of a view. This method does not calculate the projection as in the forward
        method.

        Args:
            x (Tensor): sample to be encoded

        """
        return self.encoder(x)[0]


@torch.no_grad()
def concatenate_all(tensor: 'Tensor') ->Tensor:
    """Performs ``all_gather`` operation to concatenate the provided tensor from all devices.

    This function has no gradient.
    """
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_tensor, tensor.contiguous())
    return torch.cat(gathered_tensor, 0)


class RepresentationQueue(nn.Module):
    """The queue is implemented as list of representations and a pointer to the location where the next batch of
    representations will be overwritten."""

    def __init__(self, representation_size: 'int', queue_size: 'int'):
        super().__init__()
        self.representations: 'Tensor'
        self.register_buffer('representations', torch.randn(representation_size, queue_size))
        self.representations = nn.functional.normalize(self.representations, dim=0)
        self.pointer: 'Tensor'
        self.register_buffer('pointer', torch.zeros([], dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, x: 'Tensor') ->None:
        """Replaces representations in the queue, starting at the current queue pointer, and advances the pointer.

        Args:
            x: A mini-batch of representations. The queue size has to be a multiple of the total number of
                representations across all devices.

        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            x = concatenate_all(x)
        queue_size = self.representations.shape[1]
        batch_size = x.shape[0]
        if queue_size % batch_size != 0:
            raise ValueError(f'Queue size ({queue_size}) is not a multiple of the batch size ({batch_size}).')
        end = self.pointer + batch_size
        self.representations[:, int(self.pointer):int(end)] = x.T
        self.pointer = end % queue_size


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, num_prototypes) ->None:
        super().__init__()
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module('prototypes' + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, widen=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, normalize=False, output_dim=0, hidden_mlp=0, num_prototypes=0, eval_mode=False, first_conv=True, maxpool1=True) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        num_out_filters = width_per_group * widen
        if first_conv:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        if maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2norm = normalize
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(nn.Linear(num_out_filters * block.expansion, hidden_mlp), nn.BatchNorm1d(hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim))
        self.prototypes = None
        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.eval_mode:
            return x
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in inputs]), return_counts=True)[1], 0)
        start_idx, output = 0, None
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])
            if 'cuda' in str(self.conv1.weight.device):
                _out = self.forward_backbone(_out)
            else:
                _out = self.forward_backbone(_out)
            output = _out if start_idx == 0 else torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class SWAVLoss(nn.Module):

    def __init__(self, temperature: 'float', crops_for_assign: 'tuple', num_crops: 'tuple', sinkhorn_iterations: 'int', epsilon: 'float', gpus: 'int', num_nodes: 'int') ->None:
        """Implementation for SWAV loss function.

        Args:
            temperature:  loss temperature
            crops_for_assign: list of crop ids for computing assignment
            num_crops: number of global and local crops, ex: [2, 6]
            sinkhorn_iterations: iterations for sinkhorn normalization
            epsilon: epsilon val for swav assignments
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes:  num_nodes: number of nodes to train on

        """
        super().__init__()
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.softmax = nn.Softmax(dim=1)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.num_crops = num_crops
        self.gpus = gpus
        self.num_nodes = num_nodes
        if self.gpus * self.num_nodes > 1:
            self.assignment_fn = self.distributed_sinkhorn
        else:
            self.assignment_fn = self.sinkhorn

    def forward(self, output: 'torch.Tensor', embedding: 'torch.Tensor', prototype_weights: 'torch.Tensor', batch_size: 'int', queue: 'Optional[torch.Tensor]'=None, use_queue: 'bool'=False) ->Tuple[int, Optional[torch.Tensor], bool]:
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[batch_size * crop_id:batch_size * (crop_id + 1)]
                if queue is not None:
                    if use_queue or not torch.all(queue[i, -1, :] == 0):
                        use_queue = True
                        out = torch.cat((torch.mm(queue[i], prototype_weights.t()), out))
                    queue[i, batch_size:] = self.queue[i, :-batch_size].clone()
                    queue[i, :batch_size] = embedding[crop_id * batch_size:(crop_id + 1) * batch_size]
                q = torch.exp(out / self.epsilon).t()
                q = self.assignment_fn(q, self.sinkhorn_iterations)[-batch_size:]
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                p = self.softmax(output[batch_size * v:batch_size * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.num_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss, queue, use_queue

    def sinkhorn(self, q: 'torch.Tensor', num_iters: 'int') ->torch.Tensor:
        """Implementation of Sinkhorn clustering."""
        with torch.no_grad():
            sum_q = torch.sum(q)
            q /= sum_q
            dim_k, dim_b = q.shape
            if self.gpus > 0:
                r = torch.ones(dim_k) / dim_k
                c = torch.ones(dim_b) / dim_b
            else:
                r = torch.ones(dim_k) / dim_k
                c = torch.ones(dim_b) / dim_b
            for _ in range(num_iters):
                u = torch.sum(q, dim=1)
                q *= (r / u).unsqueeze(1)
                q *= (c / torch.sum(q, dim=0)).unsqueeze(0)
            return (q / torch.sum(q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, q: 'torch.Tensor', num_iters: 'int') ->torch.Tensor:
        """Implementation of Distributed Sinkhorn."""
        with torch.no_grad():
            sum_q = torch.sum(q)
            dist.all_reduce(sum_q)
            q /= sum_q
            if self.gpus > 0:
                r = torch.ones(q.shape[0]) / q.shape[0]
                c = torch.ones(q.shape[1]) / (self.gpus * q.shape[1])
            else:
                r = torch.ones(q.shape[0]) / q.shape[0]
                c = torch.ones(q.shape[1]) / (self.gpus * q.shape[1])
            curr_sum = torch.sum(q, dim=1)
            dist.all_reduce(curr_sum)
            for _ in range(num_iters):
                u = curr_sum
                q *= (r / u).unsqueeze(1)
                q *= (c / torch.sum(q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(q, dim=1)
                dist.all_reduce(curr_sum)
            return (q / torch.sum(q, dim=0, keepdim=True)).t().float()


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm => ReLU ] x 2."""

    def __init__(self, in_ch: 'int', out_ch: 'int') ->None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: 'int', out_ch: 'int') ->None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature map
    from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: 'int', out_ch: 'int', bilinear: 'bool'=False) ->None:
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, in_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        x1 = self.upsample(x1)
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Pytorch Lightning implementation of U-Net.

    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.

    """

    def __init__(self, num_classes: 'int', input_channels: 'int'=3, num_layers: 'int'=5, features_start: 'int'=64, bilinear: 'bool'=False) ->None:
        if num_layers < 1:
            raise ValueError(f'num_layers = {num_layers}, expected: num_layers > 0')
        super().__init__()
        self.num_layers = num_layers
        layers = [DoubleConv(input_channels, features_start)]
        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2
        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: 'Tensor') ->Tensor:
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class SubModule(nn.Module):

    def __init__(self, inp, out) ->None:
        super().__init__()
        self.sub_layer = nn.Linear(inp, out)

    def forward(self, *args, **kwargs):
        return self.sub_layer(*args, **kwargs)


class ModuleDataMonitorModel(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.layer1 = nn.Linear(12, 5)
        self.layer2 = SubModule(5, 2)

    def forward(self, x):
        x = x.flatten(1)
        self.layer1_input = x
        x = self.layer1(x)
        self.layer1_output = x
        x = torch.relu(x + 1)
        self.layer2_input = x
        x = self.layer2(x)
        self.layer2_output = x
        return torch.relu(x - 2)


class PyTorchModel(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.layer = nn.Linear(5, 2)

    def forward(self, *args):
        return args


class TemplateModel(nn.Module):

    def __init__(self, mix_data=False) ->None:
        """Base model for testing.

        The setting ``mix_data=True`` simulates a wrong implementation.

        """
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(10)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        x = x.view(10, -1).permute(1, 0).view(-1, 10) if self.mix_data else x.view(-1, 10)
        return self.linear(self.bn(x))


class MultipleInputModel(TemplateModel):
    """Base model for testing verification when forward accepts multiple arguments."""

    def __init__(self, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.input_array = torch.rand(10, 5, 2), torch.rand(10, 5, 2)

    def forward(self, x, y, some_kwarg=True):
        return super().forward(x) + super().forward(y)


class MultipleOutputModel(TemplateModel):
    """Base model for testing verification when forward has multiple outputs."""

    def forward(self, x):
        out = super().forward(x)
        return None, out, out, False


class DictInputDictOutputModel(TemplateModel):
    """Base model for testing verification when forward has a collection of outputs."""

    def __init__(self, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.input_array = {'w': 42, 'x': {'a': torch.rand(3, 5, 2)}, 'y': torch.rand(3, 1, 5, 2), 'z': torch.tensor(2)}

    def forward(self, y, x, z, w):
        out1 = super().forward(x['a'])
        out2 = super().forward(y)
        out3 = out1 + out2
        return {(1): out1, (2): out2, (3): [out1, out3]}


class BatchNormModel(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.batch_norm0 = nn.BatchNorm1d(2)
        self.batch_norm1 = nn.BatchNorm1d(3)
        self.instance_norm = nn.InstanceNorm1d(4)


class SchedulerTestNet(torch.nn.Module):
    """Adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleneckBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CSPSPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CSPStage,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DCGANDiscriminator,
     lambda: ([], {'feature_maps': 4, 'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (DCGANGenerator,
     lambda: ([], {'latent_dim': 4, 'feature_maps': 4, 'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Discriminator,
     lambda: ([], {'img_shape': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ELANStage,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FastSPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiPrototypes,
     lambda: ([], {'output_dim': 4, 'num_prototypes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultipleInputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10]), torch.rand([4, 10])], {})),
    (MultipleOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10])], {})),
    (PyTorchModel,
     lambda: ([], {}),
     lambda: ([], {})),
    (ReOrg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SchedulerTestNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (ShortcutLayer,
     lambda: ([], {'source_layer': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SubModule,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (TinyStage,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (YOLOV4Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (YOLOV4TinyBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (YOLOV5Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (YOLOV7Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (YOLOXHead,
     lambda: ([], {'in_channels': 4, 'hidden_channels': 4, 'anchors_per_cell': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

