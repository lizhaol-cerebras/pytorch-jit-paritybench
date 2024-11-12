
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


from torch.utils.data import DataLoader


import numpy as np


import torch


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


import matplotlib.patches as mpatches


import random


from typing import Optional


import torch.nn.functional as F


from torch import nn


from torch import Tensor


import logging


import torch.nn as nn


from typing import List


from torch.nn.modules.loss import _Loss


from functools import partial


import math


from typing import Union


from torch.autograd import Variable


from torchvision import models


from torch.nn import Module


from torch.nn import Conv2d


from torch.nn import Parameter


from torch.nn import Softmax


from collections import OrderedDict


from torch.nn import BatchNorm2d


import torch.utils.checkpoint as checkpoint


from torchvision.models import resnet


import time


from torchvision.transforms import Pad


from torchvision.transforms import ColorJitter


from torchvision.transforms import Resize


from torchvision.transforms import FiveCrop


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomRotation


from torchvision.transforms import RandomVerticalFlip


from typing import Any


from typing import Callable


from typing import Dict


import collections


from collections import defaultdict


import copy


import re


from torch.optim import Optimizer


from torchvision.transforms import RandomCrop


def balanced_binary_cross_entropy_with_logits(logits: 'Tensor', targets: 'Tensor', gamma: 'float'=1.0, ignore_index: 'Optional[int]'=None, reduction: 'str'='mean') ->Tensor:
    """
    Balanced binary cross entropy loss.

    Args:
        logits:
        targets: This loss function expects target values to be hard targets 0/1.
        gamma: Power factor for balancing weights
        ignore_index:
        reduction:

    Returns:
        Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
        shape of `logits` tensor.
    """
    pos_targets: 'Tensor' = targets.eq(1).sum()
    neg_targets: 'Tensor' = targets.eq(0).sum()
    num_targets = pos_targets + neg_targets
    pos_weight = torch.pow(neg_targets / (num_targets + 1e-07), gamma)
    neg_weight = 1.0 - pos_weight
    pos_term = pos_weight.pow(gamma) * targets * torch.nn.functional.logsigmoid(logits)
    neg_term = neg_weight.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-logits)
    loss = -(pos_term + neg_term)
    if ignore_index is not None:
        loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    return loss


class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced binary cross-entropy loss.

    https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
    """
    __constants__ = ['gamma', 'reduction', 'ignore_index']

    def __init__(self, gamma: 'float'=1.0, reduction='mean', ignore_index: 'Optional[int]'=None):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output: 'Tensor', target: 'Tensor') ->Tensor:
        return balanced_binary_cross_entropy_with_logits(output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction)


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_binary_search(activations: 'Tensor', t: 'float', num_iters: 'int') ->Tensor:
    """Compute normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    effective_dim = torch.sum((normalized_activations > -1.0 / (1.0 - t)).to(torch.int32), dim=-1, keepdim=True)
    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)
    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True)
        update = sum_probs < 1.0
        lower = torch.reshape(lower * update + (1.0 - update) * logt_partition, shape_partition)
        upper = torch.reshape(upper * (1.0 - update) + update * logt_partition, shape_partition)
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


def compute_normalization_fixed_point(activations: 'Tensor', t: 'float', num_iters: 'int') ->Tensor:
    """Return the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - t)
    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu
    return normalization_constants


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Compute normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction='mean'):
    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels
    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (1 - label_smoothing * num_classes / (num_classes - 1)) * labels_onehot + label_smoothing / (num_classes - 1)
    probabilities = tempered_softmax(activations, t2, num_iters)
    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) - labels_onehot * log_t(probabilities, t1) - labels_onehot.pow(2.0 - t1) / (2.0 - t1) + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim=-1)
    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


class BiTemperedLogisticLoss(nn.Module):
    """

    https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
    https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: 'float', t2: 'float', smoothing=0.0, ignore_index=None, reduction: 'str'='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super(BiTemperedLogisticLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: 'Tensor', targets: 'Tensor') ->Tensor:
        loss = bi_tempered_logistic_loss(predictions, targets, t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none')
        if self.ignore_index is not None:
            mask = ~targets.eq(self.ignore_index)
            loss *= mask
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryBiTemperedLogisticLoss(nn.Module):
    """
    Modification of BiTemperedLogisticLoss for binary classification case.
    It's signature matches nn.BCEWithLogitsLoss: Predictions and target tensors must have shape [B,1,...]

    References:
        https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
        https://arxiv.org/abs/1906.03361
    """

    def __init__(self, t1: 'float', t2: 'float', smoothing: 'float'=0.0, ignore_index: 'Optional[int]'=None, reduction: 'str'='mean'):
        """

        Args:
            t1:
            t2:
            smoothing:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: 'Tensor', targets: 'Tensor') ->Tensor:
        """
        Forward method of the loss function

        Args:
            predictions: [B,1,...]
            targets: [B,1,...]

        Returns:
            Zero-sized tensor with reduced loss if self.reduction is `sum` or `mean`; Otherwise returns loss of the
            shape of `predictions` tensor.
        """
        if predictions.size(1) != 1 or targets.size(1) != 1:
            raise ValueError('Channel dimension for predictions and targets must be equal to 1')
        loss = bi_tempered_logistic_loss(torch.cat([-predictions, predictions], dim=1).moveaxis(1, -1), torch.cat([1 - targets, targets], dim=1).moveaxis(1, -1), t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction='none').unsqueeze(dim=1)
        if self.ignore_index is not None:
            mask = targets.eq(self.ignore_index)
            loss = torch.masked_fill(loss, mask, 0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


BINARY_MODE = 'binary'


MULTICLASS_MODE = 'multiclass'


MULTILABEL_MODE = 'multilabel'


def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)
    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1
    return bin_labels, valid_mask


EPS = 1e-10


def get_region_proportion(x: 'torch.Tensor', valid_mask: 'torch.Tensor'=None) ->torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            x = torch.einsum('bcwh, bcwh->bcwh', x, valid_mask)
            cardinality = torch.einsum('bcwh->bc', valid_mask)
        else:
            x = torch.einsum('bcwh,bwh->bcwh', x, valid_mask)
            cardinality = torch.einsum('bwh->b', valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3]
    region_proportion = (torch.einsum('bcwh->bc', x) + EPS) / (cardinality + EPS)
    return region_proportion


logger = logging.getLogger(__name__)


class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """

    def __init__(self, mode: 'str'=MULTICLASS_MODE, alpha: 'float'=0.1, factor: 'float'=5.0, step_size: 'int'=0, max_alpha: 'float'=100.0, temp: 'float'=1.0, ignore_index: 'int'=255, background_index: 'int'=-1, weight: 'Optional[torch.Tensor]'=None) ->None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: 'torch.Tensor', labels: 'torch.Tensor'):
        if self.mode == MULTICLASS_MODE:
            loss = F.cross_entropy(inputs, labels, weight=self.weight, ignore_index=self.ignore_index)
        else:
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: 'int') ->None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            logger.info('CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}'.format(curr_alpha, self.alpha))

    def get_gt_proportion(self, mode: 'str', labels: 'torch.Tensor', target_shape, ignore_index: 'int'=255):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            bin_labels = labels
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: 'str', logits: 'torch.Tensor', temp: 'float'=1.0, valid_mask=None):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """

    def forward(self, inputs: 'torch.Tensor', labels: 'torch.Tensor'):
        loss_ce = self.cross_entropy(inputs, labels)
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()
        loss = loss_ce + self.alpha * loss_reg
        return loss


class CrossEntropyWithKL(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * KL(gt_region || prob_region)
    """

    def kl_div(self, p: 'torch.Tensor', q: 'torch.Tensor') ->torch.Tensor:
        x = p * torch.log(p / q)
        x = torch.einsum('ij->i', x)
        return x

    def forward(self, inputs: 'torch.Tensor', labels: 'torch.Tensor'):
        loss_ce = self.cross_entropy(inputs, labels)
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        if self.mode == BINARY_MODE:
            regularizer = (self.kl_div(gt_proportion, pred_proportion) + self.kl_div(1 - gt_proportion, 1 - pred_proportion)).mean()
        else:
            regularizer = self.kl_div(gt_proportion, pred_proportion).mean()
        loss = loss_ce + self.alpha * regularizer
        return loss


def soft_dice_score(output: 'torch.Tensor', target: 'torch.Tensor', smooth: 'float'=0.0, eps: 'float'=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) ->torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {'O', 'M', 'U', 'S'}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    raise ValueError('Unsupported input type' + str(type(x)))


class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(self, mode: 'str'='multiclass', classes: 'List[int]'=None, log_loss=False, from_logits=True, smooth: 'float'=0.0, ignore_index=None, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: 'Tensor', y_true: 'Tensor') ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot(y_true * mask, num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


def focal_loss_with_logits(output: 'torch.Tensor', target: 'torch.Tensor', gamma: 'float'=2.0, alpha: 'Optional[float]'=0.25, reduction: 'str'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None, eps: 'float'=1e-06, ignore_index=None) ->torch.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the models)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type_as(output)
    p = torch.sigmoid(output)
    ce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term = torch.masked_fill(focal_term, pt < reduced_threshold, 1)
    loss = focal_term * ce_loss
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if ignore_index is not None:
        ignore_mask = target.eq(ignore_index)
        loss = torch.masked_fill(loss, ignore_mask, 0)
        if normalized:
            focal_term = torch.masked_fill(focal_term, ignore_mask, 0)
    if normalized:
        norm_factor = focal_term.sum(dtype=torch.float32).clamp_min(eps)
        loss /= norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum(dtype=torch.float32)
    if reduction == 'batchwise_mean':
        loss = loss.sum(dim=0, dtype=torch.float32)
    return loss


class BinaryFocalLoss(_Loss):

    def __init__(self, alpha=0.5, gamma: 'float'=2.0, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strenght).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param threshold:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized, ignore_index=ignore_index)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem."""
        loss = self.focal_loss_fn(label_input, label_target)
        return loss


class FocalLoss(_Loss):

    def __init__(self, alpha=0.5, gamma=2, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized)

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index
        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]
            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]
            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss


class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).

    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: 'float'=1, gamma: 'float'=2, xent: 'float'=0.1, reduction='mean'):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        cosine_loss = F.cosine_embedding_loss(input, torch.nn.functional.one_hot(target, num_classes=input.size(-1)), torch.tensor([1], device=target.device), reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction='none')
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss


def soft_jaccard_score(output: 'torch.Tensor', target: 'torch.Tensor', smooth: 'float'=0.0, eps: 'float'=1e-07, dims=None) ->torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :param dims:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


class JaccardLoss(_Loss):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multi-class and multi-label cases.
    """

    def __init__(self, mode: 'str', classes: 'List[int]'=None, log_loss=False, from_logits=True, smooth=0, eps=1e-07):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: 'Tensor', y_true: 'Tensor') ->Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true = F.one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(dims) > 0
        loss *= mask.float()
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: 'nn.Module', second: 'nn.Module', first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


def _flatten_binary_scores(scores, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return scores, labels
    valid = labels != ignore_index
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore_index)) for log, lab in zip(logits, labels))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore_index))
    return loss


class BinaryLovaszLoss(_Loss):

    def __init__(self, per_image: 'bool'=False, ignore_index: 'Optional[Union[int, float]]'=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)
    probas = probas.contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _lovasz_softmax(probas, labels, classes='present', per_image=False, ignore_index=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore_index: void class labels
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore_index), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore_index), classes=classes)
    return loss


class LovaszLoss(_Loss):

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """
    __constants__ = ['weight', 'pos_weight', 'reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, weight=None, ignore_index: 'Optional[int]'=-100, reduction='mean', smooth_factor=None, pos_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)
        loss = F.binary_cross_entropy_with_logits(input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask: 'Tensor' = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def label_smoothed_nll_loss(lprobs: 'torch.Tensor', target: 'torch.Tensor', epsilon: 'float', ignore_index=None, reduction='mean', dim=-1) ->torch.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)
    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """
    __constants__ = ['reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, reduction: 'str'='mean', smooth_factor: 'float'=0.0, ignore_index: 'Optional[int]'=-100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(log_prob, target, epsilon=self.smooth_factor, ignore_index=self.ignore_index, reduction=self.reduction, dim=self.dim)


def soft_micro_f1(preds: 'Tensor', targets: 'Tensor', eps=1e-06) ->Tensor:
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        targets (Tensor): targets array of shape (Num Samples, Num Classes)
        preds (Tensor): probability matrix of shape (Num Samples, Num Classes)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    References:
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """
    tp = torch.sum(preds * targets, dim=0)
    fp = torch.sum(preds * (1 - targets), dim=0)
    fn = torch.sum((1 - preds) * targets, dim=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - soft_f1
    return loss.mean()


class BinarySoftF1Loss(nn.Module):

    def __init__(self, ignore_index: 'Optional[int]'=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: 'Tensor', targets: 'Tensor') ->Tensor:
        targets = targets.view(-1)
        preds = preds.view(-1)
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        preds = preds.sigmoid().clamp(self.eps, 1 - self.eps)
        return soft_micro_f1(preds.view(-1, 1), targets.view(-1, 1))


class SoftF1Loss(nn.Module):

    def __init__(self, ignore_index: 'Optional[int]'=None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: 'Tensor', targets: 'Tensor') ->Tensor:
        preds = preds.softmax(dim=1).clamp(self.eps, 1 - self.eps)
        targets = torch.nn.functional.one_hot(targets, preds.size(1))
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)
        return soft_micro_f1(preds, targets)


class EdgeLoss(nn.Module):

    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index), DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0
        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)
        return edge_loss

    def forward(self, logits, targets):
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (self.edge_factor + 1)
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index), DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)
        return loss


class WingLoss(_Loss):

    def __init__(self, width=5, curvature=0.5, reduction='mean'):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(prediction, target, self.width, self.curvature, self.reduction)


class ConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)
        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionAggregationModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5, s4, s3, s2):
        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out


class Conv3x3GNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False), nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):

    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):

    def __init__(self, band, class_num=6, encoder_channels=[512, 256, 128, 64], pyramid_channels=256, segmentation_channels=128, dropout=0.2):
        super().__init__()
        self.name = 'FPN'
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])
        self.layer_down2 = self.base_layers[5]
        self.layer_down3 = self.base_layers[6]
        self.layer_down4 = self.base_layers[7]
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels, class_num, kernel_size=1, padding=0)

    def forward(self, x):
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)
        x = s5 + s4 + s3 + s2
        x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x


class A2FPN(nn.Module):

    def __init__(self, band=3, class_num=6, encoder_channels=[512, 256, 128, 64], pyramid_channels=64, segmentation_channels=64, dropout=0.2):
        super().__init__()
        self.name = 'A2FPN'
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])
        self.layer_down2 = self.base_layers[5]
        self.layer_down3 = self.base_layers[6]
        self.layer_down4 = self.base_layers[7]
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)
        self.attention = AttentionAggregationModule(segmentation_channels * 4, segmentation_channels * 4)
        self.final_conv = nn.Conv2d(segmentation_channels * 4, class_num, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x):
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)
        out = self.dropout(self.attention(s5, s4, s3, s2))
        out = self.final_conv(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return out


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2), norm_layer(out_channels), nn.ReLU6())


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)


class Output(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(Output, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionEnhancementModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(AttentionEnhancementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = Attention(out_chan)
        self.bn_atten = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        att = self.conv_atten(feat)
        return self.bn_atten(att)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):

    def __init__(self, pretrained=True, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = timm.create_model('swsl_resnet18', features_only=True, output_stride=32, out_indices=(2, 3, 4), pretrained=pretrained)
        self.arm16 = AttentionEnhancementModule(256, 128)
        self.arm32 = AttentionEnhancementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.0)
        self.up16 = nn.Upsample(scale_factor=2.0)
        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)
        return feat16_up, feat32_up

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


def l2_norm(x):
    return torch.einsum('bcn, bn->bcn', x, 1 / torch.norm(x, p=2, dim=-2))


class LinearAttention(Module):

    def __init__(self, in_places, scale=8, eps=1e-06):
        super(LinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        tailor_sum = 1 / (width * height + torch.einsum('bnc, bc->bn', Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum('bcn->bc', V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum('bnm, bmc->bcn', Q, matrix)
        weight_value = torch.einsum('bcn, bn->bcn', matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return x + (self.gamma * weight_value).contiguous()


class FeatureAggregationModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(FeatureAggregationModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = LinearAttention(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ABCNet(nn.Module):

    def __init__(self, band=3, n_classes=8, pretrained=True):
        super(ABCNet, self).__init__()
        self.name = 'ABCNet'
        self.cp = ContextPath(pretrained)
        self.sp = SpatialPath()
        self.fam = FeatureAggregationModule(256, 256)
        self.conv_out = Output(256, 256, n_classes, up_factor=8)
        if self.training:
            self.conv_out16 = Output(128, 64, n_classes, up_factor=8)
            self.conv_out32 = Output(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        if self.training:
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureAggregationModule, Output)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2))


class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2), norm_layer(out_channels))


class SeparableConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2, groups=in_channels, bias=False), norm_layer(out_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))


class GlobalLocalAttention(nn.Module):

    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size
        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        dots = q @ k.transpose(-2, -1) * self.scale
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)
        attn = dots.softmax(dim=-1)
        attn = attn @ v
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class Block(nn.Module):

    def __init__(self, dim=256, num_heads=16, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class BasicStem(nn.Module):

    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class ResT(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], norm_layer=nn.LayerNorm, apply_transform=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform
        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)
        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform) for i in range(self.depths[0])])
        cur += depths[0]
        self.stage2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform) for i in range(self.depths[1])])
        cur += depths[1]
        self.stage3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform) for i in range(self.depths[2])])
        cur += depths[2]
        self.stage4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform) for i in range(self.depths[3])])
        self.norm = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x3 = x
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x4 = x
        return x3, x4


class Attention_Embedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Attention_Embedding, self).__init__()
        self.attention = LinearAttention(in_channels)
        self.conv_attn = ConvBNReLU(in_channels, out_channels)
        self.up = UpSample(out_channels)

    def forward(self, high_feat, low_feat):
        A = self.attention(high_feat)
        A = self.conv_attn(A)
        A = self.up(A)
        output = low_feat * A
        output += low_feat
        return output


class TexturePath(nn.Module):

    def __init__(self):
        super(TexturePath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', **kwargs):
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if k in model_dict}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


class DependencyPath(nn.Module):

    def __init__(self, weight_path='pretrain_weights/rest_lite.pth'):
        super(DependencyPath, self).__init__()
        self.ResT = rest_lite(weight_path=weight_path)
        self.AE = Attention_Embedding(512, 256)
        self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2.0)

    def forward(self, x):
        e3, e4 = self.ResT(x)
        e = self.conv_avg(self.AE(e4, e3))
        return self.up(e)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class DependencyPathRes(nn.Module):

    def __init__(self):
        super(DependencyPathRes, self).__init__()
        resnet = models.resnet18(True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.AE = Attention_Embedding(512, 256)
        self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2.0)

    def forward(self, x):
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e = self.conv_avg(self.AE(e4, e3))
        return self.up(e)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BANet(nn.Module):

    def __init__(self, num_classes=6, weight_path='pretrain_weights/rest_lite.pth'):
        super(BANet, self).__init__()
        self.name = 'BANet'
        self.cp = DependencyPath(weight_path=weight_path)
        self.sp = TexturePath()
        self.fam = FeatureAggregationModule(256, 256)
        self.conv_out = Output(256, 256, num_classes, up_factor=8)
        self.init_weight()

    def forward(self, x):
        feat = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat)
        feat_out = self.conv_out(feat_fuse)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (Attention_Embedding, Output)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class MaxPoolLayer(nn.Sequential):

    def __init__(self, kernel_size=3, dilation=1, stride=1):
        super(MaxPoolLayer, self).__init__(nn.MaxPool2d(kernel_size=kernel_size, dilation=dilation, stride=stride, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2))


class AvgPoolLayer(nn.Sequential):

    def __init__(self, kernel_size=3, stride=1):
        super(AvgPoolLayer, self).__init__(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2))


class SeparableConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2, groups=in_channels, bias=False), norm_layer(out_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.ReLU6())


class SeparableConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=(stride - 1 + dilation * (kernel_size - 1)) // 2, groups=in_channels, bias=False), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))


class TransposeConvBNReLu(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBNReLu, self).__init__(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride), norm_layer(out_channels), nn.ReLU())


class TransposeConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBN, self).__init__(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride), norm_layer(out_channels))


class TransposeConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(TransposeConv, self).__init__(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))


class PyramidPool(nn.Sequential):

    def __init__(self, in_channels, out_channels, pool_size=1, norm_layer=nn.BatchNorm2d):
        super(PyramidPool, self).__init__(nn.AdaptiveAvgPool2d(pool_size), nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

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
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
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
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
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
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class SharedSpatialAttention(nn.Module):

    def __init__(self, in_places, eps=1e-06):
        super(SharedSpatialAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        tailor_sum = 1 / (width * height + torch.einsum('bnc, bc->bn', Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum('bcn->bc', V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum('bnm, bmc->bcn', Q, matrix)
        weight_value = torch.einsum('bcn, bn->bcn', matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


class SharedChannelAttention(nn.Module):

    def __init__(self, eps=1e-06):
        super(SharedChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = x.view(batch_size, chnnels, -1)
        K = x.view(batch_size, chnnels, -1)
        V = x.view(batch_size, chnnels, -1)
        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)
        tailor_sum = 1 / (width * height + torch.einsum('bnc, bn->bc', K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum('bcn->bn', V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum('bcm, bmn->bcn', matrix, Q)
        weight_value = torch.einsum('bcn, bc->bcn', matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


class DownConnection(nn.Module):

    def __init__(self, inplanes, planes, stride=2):
        super(DownConnection, self).__init__()
        self.convbn1 = ConvBN(inplanes, planes, kernel_size=3, stride=1)
        self.convbn2 = ConvBN(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ConvBN(inplanes, planes, stride=stride)

    def forward(self, x):
        residual = x
        x = self.convbn1(x)
        x = self.relu(x)
        x = self.convbn2(x)
        x = x + self.downsample(residual)
        x = self.relu(x)
        return x


class DCFAM(nn.Module):

    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12)):
        super(DCFAM, self).__init__()
        rate_1, rate_2 = tuple(atrous_rates)
        self.conv4 = Conv(encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.conv1 = Conv(encoder_channels[0], encoder_channels[0], kernel_size=1)
        self.lf4 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-1], encoder_channels[-2], dilation=rate_1), nn.UpsamplingNearest2d(scale_factor=2), SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_2), nn.UpsamplingNearest2d(scale_factor=2))
        self.lf3 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_1), nn.UpsamplingNearest2d(scale_factor=2), SeparableConvBNReLU(encoder_channels[-3], encoder_channels[-4], dilation=rate_2), nn.UpsamplingNearest2d(scale_factor=2))
        self.ca = SharedChannelAttention()
        self.pa = SharedSpatialAttention(in_places=encoder_channels[2])
        self.down12 = DownConnection(encoder_channels[0], encoder_channels[1])
        self.down231 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down232 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down34 = DownConnection(encoder_channels[2], encoder_channels[3])

    def forward(self, x1, x2, x3, x4):
        out4 = self.conv4(x4) + self.down34(self.pa(self.down232(x2)))
        out3 = self.pa(x3) + self.down231(self.ca(self.down12(x1)))
        out2 = self.ca(x2) + self.lf4(out4)
        del out4
        out1 = self.lf3(out3) + self.conv1(x1)
        del out3
        return out1, out2


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class FeatureRefinementHead(nn.Module):

    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-08
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels), nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv(decode_channels, decode_channels // 16, kernel_size=1), nn.ReLU6(), Conv(decode_channels // 16, decode_channels, kernel_size=1), nn.Sigmoid())
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


class WF(nn.Module):

    def __init__(self, in_channels=128, decode_channels=128, eps=1e-08):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoder_channels=(64, 128, 256, 512), decode_channels=64, dropout=0.1, window_size=8, num_classes=6):
        super(Decoder, self).__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels), nn.Dropout2d(p=dropout, inplace=True), Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)
            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)
            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)
            x = self.p2(x, res2)
            x = self.b2(x)
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DCSwin(nn.Module):

    def __init__(self, encoder_channels=(96, 192, 384, 768), dropout=0.05, atrous_rates=(6, 12), num_classes=6, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), frozen_stages=2):
        super(DCSwin, self).__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=frozen_stages)
        self.decoder = Decoder(encoder_channels, dropout, atrous_rates, num_classes)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x = self.decoder(x1, x2, x3, x4)
        return x


class Mlp_decoder(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FTUNetFormer(nn.Module):

    def __init__(self, decode_channels=256, dropout=0.2, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), freeze_stages=-1, window_size=8, num_classes=6):
        super().__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=freeze_stages)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


class PAM_Module(Module):

    def __init__(self, in_places, scale=8, eps=1e-06):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)
        KV = torch.einsum('bmn, bcn->bmc', K, V)
        norm = 1 / torch.einsum('bnc, bc->bn', Q, torch.sum(K, dim=-1) + self.eps)
        weight_value = torch.einsum('bnm, bmc, bn->bcn', Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):

    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)
        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):

    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)


nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class MANet(nn.Module):

    def __init__(self, num_channels=3, num_classes=5, backbone_name='resnet50', pretrained=True):
        super(MANet, self).__init__()
        self.name = 'MANet'
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, out_indices=(1, 2, 3, 4), pretrained=pretrained)
        filters = self.backbone.feature_info.channels()
        self.attention4 = PAM_CAM_Layer(filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2])
        self.attention2 = PAM_CAM_Layer(filters[1])
        self.attention1 = PAM_CAM_Layer(filters[0])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        e1, e2, e3, e4 = self.backbone(x)
        e4 = self.attention4(e4)
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class MambaLayer(nn.Module):

    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(ConvBNReLU(in_chs, dim, kernel_size=1), nn.AdaptiveAvgPool2d(1)))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), ConvBNReLU(in_chs, dim, kernel_size=1)))
        self.mamba = Mamba(d_model=dim * self.pool_len + in_chs, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvFFN(nn.Module):

    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.0):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientPyramidMamba(nn.Module):

    def __init__(self, backbone_name='swsl_resnet18', pretrained=True, num_classes=6, decoder_channels=128, last_feat_size=16):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, out_indices=(1, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x = self.decoder(x0, x3)
        return x


class PyramidMamba(nn.Module):

    def __init__(self, backbone_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True, num_classes=6, decoder_channels=128, last_feat_size=32, img_size=1024):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, img_size=img_size, out_indices=(-4, -1), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x0 = x0.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x = self.decoder(x0, x3)
        return x


class UNetFormer(nn.Module):

    def __init__(self, decode_channels=64, dropout=0.1, backbone_name='swsl_resnet18', pretrained=True, window_size=8, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AvgPoolLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BalancedBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicStem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BiTemperedLogisticLoss,
     lambda: ([], {'t1': 4, 't2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BinaryFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BinaryLovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BinarySoftF1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CAM_Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBnRelu,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'ksize': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvFFN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (DecoderBlock,
     lambda: ([], {'in_channels': 4, 'n_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownConnection,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FPN,
     lambda: ([], {'band': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FPNBlock,
     lambda: ([], {'pyramid_channels': 4, 'skip_channels': 4}),
     lambda: ([(torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16]))], {})),
    (GL,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaxPoolLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp_decoder,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PA,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PyramidPool,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableConvBNReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SharedChannelAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftBCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {})),
    (TransposeConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransposeConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransposeConvBNReLu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UpSample,
     lambda: ([], {'n_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

