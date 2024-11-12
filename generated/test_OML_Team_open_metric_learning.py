
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


from typing import Any


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from typing import Callable


from typing import List


import numpy as np


import pandas as pd


import torch


from torch import FloatTensor


from torch import BoolTensor


from torch import LongTensor


from functools import lru_cache


import torchvision


from torch import Tensor


import inspect


import logging


import warnings


from torch.utils.data import BatchSampler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


from torch.utils.data import Sampler


import itertools


from torch.distributed import all_gather_object


from torch.distributed import get_rank


from torch.distributed import get_world_size


import torch.nn.functional as F


from torch import sigmoid


from collections import defaultdict


from torch import isin


from torch import nn


from torch.nn import Module


from abc import ABC


from abc import abstractmethod


from typing import Collection


import matplotlib.pyplot as plt


from collections import Counter


from math import ceil


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim import Optimizer


from torch import device as tdevice


from torch.nn import functional as F


from torchvision.ops import MLP


from copy import deepcopy


from typing import Iterable


from random import choice


from torch import abs


from torch import arange


from torch import cartesian_prod


from torch import cat


from torch import clip


from torch import combinations


from torch import flip


from torch import no_grad


from torch import randint


from torch import randperm


from torch import tensor


from torch import unique


from torch import zeros


from itertools import combinations


from itertools import product


from random import sample


from torch.nn.modules.activation import Sigmoid


from torchvision.models import resnet18


from torchvision.models import resnet34


from torchvision.models import resnet50


from torchvision.models import resnet101


from torchvision.models import resnet152


from typing import OrderedDict


from collections import OrderedDict


from torchvision.models.resnet import resnet50


import math


from functools import partial


import torch.nn as nn


from enum import Enum


from torch.nn.init import trunc_normal_


from torch.utils.checkpoint import checkpoint


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import InterpolationMode


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from warnings import warn


import torch.optim as opt


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import CyclicLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import MultiplicativeLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import OneCycleLR


from torch.optim.lr_scheduler import StepLR


from torch import concat


from torch import finfo


import random


from typing import Generator


import matplotlib


from collections.abc import MutableMapping


from typing import Hashable


from typing import Iterator


from typing import Type


from torch import cdist


from torch.optim import Adam


from math import isclose


from random import randint


from random import shuffle


from itertools import chain


from torch.distributed import destroy_process_group


from torch.distributed import init_process_group


from torch.multiprocessing import spawn


from torch.optim import SGD


from torch.nn import TripletMarginLoss


from scipy.spatial.distance import squareform


from torch.utils.data import default_collate


from random import random


from torch.utils.data import SequentialSampler


class ITripletLossWithMiner(Module):
    """
    Base class for TripletLoss combined with Miner.

    """

    def forward(self, features: 'Tensor', labels: 'Union[Tensor, List[int]]') ->Tensor:
        """
        Args:
            features: Features with the shape ``[batch_size, features_dim]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        raise NotImplementedError()


class IExtractor(nn.Module, ABC):
    """
    Models have to inherit this interface to be comparable with the rest of the library.
    """
    pretrained_models: 'Dict[str, Any]' = {}

    def extract(self, x: 'Tensor') ->Tensor:
        return self.forward(x)

    @property
    def feat_dim(self) ->int:
        """
        The only method that obligatory to implemented.
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, weights: 'str', **kwargs) ->'IExtractor':
        """
        This method allows to download a pretrained checkpoint.
        The class field ``self.pretrained_models`` is the dictionary which keeps records of all the available
        checkpoints in the format, depending on implementation of a particular child of ``IExtractor``.
        As a user, you don't need to worry about implementing this method.

        Args:
            weights: A unique identifier (key) of a pretrained model information stored in
              a class field ``self.pretrained_models``.

        Returns: An instance of ``IExtractor``

        """
        if weights not in cls.pretrained_models:
            raise KeyError(f'There is no pretrained model {weights}. The existing ones are {list(cls.pretrained_models.keys())}.')
        extractor = cls(weights=weights, **cls.pretrained_models[weights]['init_args'], **kwargs)
        return extractor


class IPairwiseModel(nn.Module):
    """
    A model of this type takes two inputs, for example, two embeddings or two images.

    """

    def forward(self, x1: 'Any', x2: 'Any') ->Tensor:
        """

        Args:
            x1: The first input.
            x2: The second input.

        """
        raise NotImplementedError()

    def predict(self, x1: 'Any', x2: 'Any') ->Tensor:
        """
        While ``self.forward()`` is called during training, this method is called during
        inference or validation time. For example, it allows application of some activation,
        which was a part of a loss function during the training.

        Args:
            x1: The first input.
            x2: The second input.

        """
        raise NotImplementedError()


@torch.no_grad()
def label_smoothing(y: 'torch.Tensor', num_classes: 'int', epsilon: 'float'=0.2, categories: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    """
    This function is doing `label smoothing <https://arxiv.org/pdf/1512.00567v3.pdf>`_.
    You can also use modified version, where the label is smoothed only for the category corresponding to sample's
    ground truth label. To use this, you should provide the ``categories`` argument: vector, for which i-th entry
    is a corresponding category for label ``i``.

    Args:
        y: Ground truth labels with the size of batch_size where each element is from 0 (inclusive) to
            num_classes (exclusive).
        num_classes: Number of classes in total
        epsilon: Power of smoothing. The biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes`` after the transformation
        categories: Vector for which i-th entry is a corresponding category for label ``i``. Optional, used for
            category-based label smoothing. In that case the biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes_of_the_same_category``, labels outside of the category will not change
    """
    assert epsilon < 1, '`epsilon` must be less than 1.'
    ohe = F.one_hot(y, num_classes).float()
    if categories is not None:
        ohe *= 1 - epsilon
        same_category_mask = categories[y].tile(num_classes, 1).t() == categories
        return torch.where(same_category_mask, epsilon / same_category_mask.sum(-1).view(-1, 1), 0) + ohe
    else:
        ohe *= 1 - epsilon
        ohe += epsilon / num_classes
        return ohe


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss from `paper <https://arxiv.org/abs/1801.07698>`_ with possibility to use label smoothing.
    It contains projection size of ``num_features x num_classes`` inside itself. Please make sure that class labels
    started with 0 and ended as ``num_classes`` - 1.
    """
    criterion_name = 'arcface'

    def __init__(self, in_features: 'int', num_classes: 'int', m: 'float'=0.5, s: 'float'=64, smoothing_epsilon: 'float'=0, label2category: 'Optional[Dict[Any, Any]]'=None, reduction: 'str'='mean'):
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
            smoothing_epsilon: Label smoothing effect strength
            label2category: Optional, mapping from label to its category. If provided, label smoothing will redistribute
                 ``smoothing_epsilon`` only inside the category corresponding to the sample's ground truth label
            reduction: CrossEntropyLoss reduction
        """
        super(ArcFaceLoss, self).__init__()
        assert smoothing_epsilon is None or 0 <= smoothing_epsilon < 1, f'Choose another smoothing_epsilon parametrization, got {smoothing_epsilon}'
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.num_classes = num_classes
        if label2category is not None:
            mapper = {l: i for i, l in enumerate(sorted(list(set(label2category.values()))))}
            label2category = {k: mapper[v] for k, v in label2category.items()}
            self.label2category = torch.arange(num_classes).apply_(label2category.get)
        else:
            self.label2category = None
        self.smoothing_epsilon = smoothing_epsilon
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.rescale = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = -self.cos_m
        self.mm = self.sin_m * m
        self._last_logs: 'Dict[str, float]' = {}

    def fc(self, x: 'torch.Tensor') ->torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def smooth_labels(self, y: 'torch.Tensor') ->torch.Tensor:
        if self.label2category is not None:
            self.label2category = self.label2category
        return label_smoothing(y, self.num_classes, self.smoothing_epsilon, self.label2category)

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        assert torch.all(y < self.num_classes), 'You should provide labels between 0 and num_classes - 1.'
        cos = self.fc(x)
        self._log_accuracy_on_batch(cos, y)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)
        ohe = F.one_hot(y, self.num_classes)
        logit = torch.where(ohe.bool(), cos_w_margin, cos) * self.rescale
        if self.smoothing_epsilon:
            y = self.smooth_labels(y)
        return self.criterion(logit, y)

    @torch.no_grad()
    def _log_accuracy_on_batch(self, logits: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        self._last_logs['accuracy'] = torch.mean(y == torch.argmax(logits, 1))

    @property
    def last_logs(self) ->Dict[str, Any]:
        """
        Returns:
            Dictionary containing useful statistic calculated for the last batch.
        """
        return self._last_logs


class ArcFaceLossWithMLP(nn.Module):
    """
    Almost the same as ``ArcFaceLoss``, but also has MLP projector before the loss.
    You may want to use ``ArcFaceLossWithMLP`` to boost the expressive power of ArcFace loss during the training
    (for example, in a multi-head setup it may be a good idea to have task-specific projectors in each of the losses).
    Note, the criterion does not exist during the validation time.
    Thus, if you want to keep your MLP layers, you should create them as a part of the model you train.
    """

    def __init__(self, in_features: 'int', num_classes: 'int', mlp_features: 'List[int]', m: 'float'=0.5, s: 'float'=64, smoothing_epsilon: 'float'=0, label2category: 'Optional[Dict[Any, Any]]'=None, reduction: 'str'='mean'):
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            mlp_features: Layers sizes for MLP before ArcFace
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
            smoothing_epsilon: Label smoothing effect strength
            label2category: Optional, mapping from label to its category. If provided, label smoothing will redistribute
                 ``smoothing_epsilon`` only inside the category corresponding to the sample's ground truth label
            reduction: CrossEntropyLoss reduction
        """
        super().__init__()
        self.mlp = MLP(in_features, mlp_features)
        self.arcface = ArcFaceLoss(mlp_features[-1], num_classes=num_classes, label2category=label2category, smoothing_epsilon=smoothing_epsilon, m=m, s=s, reduction=reduction)

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        return self.arcface(self.mlp(x), y)

    @property
    def last_logs(self) ->Dict[str, Any]:
        """
        Returns:
             Dictionary containing useful statistic calculated for the last batch.
        """
        return self.arcface.last_logs


def pairwise_dist(x1: 'Tensor', x2: 'Tensor', p: 'int'=2) ->Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [M, D]
        p: degree

    Returns: pairwise distances with the shape of [N, M]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape[-1] == x2.shape[-1]
    return cdist(x1=x1, x2=x2, p=p)


def get_reduced(tensor: 'Tensor', reduction: 'str') ->Tensor:
    if reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    elif reduction == 'none':
        return tensor
    else:
        raise ValueError(f'Unexpected type of reduction: {reduction}')


def surrogate_precision(distances: 'Tensor', mask_gt: 'Tensor', k: 'int', t1: 'float', t2: 'float', reduction: 'str'='mean') ->Tensor:
    distances_diff = distances - distances.unsqueeze(0).permute(2, 1, 0)
    rank = sigmoid(distances_diff / t2).sum(dim=0)
    precision = (sigmoid((k - rank) / t1) * mask_gt).sum(dim=1) / torch.clip(mask_gt.sum(dim=1), max=k)
    return get_reduced(precision, reduction=reduction)


class SurrogatePrecision(torch.nn.Module):
    """
    This loss is a differentiable approximation of Precision@k metric.

    The loss is described in the following paper under a bit different name:
    `Recall@k Surrogate Loss with Large Batches and Similarity Mixup`_.

    The idea is that we express the formula for Precision@k using two step functions (aka Heaviside functions).
    Then we approximate them using two sigmoid functions with temperatures.
    The smaller temperature the close sigmoid to the step function, but the gradients are sparser,
    and vice versa. In the original paper `t1 = 1.0` and `t2 = 0.01` have been used.

    .. _Recall@k Surrogate Loss with Large Batches and Similarity Mixup:
        https://arxiv.org/pdf/2108.11179v2.pdf

    """
    criterion_name = 'surrogate_precision'

    def __init__(self, k: 'int', temperature1: 'float'=1.0, temperature2: 'float'=0.01, reduction: 'str'='mean'):
        """

        Args:
            k: Parameter of Precision@k.
            temperature1: Scaling factor for the 1st sigmoid, see docs above.
            temperature2: Scaling factor for the 2nd sigmoid, see docs above.
            reduction: ``mean``, ``sum`` or ``none``

        """
        super(SurrogatePrecision, self).__init__()
        assert k > 0
        """
        Note, since we consider all the batch samples as queries and galleries simultaneously,
        for each element we have its copy on the 1st position with the corresponding zero distance.
        Thus, to consider it we increase parameter k by 1.
        """
        self.k = k + 1
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.reduction = reduction

    def forward(self, features: 'torch.Tensor', labels: 'Tensor') ->Tensor:
        """

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        assert len(features) == len(labels)
        distances = pairwise_dist(x1=features, x2=features)
        mask_gt = labels[..., None] == labels[None, ...]
        loss = 1 - surrogate_precision(distances=distances, mask_gt=mask_gt, t1=self.temperature1, t2=self.temperature2, k=self.k, reduction=self.reduction)
        return loss


def elementwise_dist(x1: 'Tensor', x2: 'Tensor', p: 'int'=2) ->Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [N, D]
        p: degree

    Returns: elementwise distances with the shape of [N]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape == x2.shape
    if len(x1.shape) == 2:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
    dist = cdist(x1=x1, x2=x2, p=p).view(len(x1))
    return dist


class TripletLoss(Module):
    """
    Class, which combines classical `TripletMarginLoss` and `SoftTripletLoss`.
    The idea of `SoftTripletLoss` is the following:
    instead of using the classical formula
    ``loss = relu(margin + positive_distance - negative_distance)``
    we use
    ``loss = log1p(exp(positive_distance - negative_distance))``.
    It may help to solve the often problem when `TripletMarginLoss` converges to it's
    margin value (also known as `dimension collapse`).

    """
    criterion_name = 'triplet'

    def __init__(self, margin: 'Optional[float]', reduction: 'str'='mean', need_logs: 'bool'=False):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` to store some information to track in ``self.last_logs`` property.

        """
        assert reduction in ('mean', 'sum', 'none')
        assert margin is None or margin > 0
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs
        self._last_logs: 'Dict[str, float]' = {}

    def forward(self, anchor: 'Tensor', positive: 'Tensor', negative: 'Tensor') ->Tensor:
        """

        Args:
            anchor: Anchor features with the shape of ``(batch_size, feat)``
            positive: Positive features with the shape of ``(batch_size, feat)``
            negative: Negative features with the shape of ``(batch_size, feat)``

        Returns:
            Loss value

        """
        assert anchor.shape == positive.shape == negative.shape
        positive_dist = elementwise_dist(x1=anchor, x2=positive, p=2)
        negative_dist = elementwise_dist(x1=anchor, x2=negative, p=2)
        if self.margin is None:
            loss = torch.log1p(torch.exp(positive_dist - negative_dist))
        else:
            loss = torch.relu(self.margin + positive_dist - negative_dist)
        if self.need_logs:
            self._last_logs = {'active_tri': float((loss.clone().detach() > 0).float().mean()), 'pos_dist': float(positive_dist.clone().detach().mean().item()), 'neg_dist': float(negative_dist.clone().detach().mean().item())}
        loss = get_reduced(loss, reduction=self.reduction)
        return loss

    @property
    def last_logs(self) ->Dict[str, Any]:
        """
        Returns:
            Dictionary containing useful statistic calculated for the last batch.
        """
        return self._last_logs


def get_tri_ids_in_plain(n: 'int') ->Tuple[List[int], List[int], List[int]]:
    """
    Get ids for anchor, positive and negative samples for (n / 3) triplets
    to iterate over the plain structure.

    Args:
        n: (n / 3) is the number of desired triplets.

    Returns:
        Ids of anchor, positive and negative samples
        n = 1, ret = [0], [1], [2]
        n = 3, ret = [0, 3, 6], [1, 4, 7], [2, 5, 8]

    """
    assert n % 3 == 0
    anchor_ii = list(range(0, n, 3))
    positive_ii = list(range(1, n, 3))
    negative_ii = list(range(2, n, 3))
    return anchor_ii, positive_ii, negative_ii


class TripletLossPlain(Module):
    """
    The same as `TripletLoss`, but works with anchor, positive and negative features stacked together.

    """
    criterion_name = 'triplet'

    def __init__(self, margin: 'Optional[float]', reduction: 'str'='mean', need_logs: 'bool'=False):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` to store some information to track in ``self.last_logs`` property.

        """
        assert reduction in ('mean', 'sum', 'none')
        assert margin is None or margin > 0
        super(TripletLossPlain, self).__init__()
        self.criterion = TripletLoss(margin=margin, reduction=reduction, need_logs=need_logs)

    def forward(self, features: 'torch.Tensor') ->Tensor:
        """

        Args:
            features: Features with the shape of ``[batch_size, feat]`` with the following structure:
                      `0,1,2` are indices of the 1st triplet,
                      `3,4,5` are indices of the 2nd triplet,
                      and so on.
                      Thus, the features contains ``(N / 3)`` triplets

        Returns:
            Loss value

        """
        n = len(features)
        assert n % 3 == 0
        anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n)
        loss = self.criterion(features[anchor_ii], features[positive_ii], features[negative_ii])
        return loss

    @property
    def last_logs(self) ->Dict[str, Any]:
        """
        Returns:
            Dictionary containing useful statistic calculated for the last batch.
        """
        return self.criterion.last_logs


TTriplets = Tuple[Tensor, Tensor, Tensor]


class ITripletsMiner(ABC):
    """
    An abstraction of triplet miner.

    """

    @abstractmethod
    def sample(self, features: 'Tensor', labels: 'TLabels') ->TTriplets:
        """
        This method includes the logic of mining/sampling triplets.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Batch of triplets

        """
        raise NotImplementedError()


TTripletsIds = Tuple[List[int], List[int], List[int]]


def labels2list(labels: 'TLabels') ->List[int]:
    if isinstance(labels, Tensor):
        labels = labels.squeeze()
        labels_list = labels.tolist()
    elif isinstance(labels, list):
        labels_list = labels.copy()
    else:
        raise TypeError(f'Unexpected type of labels: {type(labels)}).')
    return labels_list


class ITripletsMinerInBatch(ITripletsMiner):
    """
    We expect that the child instances of this class
    will be used for mining triplets inside the batches.
    The batches must contain at least 2 samples for
    each class and at least 2 different labels,
    such behaviour can be guarantee via using samplers from
    our registry.

    But you are not limited to using it in any other way.

    """

    @staticmethod
    def _check_input_labels(labels: 'List[int]') ->None:
        """
        Args:
            labels: Labels of the samples in the batch

        """
        labels_counter = Counter(labels)
        assert all(n > 1 for n in labels_counter.values())
        assert len(labels_counter) > 1

    @abstractmethod
    def _sample(self, features: 'Tensor', labels: 'List[int]') ->TTripletsIds:
        """
        This method includes the logic of mining triplets
        inside the batch. It can be based on information about
        the distance between the features, or the
        choice can be performed randomly.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Indices of the batch samples to form the triplets

        """
        raise NotImplementedError()

    def sample(self, features: 'Tensor', labels: 'TLabels') ->TTriplets:
        """
        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
             Batch of triplets

        """
        labels = labels2list(labels)
        self._check_input_labels(labels=labels)
        ids_anchor, ids_pos, ids_neg = self._sample(features, labels=labels)
        return features[ids_anchor], features[ids_pos], features[ids_neg]


class TripletMinerWithMemory(ITripletsMiner):
    """
    This miner has a memory bank that allows to sample not only the triplets from the original batch,
    but also add batches obtained from both the bank and the original batch.

    """

    def __init__(self, bank_size_in_batches: 'int', tri_expand_k: 'int'):
        """

        Args:
            bank_size_in_batches: The size of the bank calculated in the number batches
            tri_expand_k: This parameter defines how many triplets we sample from the bank.
                 Specifically, we return ``tri_expand_k * number of original triplets``.
                 In particular, if ``tri_expand_k == 1`` we sample no triplets from the bank

        """
        assert tri_expand_k >= 1
        self.bank_size_in_batches = bank_size_in_batches
        self.tri_expand_k = tri_expand_k
        self.bank_features: 'Optional[Tensor]' = None
        self.bank_labels: 'Optional[Tensor]' = None
        self.bs = -1
        self.bank_size = -1
        self.ptr = 0

    @no_grad()
    def __allocate_if_needed(self, features: 'Tensor', labels: 'Tensor') ->None:
        if self.bank_features is None:
            assert len(features) == len(labels)
            self.bs = features.shape[0]
            self.feat_dim = features.shape[-1]
            self.bank_size = self.bank_size_in_batches * self.bs
            self.bank_labels = -1 * arange(1, self.bs + 1).repeat(self.bank_size_in_batches).long()
            self.bank_features = zeros([self.bank_size, self.feat_dim], dtype=features.dtype)
            self.bank_features[arange(self.bank_size), clip(abs(self.bank_labels), max=self.feat_dim - 1)] = 1

    @no_grad()
    def update_bank(self, features: 'Tensor', labels: 'Tensor') ->None:
        self.bank_features[self.ptr:self.ptr + self.bs] = features.clone().detach()
        self.bank_labels[self.ptr:self.ptr + self.bs] = labels.clone()
        self.ptr = (self.ptr + self.bs) % self.bank_size

    @no_grad()
    def get_pos_pairs(self, lbl2idx: 'Dict[Tensor, Tensor]', n: 'int'=None) ->Tensor:
        pos_batch_pairs = zeros(0, 2)
        if n is not None:
            while len(pos_batch_pairs) < n:
                pos_ii = choice(list(lbl2idx.values()))
                combs = combinations(pos_ii, r=2)
                pos_batch_pairs = cat([pos_batch_pairs, combs, flip(combs, [1])])
        else:
            for pos_ii in lbl2idx.values():
                combs = combinations(pos_ii, r=2)
                pos_batch_pairs = cat([pos_batch_pairs, combs, flip(combs, [1])])
        return pos_batch_pairs.long()[randperm(len(pos_batch_pairs))[:n]]

    def sample(self, features: 'Tensor', labels: 'Tensor') ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            features: Features with the shape of ``(batch_size, feat_dim)``
            labels: Labels with the size of ``batch_size``

        Returns:
            Triplets made from the original batch and those that were combined from the bank and the batch.
            We also return an indicator of whether triplet was obtained from the original batch.
            So, output is the following ``(anchor, positive, negative, indicators)``

        """
        labels = tensor(labels).long()
        self.__allocate_if_needed(features=features, labels=labels)
        assert len(features) == len(labels) == self.bs, (len(features), len(labels), self.bs)
        lbl2idx_bank = {lb: arange(self.bank_size)[self.bank_labels == lb] for lb in unique(self.bank_labels)}
        lbl2idx_batch = {lb: arange(self.bs)[labels == lb] for lb in unique(labels)}
        ii_anch_pos_1 = self.get_pos_pairs(lbl2idx_batch)
        ii_all = arange(self.bs)
        ii_pos_pairs_1, ii_neg_1 = cartesian_prod(arange(len(ii_anch_pos_1)), ii_all).T
        ii_anch_1, ii_pos_1 = ii_anch_pos_1[ii_pos_pairs_1].T
        ii_anch_1, ii_pos_1, ii_neg_1 = self.take_tri_by_mask(ii_anch_1, ii_pos_1, ii_neg_1, mask=labels[ii_anch_1] != labels[ii_neg_1])
        n_batch_tri = len(ii_anch_1)
        n_tri_positives_from_bank = int(n_batch_tri * (self.tri_expand_k - 1) / 2)
        ii_anch_2, ii_pos_2 = self.get_pos_pairs(lbl2idx_bank, n_tri_positives_from_bank).T
        ii_neg_2 = randint(0, self.bs, size=(len(ii_anch_2),))
        ii_anch_2, ii_pos_2, ii_neg_2 = self.take_tri_by_mask(ii_anch_2, ii_pos_2, ii_neg_2, mask=self.bank_labels[ii_anch_2] != labels[ii_neg_2])
        n_tri_negatives_from_bank = n_tri_positives_from_bank
        ii_anch_3, ii_pos_3 = self.get_pos_pairs(lbl2idx_batch).T
        ii_neg_3 = randint(0, self.bank_size, size=(n_tri_negatives_from_bank,))
        ii_3 = randint(0, len(ii_anch_3), size=(n_tri_negatives_from_bank,))
        ii_anch_3 = ii_anch_3[ii_3]
        ii_pos_3 = ii_pos_3[ii_3]
        ii_anch_3, ii_pos_3, ii_neg_3 = self.take_tri_by_mask(ii_anch_3, ii_pos_3, ii_neg_3, mask=labels[ii_anch_3] != self.bank_labels[ii_neg_3])
        features_anch = cat([features[ii_anch_3], self.bank_features[ii_anch_2], features[ii_anch_1]])
        features_pos = cat([features[ii_pos_3], self.bank_features[ii_pos_2], features[ii_pos_1]])
        features_neg = cat([self.bank_features[ii_neg_3], features[ii_neg_2], features[ii_neg_1]])
        assert len(features_anch) == len(features_pos) == len(features_neg)
        self.update_bank(features=features, labels=labels)
        is_original_tri = zeros(len(features_anch), dtype=bool).cpu()
        is_original_tri[-len(ii_anch_1):] = True
        return features_anch, features_pos, features_neg, is_original_tri

    @staticmethod
    def take_tri_by_mask(ii_a: 'Tensor', ii_p: 'Tensor', ii_n: 'Tensor', mask: 'Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        ii_a = ii_a[mask]
        ii_p = ii_p[mask]
        ii_n = ii_n[mask]
        return ii_a, ii_p, ii_n


class IFreezable(ABC):
    """
    Models which can freeze and unfreeze their parts.
    """

    def freeze(self) ->None:
        """
        Function for freezing. You can use it to partially freeze a model.
        """
        raise NotImplementedError()

    def unfreeze(self) ->None:
        """
        Function for unfreezing. You can use it to unfreeze a model.
        """
        raise NotImplementedError()


STORAGE_URL = 'https://oml.daloroserver.com'


STORAGE_CKPTS = STORAGE_URL + '/download/checkpoints'


_FB_URL = 'https://dl.fbaipublicfiles.com'


def dino_vitb16(pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    Achieves 76.1% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_base'](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    Achieves 77.4% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_base'](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_small'](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


class Weights(Enum):
    LVD142M = 'LVD142M'


_DINOV2_BASE_URL = 'https://dl.fbaipublicfiles.com/dinov2'


def _make_dinov2_model_name(arch_name: 'str', patch_size: 'int', num_register_tokens: 'int'=0) ->str:
    compact_arch_name = arch_name.replace('_', '')[:4]
    registers_suffix = f'_reg{num_register_tokens}' if num_register_tokens else ''
    return f'dinov2_{compact_arch_name}{patch_size}{registers_suffix}'


def _make_dinov2_model(*, arch_name: str='vit_large', img_size: int=518, patch_size: int=14, init_values: float=1.0, ffn_layer: str='mlp', block_chunks: int=0, num_register_tokens: int=0, interpolate_antialias: bool=False, interpolate_offset: float=0.1, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError as e:
            raise AssertionError(f'Unsupported weights: {weights}') from e
    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = {'img_size': img_size, 'patch_size': patch_size, 'init_values': init_values, 'ffn_layer': ffn_layer, 'block_chunks': block_chunks, 'num_register_tokens': num_register_tokens, 'interpolate_antialias': interpolate_antialias, 'interpolate_offset': interpolate_offset}
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)
    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f'/{model_base_name}/{model_full_name}_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', model_dir=str(CKPT_SAVE_ROOT.resolve()))
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov2_vitb14(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name='vit_base', pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14_reg(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset
    """
    return _make_dinov2_model(arch_name='vit_base', pretrained=pretrained, weights=weights, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0, **kwargs)


def dinov2_vitl14(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name='vit_large', pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14_reg(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset
    """
    return _make_dinov2_model(arch_name='vit_large', pretrained=pretrained, weights=weights, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0, **kwargs)


def dinov2_vits14(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name='vit_small', pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vits14_reg(*, pretrained: bool=True, weights: Union[Weights, str]=Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset
    """
    return _make_dinov2_model(arch_name='vit_small', pretrained=pretrained, weights=weights, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0, **kwargs)


def calc_hash(source: 'Union[Path, str, bytes]') ->str:
    if isinstance(source, (Path, str)):
        source = Path(source).resolve()
        descriptor = FileIO(str(source), mode='rb')
    elif isinstance(source, bytes):
        descriptor = BytesIO(source)
    else:
        raise TypeError('Not supported type')
    descriptor.seek(0)
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda : descriptor.read(4096), b''):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


REQUESTS_TIMEOUT = 120.0


def download_file_from_url(url: 'str', fname: 'Optional[str]'=None, timeout: 'float'=REQUESTS_TIMEOUT) ->Optional[bytes]:
    assert validators.url(url), 'Invalid URL'
    response = requests.get(url, timeout=timeout)
    if response.status_code == 200:
        if fname is not None:
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            with open(fname, 'wb+') as f:
                f.write(response.content)
            return None
        else:
            return response.content
    else:
        raise RuntimeError(f"Can not download file from '{url}'")


def download_checkpoint(url_or_fid: 'str', hash_md5: 'str', fname: 'Optional[str]'=None) ->str:
    """
    Args:
        url_or_fid: URL to the checkpoint or file id on Google Drive
        hash_md5: Value of md5sum
        fname: Name of the checkpoint after the downloading process

    Returns:
        Path to the checkpoint

    """
    CKPT_SAVE_ROOT.mkdir(exist_ok=True, parents=True)
    fname = fname if fname else Path(url_or_fid).name
    save_path = str(CKPT_SAVE_ROOT / fname)
    if Path(save_path).exists():
        actual_hash = calc_hash(save_path)
        if actual_hash.startswith(hash_md5):
            None
            return save_path
        else:
            None
            Path(save_path).unlink()
    None
    if validators.url(url_or_fid):
        download_file_from_url(url=url_or_fid, fname=save_path)
    else:
        gdown.download(id=url_or_fid, output=save_path, quiet=False)
    if not calc_hash(save_path).startswith(hash_md5):
        raise Exception("Downloaded checkpoint is probably broken. Hash values don't match.")
    return str(save_path)


def download_checkpoint_one_of(url_or_fid_list: 'Union[List[str], str]', hash_md5: 'str', fname: 'Optional[str]'=None) ->str:
    """
    The function iteratively tries to download a checkpoint from the list of resources and stops at the first
    one available for download.
    """
    if not isinstance(url_or_fid_list, (tuple, list)):
        url_or_fid_list = [url_or_fid_list]
    attempt = 0
    for url_or_fid in url_or_fid_list:
        attempt += 1
        None
        try:
            return download_checkpoint(url_or_fid, hash_md5, fname)
        except Exception:
            if attempt == len(url_or_fid_list):
                raise
    return None


def normalise(x: 'Tensor', p: 'int'=2) ->Tensor:
    """
    Args:
        x: A 2D tensor
        p: Specifies the exact p-norm

    Returns:
        Normalised input

    """
    assert x.ndim == 2
    xn = torch.linalg.norm(x, p, dim=1).detach()
    x = x.div(xn.unsqueeze(1))
    return x


TStateDict = OrderedDict[str, torch.Tensor]


def remove_criterion_in_state_dict(state_dict: 'TStateDict') ->TStateDict:
    if 'criterion.weight' in state_dict:
        del state_dict['criterion.weight']
    return state_dict


def find_prefix_in_state_dict(state_dict: 'TStateDict', trial_key: 'str') ->str:
    keys_starting_with_trial_key = [k for k in state_dict.keys() if trial_key in k]
    assert keys_starting_with_trial_key, f'There are no keys starting from {trial_key}.\nThe existing keys are: {list(state_dict.keys())}'
    k0 = keys_starting_with_trial_key[0]
    prefix = k0[:k0.index(trial_key)]
    keys_not_starting_with_prefix = list(filter(lambda x: not x.startswith(prefix), state_dict.keys()))
    assert not keys_not_starting_with_prefix, f'There are keys not starting from the found prefix {prefix}: {keys_not_starting_with_prefix}'
    return prefix


def remove_prefix_from_state_dict(state_dict: 'TStateDict', trial_key: 'str') ->TStateDict:
    prefix = find_prefix_in_state_dict(state_dict, trial_key)
    if prefix == '':
        return state_dict
    else:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict[k]
                del state_dict[k]
        None
        return state_dict


class ViTExtractor(IExtractor):
    """
    The base class for the extractors that follow VisualTransformer architecture.

    """
    constructors = {'vits8': dino_vits8, 'vits16': dino_vits16, 'vitb8': dino_vitb8, 'vitb16': dino_vitb16, 'vits14': dinov2_vits14, 'vitb14': dinov2_vitb14, 'vitl14': dinov2_vitl14, 'vits14_reg': dinov2_vits14_reg, 'vitb14_reg': dinov2_vitb14_reg, 'vitl14_reg': dinov2_vitl14_reg}
    pretrained_models = {'vits16_dino': {'url': f'{_FB_URL}/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth', 'hash': 'cf0f22', 'fname': 'vits16_dino.ckpt', 'init_args': {'arch': 'vits16', 'normalise_features': False}}, 'vits8_dino': {'url': f'{_FB_URL}/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth', 'hash': '230cd5', 'fname': 'vits8_dino.ckpt', 'init_args': {'arch': 'vits8', 'normalise_features': False}}, 'vitb16_dino': {'url': f'{_FB_URL}/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', 'hash': '552daf', 'fname': 'vitb16_dino.ckpt', 'init_args': {'arch': 'vitb16', 'normalise_features': False}}, 'vitb8_dino': {'url': f'{_FB_URL}/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth', 'hash': '556550', 'fname': 'vitb8_dino.ckpt', 'init_args': {'arch': 'vitb8', 'normalise_features': False}}, 'vits14_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', 'hash': '2e405c', 'fname': 'dinov2_vits14.ckpt', 'init_args': {'arch': 'vits14', 'normalise_features': False}}, 'vits14_reg_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth', 'hash': '2a50c5', 'fname': 'dinov2_vits14_reg4.ckpt', 'init_args': {'arch': 'vits14_reg', 'normalise_features': False}}, 'vitb14_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth', 'hash': '8635e7', 'fname': 'dinov2_vitb14.ckpt', 'init_args': {'arch': 'vitb14', 'normalise_features': False}}, 'vitb14_reg_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth', 'hash': '13d13c', 'fname': 'dinov2_vitb14_reg4.ckpt', 'init_args': {'arch': 'vitb14_reg', 'normalise_features': False}}, 'vitl14_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'hash': '19a02c', 'fname': 'dinov2_vitl14.ckpt', 'init_args': {'arch': 'vitl14', 'normalise_features': False}}, 'vitl14_reg_dinov2': {'url': f'{_FB_URL}/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth', 'hash': '8b6364', 'fname': 'dinov2_vitl14_reg4.ckpt', 'init_args': {'arch': 'vitl14_reg', 'normalise_features': False}}, 'vits16_inshop': {'url': [f'{STORAGE_CKPTS}/inshop/vits16_inshop_a76b85.ckpt', '1niX-TC8cj6j369t7iU2baHQSVN3MVJbW'], 'hash': 'a76b85', 'fname': 'vits16_inshop.ckpt', 'init_args': {'arch': 'vits16', 'normalise_features': False}}, 'vits16_sop': {'url': [f'{STORAGE_CKPTS}/sop/vits16_sop_21e743.ckpt', '1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A'], 'hash': '21e743', 'fname': 'vits16_sop.ckpt', 'init_args': {'arch': 'vits16', 'normalise_features': True}}, 'vits16_cub': {'url': [f'{STORAGE_CKPTS}/cub/vits16_cub.ckpt', '1p2tUosFpGXh5sCCdzlXtjV87kCDfG34G'], 'hash': 'e82633', 'fname': 'vits16_cub.ckpt', 'init_args': {'arch': 'vits16', 'normalise_features': False}}, 'vits16_cars': {'url': [f'{STORAGE_CKPTS}/cars/vits16_cars.ckpt', '1hcOxDRRXrKr6ZTCyBauaY8Ue-pok4Icg'], 'hash': '9f1e59', 'fname': 'vits16_cars.ckpt', 'init_args': {'arch': 'vits16', 'normalise_features': False}}}

    def __init__(self, weights: 'Optional[Union[Path, str]]', arch: 'str', normalise_features: 'bool', use_multi_scale: 'bool'=False):
        """
        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to
             randomly initialize model's weights. You can check the available pretrained checkpoints
             in ``self.pretrained_models``.
            arch: Might be one of ``vits8``, ``vits16``, ``vitb8``, ``vitb16``. You can check all the available options
             in ``self.constructors``
            normalise_features: Set ``True`` to normalise output features
            use_multi_scale: Set ``True`` to use multiscale (the analogue of test time augmentations)

        """
        assert arch in self.constructors
        super(ViTExtractor, self).__init__()
        self.normalise_features = normalise_features
        self.mscale = use_multi_scale
        self.arch = arch
        factory_fun = self.constructors[self.arch]
        self.model = factory_fun(pretrained=False)
        if weights is None:
            return
        if weights in self.pretrained_models:
            pretrained = self.pretrained_models[weights]
            weights = download_checkpoint_one_of(url_or_fid_list=pretrained['url'], hash_md5=pretrained['hash'], fname=pretrained['fname'])
        ckpt = torch.load(weights, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        state_dict = remove_criterion_in_state_dict(state_dict)
        ckpt = remove_prefix_from_state_dict(state_dict, trial_key='norm.bias')
        self.model.load_state_dict(ckpt, strict=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.mscale:
            x = self._multi_scale(x)
        else:
            x = self.model(x)
        if self.normalise_features:
            x = normalise(x)
        return x

    @property
    def feat_dim(self) ->int:
        return len(self.model.norm.bias)

    def _multi_scale(self, samples: 'torch.Tensor') ->torch.Tensor:
        v = torch.zeros((len(samples), self.feat_dim), device=samples.device)
        scales = [1.0, 1 / 2 ** (1 / 2), 1 / 2]
        for s in scales:
            if s == 1:
                inp = samples.clone()
            else:
                inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
            feats = self.model.forward(inp).clone()
            v += feats
        v /= len(scales)
        return v

    def draw_attention(self, image: 'Union[TPILImage, np.ndarray]') ->np.ndarray:
        """
        Args:
            image: An image with pixel values in the range of ``[0..255]``.

        Returns:
            An image with drawn attention maps.

        Visualization of the multi-head attention on a particular image.

        """
        return vis_vit(vit=self, image=image)


class ExtractorWithMLP(IExtractor, IFreezable):
    """
    Class-wrapper for extractors which an additional MLP.

    """
    pretrained_models = {'vits16_224_mlp_384_inshop': {'url': f'{STORAGE_CKPTS}/inshop/vits16_224_mlp_384_inshop.ckpt', 'hash': '35244966', 'fname': 'vits16_224_mlp_384_inshop.ckpt', 'init_args': {'extractor_creator': lambda : ViTExtractor(None, 'vits16', False, use_multi_scale=False), 'mlp_features': [384], 'train_backbone': True}}}

    def __init__(self, extractor: 'IExtractor', mlp_features: 'List[int]', weights: 'Optional[Union[str, Path]]'=None, train_backbone: 'bool'=False):
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)
            mlp_features: Sizes of projection layers
            weights: Path to weights file or ``None`` for random initialization
            train_backbone: set ``False`` if you want to train only MLP head

        """
        IExtractor.__init__(self)
        self.extractor = extractor
        self.mlp_features = mlp_features
        self.train_backbone = train_backbone
        self.projection = MLP(self.extractor.feat_dim, self.mlp_features)
        if weights:
            if weights in self.pretrained_models:
                pretrained = self.pretrained_models[weights]
                weights = download_checkpoint(url_or_fid=pretrained['url'], hash_md5=pretrained['hash'], fname=pretrained['fname'])
            loaded = torch.load(weights, map_location='cpu')
            loaded = loaded.get('state_dict', loaded)
            loaded = remove_criterion_in_state_dict(loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key='extractor.')
            self.load_state_dict(loaded, strict=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        with torch.set_grad_enabled(self.train_backbone):
            features = self.extractor(x)
        return self.projection(features)

    @property
    def feat_dim(self) ->int:
        return self.mlp_features[-1]

    def freeze(self) ->None:
        self.train_backbone = False

    def unfreeze(self) ->None:
        self.train_backbone = True

    @classmethod
    def from_pretrained(cls, weights: 'str', **kwargs) ->'IExtractor':
        ini = cls.pretrained_models[weights]['init_args']
        ini['extractor'] = ini.pop('extractor_creator')()
        return super().from_pretrained(weights, **kwargs)


class LinearTrivialDistanceSiamese(IPairwiseModel):
    """
    This model is a useful tool mostly for development.

    """

    def __init__(self, feat_dim: 'int', identity_init: 'bool', output_bias: 'float'=0):
        """
        Args:
            feat_dim: Expected size of each input.
            identity_init: If ``True``, models' weights initialised in a way when
                the model simply estimates L2 distance between the original embeddings.
            output_bias: Value to add to the output.

        """
        super(LinearTrivialDistanceSiamese, self).__init__()
        self.feat_dim = feat_dim
        self.output_bias = output_bias
        self.proj = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)
        if identity_init:
            self.proj.load_state_dict({'weight': torch.eye(feat_dim)})

    def forward(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        """
        Args:
            x1: Embedding with the shape of ``[batch_size, feat_dim]``
            x2: Embedding with the shape of ``[batch_size, feat_dim]``

        Returns:
            Distance between transformed inputs.

        """
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        y = elementwise_dist(x1, x2, p=2) + self.output_bias
        return y

    def predict(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        return self.forward(x1=x1, x2=x2)


class ConcatSiamese(IPairwiseModel, IFreezable):
    """
    This model concatenates two inputs and passes them through
    a given backbone and applyies a head after that.

    """
    pretrained_models: 'Dict[str, Any]' = {}

    def __init__(self, extractor: 'IExtractor', mlp_hidden_dims: 'List[int]', use_tta: 'bool'=False, weights: 'Optional[Union[str, Path]]'=None) ->None:
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)
            mlp_hidden_dims: Hidden dimensions of the head
            use_tta: Set ``True`` if you want to average the results obtained by two different orders of concatenating
             input images. Affects only ``self.predict()`` method.
            weights: Path to weights file or ``None`` for random initialization

        """
        super(ConcatSiamese, self).__init__()
        self.extractor = extractor
        self.use_tta = use_tta
        self.head = MLP(in_channels=self.extractor.feat_dim, hidden_channels=[*mlp_hidden_dims, 1], activation_layer=Sigmoid, dropout=0.5, inplace=None)
        self.head[-2] = nn.Linear(self.head[-2].in_features, self.head[-2].out_features, bias=False)
        self.head[-1] = nn.Identity()
        self.train_backbone = True
        if weights:
            if weights in self.pretrained_models:
                url_or_fid, hash_md5, fname = self.pretrained_models[weights]
                weights = download_checkpoint(url_or_fid=url_or_fid, hash_md5=hash_md5, fname=fname)
            loaded = torch.load(weights, map_location='cpu')
            loaded = loaded.get('state_dict', loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key='extractor.')
            self.load_state_dict(loaded, strict=True)

    def forward(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        x = torch.concat([x1, x2], dim=2)
        with torch.set_grad_enabled(self.train_backbone):
            x = self.extractor(x)
        x = self.head(x)
        x = x.view(len(x))
        return x

    def predict(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        x = self.forward(x1=x1, x2=x2)
        x = torch.sigmoid(x)
        if self.use_tta:
            y = self.forward(x1=x2, x2=x1)
            y = torch.sigmoid(y)
            return (x + y) / 2
        else:
            return x

    def freeze(self) ->None:
        self.train_backbone = False

    def unfreeze(self) ->None:
        self.train_backbone = True


class TrivialDistanceSiamese(IPairwiseModel):
    """
    This model is a useful tool mostly for development.

    """
    pretrained_models: 'Dict[str, Any]' = {}

    def __init__(self, extractor: 'IExtractor', output_bias: 'float'=0) ->None:
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)
            output_bias: Value to add to the outputs.

        """
        super(TrivialDistanceSiamese, self).__init__()
        self.extractor = extractor
        self.output_bias = output_bias

    def forward(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        """
        Args:
            x1: The first input.
            x2: The second input.

        Returns:
            Distance between inputs.

        """
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        return elementwise_dist(x1, x2, p=2) + self.output_bias

    def predict(self, x1: 'Tensor', x2: 'Tensor') ->Tensor:
        return self.forward(x1=x1, x2=x2)


class GEM(nn.Module):

    def __init__(self, p: 'float', eps: 'float'=1e-06):
        """
        Generalised Mean Pooling (GEM)
        https://paperswithcode.com/method/generalized-mean-pooling

        Args:
            p: if p == 1 it's average pooling, if p == inf it's max-pooling
            eps: eps for numerical stability
        """
        super(GEM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = torch.clip(x, min=self.eps, max=np.inf)
        x = torch.pow(x, self.p)
        bs, feat_dim, h, w = x.shape
        x = x.view(bs, feat_dim, h * w)
        x = x.mean(axis=-1)
        x = torch.pow(x, 1.0 / self.p)
        return x


def get_device(model: 'torch.nn.Module') ->str:
    return str(next(model.parameters()).device)


def load_moco_weights(path_to_model: 'Union[str, Path]') ->Dict[str, Any]:
    """
    Args:
        path_to_model: Path to model trained using original
           code from MoCo repository:
           https://github.com/facebookresearch/moco

    Returns:
        State dict without weights of student

    """
    checkpoint = torch.load(path_to_model, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('module.encoder_q'):
            new_key = key[len('module.encoder_q.'):]
            state_dict[new_key] = state_dict[key]
        del state_dict[key]
    return state_dict


def resnet50_projector() ->nn.Module:
    model = resnet50(weights=None, num_classes=128)
    model.fc = nn.Sequential(nn.Linear(model.fc.weight.shape[1], model.fc.weight.shape[1]), nn.ReLU(), model.fc)
    return model

