
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


import copy


from typing import List


from typing import Tuple


import torch


from torch import Tensor


from torch.nn import Identity


from torchvision.models import resnet50


import math


from typing import Union


from torch.optim import SGD


from torch.optim.optimizer import Optimizer


from typing import Dict


from torch.nn import Module


from torch.utils.data import DataLoader


from torchvision import transforms as T


from typing import Sequence


from torch.nn import ModuleList


from torch.nn import functional as F


from typing import Optional


from torch.nn import MSELoss


from torch.optim import AdamW


from torch.nn import BatchNorm1d


from torch.nn import Linear


from torch.nn import Sequential


from torch.nn import Parameter


from collections import OrderedDict


import time


import numpy as np


import torch.nn as nn


import torchvision


from sklearn.cluster import KMeans


from torch.optim.lr_scheduler import LambdaLR


from torchvision.models.vision_transformer import VisionTransformer


from torchvision import transforms


from torchvision.datasets import Food101


import matplotlib.pyplot as plt


import pandas


from sklearn.neighbors import NearestNeighbors


from sklearn.preprocessing import normalize


import matplotlib.offsetbox as osb


import torchvision.transforms.functional as functional


from matplotlib import rcParams as rcp


from sklearn import random_projection


from torch import nn


import warnings


from torch import nn as nn


from typing import Any


from torch.utils.data import Dataset


from torchvision import datasets


from torchvision import io


from warnings import warn


import torchvision.transforms as T


from typing import Callable


import torchvision.datasets as datasets


from torchvision.datasets.vision import StandardTransform


from torchvision.datasets.vision import VisionDataset


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from typing import TYPE_CHECKING


import torch.distributed as dist


import torch.nn.functional as F


from functools import partial


from torch import distributed as torch_dist


from torch.nn import PairwiseDistance


from torch.nn import functional


from torch.linalg import svd


from torch.nn.functional import cosine_similarity


from typing import Iterable


from torch.nn.parameter import Parameter


from typing import Type


from torch.nn import GELU


from torch.nn import LayerNorm


from torchvision.models import vision_transformer


from torchvision.models.vision_transformer import ConvStemConfig


from torch import jit


from abc import ABC


from abc import abstractmethod


import random


from torch.nn import init


from torch.nn.modules import CrossMapLRN2d


from torch.nn.modules import GroupNorm


from torch.nn.modules import LayerNorm


from torch.nn.modules import LocalResponseNorm


from torch.nn.modules.batchnorm import _NormBase


from torchvision.ops import StochasticDepth


from torch.distributions import Uniform


import torch.fft


from torch.distributions.bernoulli import Bernoulli


import torchvision.transforms.functional as F


from torchvision.transforms import functional as TF


from torchvision.transforms import ToTensor as ToTensorV1


from torch.nn import CrossEntropyLoss


import functools


from typing import TypeVar


from torch.autograd.function import FunctionCtx


from typing import overload


import torchvision.transforms as transforms


from torch import manual_seed


from torch import distributed as dist


import itertools


from torchvision.models import VisionTransformer


import re


from torch.nn import Flatten


from torchvision.datasets import FakeData


from torchvision.transforms import ToTensor


from typing import Generator


import torch.distributed


from torch.utils.data import TensorDataset


class SelectStage(torch.nn.Module):
    """Selects features from a given stage."""

    def __init__(self, stage: 'str'='res5'):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer, use_bias (optional)).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(self, blocks: 'Sequence[Union[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]], Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool]],]') ->None:
        """Initializes the ProjectionHead module with the specified blocks."""
        super().__init__()
        layers: 'List[nn.Module]' = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: 'Tensor') ->Tensor:
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        projection: 'Tensor' = self.layers(x)
        return projection


class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam.

    "The prediction MLP (h) has BN applied to its hidden fc layers. Its output
    fc does not have BN (...) or ReLU. This MLP has 2 layers." [0]

    - [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=512, output_dim: 'int'=2048):
        """Initializes the SimSiamPredictionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(SimSiamPredictionHead, self).__init__([(input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, output_dim, None, None)])


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.

    "The projection MLP (in f) has BN applied to each fully-connected (fc)
    layer, including its output fc. Its output fc has no ReLU. The hidden fc is
    2048-d. This MLP has 3 layers." [0]

    - [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=2048, output_dim: 'int'=2048):
        super(SimSiamProjectionHead, self).__init__([(input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, output_dim, nn.BatchNorm1d(output_dim, affine=False), None)])


class SimSiam(nn.Module):
    """Implementation of SimSiam[0] network

    Recommended loss: :py:class:`lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss`

    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head. This should
            be `num_ftrs` / 4.
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self, backbone: 'nn.Module', num_ftrs: 'int'=2048, proj_hidden_dim: 'int'=2048, pred_hidden_dim: 'int'=512, out_dim: 'int'=2048):
        super(SimSiam, self).__init__()
        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim
        self.projection_mlp = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_mlp = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)
        warnings.warn(Warning('The high-level building block SimSiam will be deprecated in version 1.3.0. ' + 'Use low-level building blocks instead. ' + 'See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information'), DeprecationWarning)

    def forward(self, x0: 'torch.Tensor', x1: 'torch.Tensor'=None, return_features: 'bool'=False):
        """Forward pass through SimSiam.

        Extracts features with the backbone and applies the projection
        head and prediction head to the output space. If both x0 and x1 are not
        None, both will be passed through the backbone, projection, and
        prediction head. If x1 is None, only x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output prediction and projection of x0 and (if x1 is not None)
            the output prediction and projection of x1. If return_features is
            True, the output for each x is a tuple (out, f) where f are the
            features before the projection head.

        Examples:
            >>> # single input, single output
            >>> out = model(x)
            >>>
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
        """
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_mlp(f0)
        p0 = self.prediction_mlp(z0)
        out0 = z0, p0
        if return_features:
            out0 = out0, f0
        if x1 is None:
            return out0
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_mlp(f1)
        p1 = self.prediction_mlp(z1)
        out1 = z1, p1
        if return_features:
            out1 = out1, f1
        return out0, out1


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]

    "We use a 3-layer MLP projection head on top of a ResNet encoder." [1]

    - [0] SimCLR v1, 2020, https://arxiv.org/abs/2002.05709
    - [1] SimCLR v2, 2020, https://arxiv.org/abs/2006.10029
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=2048, output_dim: 'int'=128, num_layers: 'int'=2, batch_norm: 'bool'=True):
        """Initialize a new SimCLRProjectionHead instance.

        Args:
            input_dim:
                Number of input dimensions.
            hidden_dim:
                Number of hidden dimensions.
            output_dim:
                Number of output dimensions.
            num_layers:
                Number of hidden layers (2 for v1, 3+ for v2).
            batch_norm:
                Whether or not to use batch norms.
        """
        layers: 'List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]' = []
        layers.append((input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim) if batch_norm else None, nn.ReLU()))
        for _ in range(2, num_layers):
            layers.append((hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim) if batch_norm else None, nn.ReLU()))
        layers.append((hidden_dim, output_dim, nn.BatchNorm1d(output_dim) if batch_norm else None, None))
        super().__init__(layers)


class SimCLR(nn.Module):
    """Implementation of the SimCLR[0] architecture

    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self, backbone: 'nn.Module', num_ftrs: 'int'=32, out_dim: 'int'=128):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs, out_dim, batch_norm=False)
        warnings.warn(Warning('The high-level building block SimCLR will be deprecated in version 1.3.0. ' + 'Use low-level building blocks instead. ' + 'See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information'), DeprecationWarning)

    def forward(self, x0: 'torch.Tensor', x1: 'torch.Tensor'=None, return_features: 'bool'=False):
        """Embeds and projects the input images.

        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection head. If x1 is None, only
        x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None) the output
            projection of x1. If return_features is True, the output for each x
            is a tuple (out, f) where f are the features before the projection
            head.

        Examples:
            >>> # single input, single output
            >>> out = model(x)
            >>>
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)

        """
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_head(f0)
        if return_features:
            out0 = out0, f0
        if x1 is None:
            return out0
        f1 = self.backbone(x1).flatten(start_dim=1)
        out1 = self.projection_head(f1)
        if return_features:
            out1 = out1, f1
        return out0, out1


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    - [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=8192, output_dim: 'int'=8192):
        """Initializes the BarlowTwinsProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(BarlowTwinsProjectionHead, self).__init__([(input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, output_dim, None, None)])


class BarlowTwins(nn.Module):
    """Implementation of BarlowTwins[0] network.

    Recommended loss: :py:class:`lightly.loss.barlow_twins_loss.BarlowTwinsLoss`

    Default params are the ones explained in the original paper [0].
    [0] Zbontar,J. et.al. 2021. Barlow Twins... https://arxiv.org/abs/2103.03230

    Attributes:
        backbone:
            Backbone model to extract features from images.
            ResNet-50 in original paper [0].
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self, backbone: 'nn.Module', num_ftrs: 'int'=2048, proj_hidden_dim: 'int'=8192, out_dim: 'int'=8192):
        super(BarlowTwins, self).__init__()
        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.out_dim = out_dim
        self.projection_mlp = BarlowTwinsProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        warnings.warn(Warning('The high-level building block BarlowTwins will be deprecated in version 1.3.0. ' + 'Use low-level building blocks instead. ' + 'See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information'), DeprecationWarning)

    def forward(self, x0: 'torch.Tensor', x1: 'torch.Tensor'=None, return_features: 'bool'=False):
        """Forward pass through BarlowTwins.

        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection. If x1 is None, only x0 will
        be forwarded.
        Barlow Twins only implement a projection head unlike SimSiam.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None)
            the output projection of x1. If return_features is
            True, the output for each x is a tuple (out, f) where f are the
            features before the projection head.

        Examples:
            >>> # single input, single output
            >>> out = model(x)
            >>>
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
        """
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_mlp(f0)
        if return_features:
            out0 = out0, f0
        if x1 is None:
            return out0
        f1 = self.backbone(x1).flatten(start_dim=1)
        out1 = self.projection_mlp(f1)
        if return_features:
            out1 = out1, f1
        return out0, out1


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    - [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=4096, output_dim: 'int'=256):
        """Initializes the BYOLProjectionHead with the specified dimensions."""
        super(BYOLProjectionHead, self).__init__([(input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()), (hidden_dim, output_dim, None, None)])


def _deactivate_requires_grad(params: 'Iterable[Parameter]') ->None:
    """Deactivates the requires_grad flag for all parameters."""
    for param in params:
        param.requires_grad = False


def _do_momentum_update(prev_params: 'Iterable[Parameter]', params: 'Iterable[Parameter]', m: 'float') ->None:
    """Updates the weights of the previous parameters."""
    for prev_param, param in zip(prev_params, params):
        prev_param.data = prev_param.data * m + param.data * (1.0 - m)


class _MomentumEncoderMixin:
    """Mixin to provide momentum encoder functionalities.

    Provides the following functionalities:
        - Momentum encoder initialization.
        - Momentum updates.
        - Batch shuffling and unshuffling.

    To make use of the mixin, simply inherit from it:

    >>> class MyMoCo(nn.Module, _MomentumEncoderMixin):
    >>>
    >>>     def __init__(self, backbone):
    >>>         super(MyMoCo, self).__init__()
    >>>
    >>>         self.backbone = backbone
    >>>         self.projection_head = get_projection_head()
    >>>
    >>>         # initialize momentum_backbone and momentum_projection_head
    >>>         self._init_momentum_encoder()
    >>>
    >>>     def forward(self, x: Tensor):
    >>>
    >>>         # do the momentum update
    >>>         self._momentum_update(0.999)
    >>>
    >>>         # use momentum backbone
    >>>         y = self.momentum_backbone(x)
    >>>         y = self.momentum_projection_head(y)

    """
    m: 'float'
    backbone: 'nn.Module'
    projection_head: 'nn.Module'
    momentum_backbone: 'nn.Module'
    momentum_projection_head: 'nn.Module'

    def _init_momentum_encoder(self) ->None:
        """Initializes momentum backbone and a momentum projection head."""
        assert self.backbone is not None
        assert self.projection_head is not None
        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection_head = copy.deepcopy(self.projection_head)
        _deactivate_requires_grad(self.momentum_backbone.parameters())
        _deactivate_requires_grad(self.momentum_projection_head.parameters())

    @torch.no_grad()
    def _momentum_update(self, m: 'float'=0.999) ->None:
        """Performs the momentum update for the backbone and projection head."""
        _do_momentum_update(self.momentum_backbone.parameters(), self.backbone.parameters(), m=m)
        _do_momentum_update(self.momentum_projection_head.parameters(), self.projection_head.parameters(), m=m)

    @torch.no_grad()
    def _batch_shuffle(self, batch: 'Tensor') ->Tuple[Tensor, Tensor]:
        """Returns the shuffled batch and the indices to undo."""
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle

    @torch.no_grad()
    def _batch_unshuffle(self, batch: 'Tensor', shuffle: 'Tensor') ->Tensor:
        """Returns the unshuffled batch."""
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]


class BYOL(nn.Module, _MomentumEncoderMixin):
    """Implementation of the BYOL architecture.

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection mlp).
        hidden_dim:
            Dimension of the hidden layer in the projection and prediction mlp.
        out_dim:
            Dimension of the output (after the projection/prediction mlp).
        m:
            Momentum for the momentum update of encoder.
    """

    def __init__(self, backbone: 'nn.Module', num_ftrs: 'int'=2048, hidden_dim: 'int'=4096, out_dim: 'int'=256, m: 'float'=0.9):
        super(BYOL, self).__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = BYOLProjectionHead(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None
        self._init_momentum_encoder()
        self.m = m
        warnings.warn(Warning('The high-level building block BYOL will be deprecated in version 1.3.0. ' + 'Use low-level building blocks instead. ' + 'See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information'), DeprecationWarning)

    def _forward(self, x0: 'torch.Tensor', x1: 'torch.Tensor'=None):
        """Forward pass through the encoder and the momentum encoder.

        Performs the momentum update, extracts features with the backbone and
        applies the projection (and prediciton) head to the output space. If
        x1 is None, only x0 will be processed otherwise, x0 is processed with
        the encoder and x1 with the momentum encoder.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns:
            The output proejction of x0 and (if x1 is not None) the output
            projection of x1.

        Examples:
            >>> # single input, single output
            >>> out = model._forward(x)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model._forward(x0, x1)

        """
        self._momentum_update(self.m)
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)
        if x1 is None:
            return out0
        with torch.no_grad():
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1)
        return out0, out1

    def forward(self, x0: 'torch.Tensor', x1: 'torch.Tensor', return_features: 'bool'=False):
        """Symmetrizes the forward pass (see _forward).

        Performs two forward passes, once where x0 is passed through the encoder
        and x1 through the momentum encoder and once the other way around.

        Note that this model currently requires two inputs for the forward pass
        (x0 and x1) which correspond to the two augmentations.
        Furthermore, `the return_features` argument does not work yet.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns:
            A tuple out0, out1, where out0 and out1 are tuples containing the
            predictions and projections of x0 and x1: out0 = (z0, p0) and
            out1 = (z1, p1).

        Examples:
            >>> # initialize the model and the loss function
            >>> model = BYOL()
            >>> criterion = SymNegCosineSimilarityLoss()
            >>>
            >>> # forward pass for two batches of transformed images x1 and x2
            >>> out0, out1 = model(x0, x1)
            >>> loss = criterion(out0, out1)

        """
        if x0 is None:
            raise ValueError('x0 must not be None!')
        if x1 is None:
            raise ValueError('x1 must not be None!')
        if not all([(s0 == s1) for s0, s1 in zip(x0.shape, x1.shape)]):
            raise ValueError(f'x0 and x1 must have same shape but got shapes {x0.shape} and {x1.shape}!')
        p0, z1 = self._forward(x0, x1)
        p1, z0 = self._forward(x1, x0)
        return (z0, p0), (z1, p1)


class DCLLoss(nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images `i`
    and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is performed
            if weight_fn is None. The function must take the two input tensors
            passed to the forward call as input and return a weight tensor. The
            returned weight tensor must have the same length as the input tensors.
        gather_distributed:
            If True, negatives from all GPUs are gathered before the
            loss calculation.

    Examples:
        >>> loss_fn = DCLLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
        >>>
        >>> # you can also add a custom weighting function
        >>> weight_fn = lambda out0, out1: torch.sum((out0 - out1) ** 2, dim=1)
        >>> loss_fn = DCLLoss(weight_fn=weight_fn)
    """

    def __init__(self, temperature: 'float'=0.1, weight_fn: 'Optional[Callable[[Tensor, Tensor], Tensor]]'=None, gather_distributed: 'bool'=False):
        """Initialzes the DCLoss module.

        Args:
            temperature:
                Similarities are scaled by inverse temperature.
            weight_fn:
                 Weighting function `w` from the paper. Scales the loss between the
                positive views (views from the same image). No weighting is performed
                if weight_fn is None. The function must take the two input tensors
                passed to the forward call as input and return a weight tensor. The
                returned weight tensor must have the same length as the input tensors.
            gather_distributed:
                If True, negatives from all GPUs are gathered before the
                loss calculation.

        Raises:
            ValuesError: If gather_distributed is True but torch.distributed is not available.
        """
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed
        if gather_distributed and not torch_dist.is_available():
            raise ValueError('gather_distributed is True but torch.distributed is not available. Please set gather_distributed=False or install a torch version with distributed support.')

    def forward(self, out0: 'Tensor', out1: 'Tensor') ->Tensor:
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)
        if self.gather_distributed and dist.world_size() > 1:
            out0_all = torch.cat(dist.gather(out0), 0)
            out1_all = torch.cat(dist.gather(out1), 0)
        else:
            out0_all = out0
            out1_all = out1
        loss0 = self._loss(out0, out1, out0_all, out1_all)
        loss1 = self._loss(out1, out0, out1_all, out0_all)
        return 0.5 * (loss0 + loss1)

    def _loss(self, out0, out1, out0_all, out1_all):
        """Calculates DCL loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        This code implements Equation 6 in [0], including the sum over all images `i`
        but with `k` fixed at 0.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape: (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/GPUs. Should be equal to out1 in an
                undistributed setting.
                Shape: (batch_size * world_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        batch_size = out0.shape[0]
        if self.gather_distributed and dist.world_size() > 1:
            diag_mask = dist.eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        sim_00 = torch.einsum('nc,mc->nm', out0, out0_all) / self.temperature
        sim_01 = torch.einsum('nc,mc->nm', out0, out1_all) / self.temperature
        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)
        negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_00 + negative_loss_01).mean()


class DenseCLProjectionHead(ProjectionHead):
    """Projection head for DenseCL [0].

    The projection head consists of a 2-layer MLP. It can be used for global and local
    features.

    - [0]: 2021, DenseCL: https://arxiv.org/abs/2011.09157
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=2048, output_dim: 'int'=128):
        """Initializes the DenseCLProjectionHead with the specified dimensions."""
        super().__init__([(input_dim, hidden_dim, None, nn.ReLU()), (hidden_dim, output_dim, None, None)])


class MemoryBankModule(Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Size of the memory bank as (num_features, dim) tuple. If num_features is 0
            then the memory bank is disabled. Deprecated: If only a single integer is
            passed, it is interpreted as the number of features and the feature
            dimension is inferred from the first batch stored in the memory bank.
            Leaving out the feature dimension might lead to errors in distributed
            training.
        gather_distributed:
            If True then negatives from all gpus are gathered before the memory bank
            is updated. This results in more frequent updates of the memory bank and
            keeps the memory bank contents independent of the number of gpus. But it has
            the drawback that synchronization between processes is required and
            diversity of the memory bank content is reduced.
        feature_dim_first:
            If True, the memory bank returns features with shape (dim, num_features).
            If False, the memory bank returns features with shape (num_features, dim).

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: Tuple[int, int] = (2 ** 16, 128)):
        >>>         super().__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor, labels: Optional[Tensor] = None):
        >>>         output, negatives = super().forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(self, size: 'Union[int, Sequence[int]]'=65536, gather_distributed: 'bool'=False, feature_dim_first: 'bool'=True):
        super().__init__()
        size_tuple = (size,) if isinstance(size, int) else tuple(size)
        if any(x < 0 for x in size_tuple):
            raise ValueError(f'Illegal memory bank size {size}, all entries must be non-negative.')
        self.size = size_tuple
        self.gather_distributed = gather_distributed
        self.feature_dim_first = feature_dim_first
        self.bank: 'Tensor'
        self.register_buffer('bank', tensor=torch.empty(size=size_tuple, dtype=torch.float), persistent=False)
        self.bank_ptr: 'Tensor'
        self.register_buffer('bank_ptr', tensor=torch.empty(1, dtype=torch.long), persistent=False)
        if isinstance(size, int) and size > 0:
            warnings.warn(f"Memory bank size 'size={size}' does not specify feature dimension. It is recommended to set the feature dimension with 'size=(n, dim)' when creating the memory bank. Distributed training might fail if the feature dimension is not set.", UserWarning)
        elif len(size_tuple) > 1:
            self._init_memory_bank(size=size_tuple)

    def forward(self, output: 'Tensor', labels: 'Optional[Tensor]'=None, update: 'bool'=False) ->Tuple[Tensor, Union[Tensor, None]]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.
            update:
                If True, the memory bank will be updated with the current output.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank. Entries from the memory bank have
            shape (dim, num_features) if feature_dim_first is True and
            (num_features, dim) otherwise.

        """
        if self.size[0] == 0:
            return output, None
        if self.bank.ndim == 1:
            dim = output.shape[1:]
            self._init_memory_bank(size=(*self.size, *dim))
        bank = self.bank.clone().detach()
        if self.feature_dim_first:
            bank = bank.transpose(0, -1)
        if update:
            self._dequeue_and_enqueue(output)
        return output, bank

    @torch.no_grad()
    def _init_memory_bank(self, size: 'Tuple[int, ...]') ->None:
        """Initialize the memory bank.

        Args:
            size:
                Size of the memory bank as (num_features, dim) tuple.

        """
        self.bank = torch.randn(size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: 'Tensor') ->None:
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        if self.gather_distributed and dist.world_size() > 1:
            batch = utils.concat_all_gather(batch)
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size[0]:
            self.bank[ptr:] = batch[:self.size[0] - ptr].detach()
            self.bank_ptr.zero_()
        else:
            self.bank[ptr:ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Size of the memory bank as (num_features, dim) tuple. num_features are the
            number of negative samples stored in the memory bank. If num_features is 0,
            the memory bank is disabled. Use 0 for SimCLR. For MoCo we typically use
            numbers like 4096 or 65536.
            Deprecated: If only a single integer is passed, it is interpreted as the
            number of features and the feature dimension is inferred from the first
            batch stored in the memory bank. Leaving out the feature dimension might
            lead to errors in distributed training.
        gather_distributed:
            If True then negatives from all GPUs are gathered before the
            loss calculation. If a memory bank is used and gather_distributed is True,
            then tensors from all gpus are gathered before the memory bank is updated.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:
        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self, temperature: 'float'=0.5, memory_bank_size: 'Union[int, Sequence[int]]'=0, gather_distributed: 'bool'=False):
        """Initializes the NTXentLoss module with the specified parameters.

        Args:
            temperature:
                 Scale logits by the inverse of the temperature.
            memory_bank_size:
                 Size of the memory bank.
            gather_distributed:
                 If True, negatives from all GPUs are gathered before the loss calculation.

        Raises:
            ValueError: If temperature is less than 1e-8 to prevent divide by zero.
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super().__init__(size=memory_bank_size, gather_distributed=gather_distributed)
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.eps = 1e-08
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'.format(self.temperature))
        if gather_distributed and not torch_dist.is_available():
            raise ValueError('gather_distributed is True but torch.distributed is not available. Please set gather_distributed=False or install a torch version with distributed support.')

    def forward(self, out0: 'torch.Tensor', out1: 'torch.Tensor'):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        device = out0.device
        batch_size, _ = out0.shape
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)
        out1, negatives = super(NTXentLoss, self).forward(out1, update=out0.requires_grad)
        if negatives is not None:
            negatives = negatives
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)
            sim_neg = torch.einsum('nc,ck->nk', out0, negatives)
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
        else:
            if self.gather_distributed and dist.world_size() > 1:
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
            logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
            logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
            logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
            logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)
        loss = self.cross_entropy(logits, labels)
        return loss


def cosine_schedule(step: 'int', max_steps: 'int', start_value: 'float', end_value: 'float', period: 'Optional[int]'=None) ->float:
    """Use cosine decay to gradually modify start_value to reach target end_value.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.
        period:
            The number of steps over which the cosine function completes a full cycle.
            Defaults to max_steps.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if max_steps < 1:
        raise ValueError(f'Total step number {max_steps} must be >= 1.')
    if period is None and step > max_steps:
        warnings.warn(f'Current step number {step} exceeds max_steps {max_steps}.', category=RuntimeWarning)
    if period is not None and period <= 0:
        raise ValueError(f'Period {period} must be >= 1')
    decay: 'float'
    if period is not None:
        decay = end_value - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
    elif max_steps == 1:
        decay = end_value
    elif step == max_steps:
        decay = end_value
    else:
        decay = end_value - (end_value - start_value) * (np.cos(np.pi * step / (max_steps - 1)) + 1) / 2
    return decay


def covariance_loss(x: 'Tensor') ->Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x: Tensor with shape (batch_size, ..., dim).

    Returns:
          The computed VICReg covariance loss.
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    cov = torch.einsum('b...c,b...d->...cd', x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    Adapted from the Solo-Learn project:
    https://github.com/vturrisi/solo-learn/blob/b69b4bd27472593919956d9ac58902a301537a4d/solo/utils/misc.py#L187

    """

    @staticmethod
    def forward(ctx: 'FunctionCtx', input: 'torch.Tensor') ->Tuple[torch.Tensor, ...]:
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx: 'FunctionCtx', *grads: torch.Tensor) ->torch.Tensor:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        grad_out = all_gradients[dist.get_rank()]
        return grad_out


def gather(input: 'torch.Tensor') ->Tuple[torch.Tensor]:
    """Gathers a tensor from all processes and supports backpropagation."""
    return GatherLayer.apply(input)


def invariance_loss(x: 'Tensor', y: 'Tensor') ->Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).

    Returns:
        The computed VICReg invariance loss.
    """
    return F.mse_loss(x, y)


def variance_loss(x: 'Tensor', eps: 'float'=0.0001) ->Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.

    Returns:
        The computed VICReg variance loss.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


class VICRegLoss(torch.nn.Module):
    """Implementation of the VICReg loss [0].

    This implementation is based on the code published by the authors [1].

    - [0] VICReg, 2022, https://arxiv.org/abs/2105.04906
    - [1] https://github.com/facebookresearch/vicreg/

    Attributes:
        lambda_param:
            Scaling coefficient for the invariance term of the loss.
        mu_param:
            Scaling coefficient for the variance term of the loss.
        nu_param:
            Scaling coefficient for the covariance term of the loss.
        gather_distributed:
            If True, the cross-correlation matrices from all GPUs are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = VICRegLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(self, lambda_param: 'float'=25.0, mu_param: 'float'=25.0, nu_param: 'float'=1.0, gather_distributed: 'bool'=False, eps=0.0001):
        """Initializes the VICRegLoss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(VICRegLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError('gather_distributed is True but torch.distributed is not available. Please set gather_distributed=False or install a torch version with distributed support.')
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: 'torch.Tensor', z_b: 'torch.Tensor') ->torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).

        Returns:
            The computed VICReg loss.

        Raises:
            AssertionError: If z_a or z_b have a batch size <= 1.
            AssertionError: If z_a and z_b do not have the same shape.
        """
        assert z_a.shape[0] > 1 and z_b.shape[0] > 1, f'z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}'
        assert z_a.shape == z_b.shape, f'z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}.'
        inv_loss = invariance_loss(x=z_a, y=z_b)
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)
        var_loss = 0.5 * (variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps))
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)
        loss = self.lambda_param * inv_loss + self.mu_param * var_loss + self.nu_param * cov_loss
        return loss


def nearest_neighbors(input_maps: 'torch.Tensor', candidate_maps: 'torch.Tensor', distances: 'torch.Tensor', num_matches: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    """Finds the nearest neighbors of the maps in input_maps in candidate_maps.

    Args:
        input_maps:
            A tensor of maps for which to find nearest neighbors.
            It has shape: [batch_size, input_map_size, feature_dimension]
        candidate_maps:
            A tensor of maps to search for nearest neighbors.
            It has shape: [batch_size, candidate_map_size, feature_dimension]
        distances:
            A tensor of distances between the maps in input_maps and candidate_maps.
            It has shape: [batch_size, input_map_size, candidate_map_size]
        num_matches:
            Number of nearest neighbors to return. If num_matches is None or -1,
            all the maps in candidate_maps are considered.

    Returns:
        A tuple of tensors, containing the nearest neighbors in input_maps and candidate_maps.
        They both have shape: [batch_size, input_map_size, feature_dimension]
    """
    if num_matches is None or num_matches == -1 or num_matches > input_maps.size(1):
        num_matches = input_maps.size(1)
    topk_values, topk_indices = distances.topk(k=1, dim=2, largest=False)
    topk_values = topk_values.squeeze(-1)
    _, min_indices = topk_values.topk(k=num_matches, dim=1, largest=False)
    feature_dimension = input_maps.shape[2]
    filtered_input_maps = torch.gather(input_maps, 1, min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension))
    selected_candidate_maps = torch.gather(candidate_maps, 1, topk_indices.expand(-1, -1, feature_dimension))
    filtered_candidate_maps = torch.gather(selected_candidate_maps, 1, min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension))
    return filtered_input_maps, filtered_candidate_maps


class VICRegLLoss(torch.nn.Module):
    """Implementation of the VICRegL loss from VICRegL paper [0].

    This implementation follows the code published by the authors [1].

    - [0]: VICRegL, 2022, https://arxiv.org/abs/2210.01571
    - [1]: https://github.com/facebookresearch/VICRegL

    Attributes:
        lambda_param:
            Coefficient for the invariance term of the loss.
        mu_param:
            Coefficient for the variance term of the loss.
        nu_param:
            Coefficient for the covariance term of the loss.
        alpha:
            Coefficient to weight global with local loss. The final loss is computed as
            (self.alpha * global_loss + (1-self.alpha) * local_loss).
        gather_distributed:
            If True, the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.
        num_matches:
            Number of local features to match using nearest neighbors.

    Examples:
        >>> # initialize loss function
        >>> criterion = VICRegLLoss()
        >>> transform = VICRegLTransform(n_global_views=2, n_local_views=4)
        >>>
        >>> # generate two random transforms of images
        >>> views_and_grids = transform(images)
        >>> views = views_and_grids[:6] # 2 global views + 4 local views
        >>> grids = views_and_grids[6:]
        >>>
        >>> # feed through model images
        >>> features = [model(view) for view in views]
        >>>
        >>> # calculate loss
        >>> loss = criterion(
        ...     global_view_features=features[:2],
        ...     global_view_grids=grids[:2],
        ...     local_view_features=features[2:],
        ...     local_view_grids=grids[2:],
        ... )
    """

    def __init__(self, lambda_param: 'float'=25.0, mu_param: 'float'=25.0, nu_param: 'float'=1.0, alpha: 'float'=0.75, gather_distributed: 'bool'=False, eps: 'float'=0.0001, num_matches: 'Tuple[int, int]'=(20, 4)):
        """Initializes the VICRegL loss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(VICRegLLoss, self).__init__()
        self.alpha = alpha
        self.num_matches = num_matches
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.eps = eps
        self.gather_distributed = gather_distributed
        self.vicreg_loss = VICRegLoss(lambda_param=lambda_param, mu_param=mu_param, nu_param=0.5 * nu_param, eps=eps, gather_distributed=gather_distributed)

    def forward(self, global_view_features: 'Sequence[Tuple[Tensor, Tensor]]', global_view_grids: 'Sequence[Tensor]', local_view_features: 'Optional[Sequence[Tuple[Tensor, Tensor]]]'=None, local_view_grids: 'Optional[Sequence[Tensor]]'=None) ->Tensor:
        """Computes the global and local VICRegL loss from the input features.

        Args:
            global_view_features:
                Sequence of (global_features, local_features) tuples from the global
                crop views. global_features must have size
                (batch_size, global_feature_dim) and local_features must have size
                (batch_size, grid_height, grid_width, local_feature_dim).
            global_view_grids:
                Sequence of grid tensors from the global crop views. Every tensor must
                have shape (batch_size, grid_height, grid_width, 2).
            local_view_features:
                Sequence of (global_features, local_features) tuples from the local crop
                views. global_features must have size
                (batch_size, global_feature_dim) and local_features must have size
                (batch_size, grid_height, grid_width, local_feature_dim). Note that
                grid_height and grid_width can differ between global_view_features and
                local_view_features.
            local_view_grids:
                Sequence of grid tensors from the local crop views. Every tensor must
                have shape (batch_size, grid_height, grid_width, 2). Note that
                grid_height and grid_width can differ between global_view_features and
                local_view_features.

        Returns:
            Weighted sum of the global and local loss, calculated as:
            `self.alpha * global_loss + (1-self.alpha) * local_loss`.

        Raises:
            ValueError: If the lengths of global_view_features and global_view_grids are not the same.
            ValueError: If the lengths of local_view_features and local_view_grids are not the same.
            ValueError: If only one of local_view_features or local_view_grids is set.
        """
        if len(global_view_features) != len(global_view_grids):
            raise ValueError(f'global_view_features and global_view_grids must have same length but found {len(global_view_features)} and {len(global_view_grids)}.')
        if local_view_features is not None and local_view_grids is not None:
            if len(local_view_features) != len(local_view_grids):
                raise ValueError(f'local_view_features and local_view_grids must have same length but found {len(local_view_features)} and {len(local_view_grids)}.')
        elif local_view_features is not None or local_view_grids is not None:
            raise ValueError(f'local_view_features and local_view_grids must either both be set or None but found {type(local_view_features)} and {type(local_view_grids)}.')
        global_loss = self._global_loss(global_view_features=global_view_features, local_view_features=local_view_features)
        local_loss = self._local_loss(global_view_features=global_view_features, global_view_grids=global_view_grids, local_view_features=local_view_features, local_view_grids=local_view_grids)
        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss
        return loss

    def _global_loss(self, global_view_features: 'Sequence[Tuple[Tensor, Tensor]]', local_view_features: 'Optional[Sequence[Tuple[Tensor, Tensor]]]'=None) ->Tensor:
        """Returns global features loss.

        Args:
        global_view_features:
                Sequence of (global_features, local_features)
                tuples from the global crop views.
        local_view_features:
                Sequence of (global_features,local_features)
                tuples from the local crop views.

        Returns:
            The computed global features loss.
        """
        inv_loss = self._global_invariance_loss(global_view_features=global_view_features, local_view_features=local_view_features)
        var_loss, cov_loss = self._global_variance_and_covariance_loss(global_view_features=global_view_features, local_view_features=local_view_features)
        return self.lambda_param * inv_loss + self.mu_param * var_loss + self.nu_param * cov_loss

    def _global_invariance_loss(self, global_view_features: 'Sequence[Tuple[Tensor, Tensor]]', local_view_features: 'Optional[Sequence[Tuple[Tensor, Tensor]]]'=None) ->Tensor:
        """Returns invariance loss from global features.

        Args:
            global_view_features:
                        Sequence of (global_features, local_features)
                        tuples from the global crop views.
            local_view_features:
                        Sequence of (global_features,local_features)
                        tuples from the local crop views.

        Returns:
            The computed invariance loss from global features.
        """
        loss = 0
        loss_count = 0
        for global_features_a, _ in global_view_features:
            for global_features_b, _ in global_view_features:
                if global_features_a is not global_features_b:
                    loss += invariance_loss(global_features_a, global_features_b)
                    loss_count += 1
            if local_view_features is not None:
                for global_features_b, _ in local_view_features:
                    loss += invariance_loss(global_features_a, global_features_b)
                    loss_count += 1
        return loss / loss_count

    def _global_variance_and_covariance_loss(self, global_view_features: 'Sequence[Tuple[Tensor, Tensor]]', local_view_features: 'Optional[Sequence[Tuple[Tensor, Tensor]]]'=None) ->Tuple[Tensor, Tensor]:
        """Returns variance and covariance loss from global features.

        Args:
            global_view_features: Sequence of (global_features, local_features)
                    tuples from the global crop views.
            local_view_features: Sequence of (global_features,local_features)
                    tuples from the local crop views.

        Returns:
            The computed variance and covariance loss from global features.
        """
        view_features = list(global_view_features)
        if local_view_features is not None:
            view_features = view_features + list(local_view_features)
        var_loss = 0
        cov_loss = 0
        loss_count = 0
        for global_features, _ in view_features:
            if self.gather_distributed and dist.is_initialized():
                world_size = dist.get_world_size()
                if world_size > 1:
                    global_features = torch.cat(gather(global_features), dim=0)
            var_loss += variance_loss(x=global_features, eps=self.eps)
            cov_loss += covariance_loss(x=global_features)
            loss_count += 1
        return var_loss / loss_count, cov_loss / loss_count

    def _local_loss(self, global_view_features: 'Sequence[Tuple[Tensor, Tensor]]', global_view_grids: 'Sequence[Tensor]', local_view_features: 'Optional[Sequence[Tuple[Tensor, Tensor]]]'=None, local_view_grids: 'Optional[Sequence[Tensor]]'=None) ->Tensor:
        """Returns loss from local features based on nearest neighbor matching.

        Note: Our nearest neighbor implementation returns the selected features sorted
        by increasing matching distance, whereas the implementation by the VICRegL
        authors returns features in a different order [1]. This results in slight
        differences of the final local loss. The difference results from feature
        centering which depends on the order.

        Note: Nearest neighbor matching slightly differs between the paper [0] and the
        original implementation of the authors [1]. The paper mentions that
        num_matches is set to 20 for global views and 4 for local views. The code
        uses 20 matches for the first NN search and 4 matches for the second search,
        regardless of global or local views:
        https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L329-L334
        Our implementation follows the original code and ignores view type.

        Args:
            global_view_features:
                Sequence of (global_features, local_features) tuples from the global crop views.
            global_view_grids:
                Sequence of grid tensors from the global crop views.
            local_view_features:
                Sequence of (global_features,local_features) tuples from the local crop views.
            local_view_grids:
                Sequence of grid tensors from the local crop views.

        Returns:
            The computed loss from local features based on nearest neighbor matching.
        """
        loss = 0
        loss_count = 0
        for (_, z_a_local_features), grid_a in zip(global_view_features, global_view_grids):
            for (_, z_b_local_features), grid_b in zip(global_view_features, global_view_grids):
                if z_a_local_features is not z_b_local_features:
                    loss += self._local_l2_loss(z_a=z_a_local_features, z_b=z_b_local_features)
                    loss += self._local_location_loss(z_a=z_a_local_features, z_b=z_b_local_features, grid_a=grid_a, grid_b=grid_b)
                    loss_count += 1
            if local_view_features is not None and local_view_grids is not None:
                for (_, z_b_local_features), grid_b in zip(local_view_features, local_view_grids):
                    loss += self._local_l2_loss(z_a=z_a_local_features, z_b=z_b_local_features)
                    loss += self._local_location_loss(z_a=z_a_local_features, z_b=z_b_local_features, grid_a=grid_a, grid_b=grid_b)
                    loss_count += 1
        return loss / loss_count

    def _local_l2_loss(self, z_a: 'Tensor', z_b: 'Tensor') ->Tensor:
        """Returns loss for local features matched with neareast neighbors using L2
        distance in the feature space.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, height, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, height, width, dim).

        Returns:
            The computed loss for local features.
        """
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(input_features=z_a, candidate_features=z_b, num_matches=self.num_matches[0])
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(input_features=z_b, candidate_features=z_a, num_matches=self.num_matches[1])
        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

    def _local_location_loss(self, z_a: 'Tensor', z_b: 'Tensor', grid_a: 'Tensor', grid_b: 'Tensor') ->Tensor:
        """Returns loss for local features matched with nearest neighbors based on
        the feature location.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, height, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, height, width, dim).
                Note that height and width can be different from z_a.
            grid_a:
                Grid tensor with shape (batch_size, height, width, 2).
            grid_b:
                Grid tensor with shape (batch_size, height, width, 2).
                Note that height and width can be different from grid_a.

        Returns:
            The computed loss for local features based on nearest neighbour matching.
        """
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)
        grid_a = grid_a.flatten(start_dim=1, end_dim=2)
        grid_b = grid_b.flatten(start_dim=1, end_dim=2)
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_grid(input_features=z_a, candidate_features=z_b, input_grid=grid_a, candidate_grid=grid_b, num_matches=self.num_matches[0])
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_grid(input_features=z_b, candidate_features=z_a, input_grid=grid_b, candidate_grid=grid_a, num_matches=self.num_matches[1])
        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

    def _nearest_neighbors_on_l2(self, input_features: 'Tensor', candidate_features: 'Tensor', num_matches: 'int') ->Tuple[Tensor, Tensor]:
        """Finds num_matches closest neighbors of input_features in candidate_features.

        Args:
            input_features:
                Local features tensor with shape (batch_size, height * width, dim).
            candidate_features:
                Local features tensor with shape (batch_size, height * width, dim).
                Note that height and width can be different from input_features.

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (batch_size, num_matches, dim).
        """
        distances = torch.cdist(input_features, candidate_features)
        return nearest_neighbors(input_features, candidate_features, distances, num_matches)

    def _nearest_neighbors_on_grid(self, input_features: 'Tensor', candidate_features: 'Tensor', input_grid: 'Tensor', candidate_grid: 'Tensor', num_matches: 'int') ->Tuple[Tensor, Tensor]:
        """Finds num_matches closest neighbors of input_features in candidate_features
        based on the distance between the features defined by input_grid and
        candidate_grid.

        Args:
            input_features:
                Local features tensor with shape (batch_size, height * width, dim).
            candidate_features:
                Local features tensor with shape (batch_size, height * width, dim).
                Note that height and width can be different from input_features.
            input_grid:
                Grid tensor with shape (batch_size, height, width, 2).
            candidate_grid:
                Grid tensor with shape (batch_size, height, width, 2). Note that height
                and width can be different from input_grid.

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (batch_size, num_matches, dim).
        """
        distances = torch.cdist(input_grid, candidate_grid)
        return nearest_neighbors(input_features, candidate_features, distances, num_matches)


class VicRegLLocalProjectionHead(ProjectionHead):
    """Projection head used for the local head of VICRegL.

    "The projector network has three linear layers. The first two layers of the projector
    are followed by a batch normalization layer and rectified linear units." [0]

    - [0]: 2022, VICRegL, https://arxiv.org/abs/2210.01571
    """

    def __init__(self, input_dim: 'int'=2048, hidden_dim: 'int'=8192, output_dim: 'int'=8192):
        """Initializes the VicRegLLocalProjectionHead with the specified dimensions."""
        super(VicRegLLocalProjectionHead, self).__init__([(input_dim, hidden_dim, nn.LayerNorm(hidden_dim), nn.ReLU()), (hidden_dim, hidden_dim, nn.LayerNorm(hidden_dim), nn.ReLU()), (hidden_dim, output_dim, None, None)])

