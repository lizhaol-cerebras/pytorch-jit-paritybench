
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


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


from typing import Optional


from typing import Sequence


from typing import Union


from typing import Any


import torch


from torch import Tensor


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import RandomRotation


import numpy as np


from typing import List


from torchvision.transforms.transforms import Compose


from torch import nn


from torch.utils.data import DataLoader


from torchvision.models import mobilenet_v2


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomCrop


from torchvision.transforms import CenterCrop


from torchvision.transforms import Resize


import copy


from typing import Literal


from torchvision import transforms


import math


from typing import Tuple


from torchvision.datasets.folder import default_loader


from warnings import warn


from abc import abstractmethod


from abc import ABC


from typing import TypeVar


from torch.utils.data.dataset import Dataset


from torchvision.datasets.utils import download_and_extract_archive


from torchvision.datasets.utils import extract_archive


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import check_integrity


from torch.utils.data import Dataset


from typing import Dict


from typing import Iterator


from torchvision.datasets.folder import ImageFolder


from torchvision.datasets.utils import verify_str_arg


from typing import Set


import logging


from typing import TypedDict


import random


from typing import Callable


from typing import Generator


from typing import Generic


from typing import Iterable


import re


from typing import Mapping


from typing import TYPE_CHECKING


from collections import OrderedDict


from typing import SupportsInt


from torch.nn import Module


from functools import partial


from typing import Protocol


import warnings


from torch.distributions.categorical import Categorical


from torch.utils.data import Sampler


from copy import copy


from torch.utils.data.dataset import Subset


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data.dataset import TensorDataset


from typing import overload


import itertools


from collections import defaultdict


from torch.utils.data.dataloader import default_collate


from torch.utils.data import Dataset as TorchDataset


from typing import Sized


from torch.utils.data import DistributedSampler


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


from torch.utils.data import ConcatDataset


from collections import deque


from torch.utils.data import Subset


from torchvision.datasets.vision import StandardTransform


from typing import final


from numpy import ndarray


import torch.utils.data as data


from torchvision.transforms.functional import crop


from queue import Queue


from typing import Type


import torch as ch


from typing import NamedTuple


from torchvision.transforms import ToTensor as ToTensorTV


from torchvision.transforms import PILToTensor as PILToTensorTV


from torchvision.transforms import Normalize as NormalizeTV


from torchvision.transforms import ConvertImageDtype as ConvertTV


from torchvision.transforms import RandomResizedCrop as RandomResizedCropTV


from torchvision.transforms import CenterCrop as CenterCropTV


from torchvision.transforms import RandomHorizontalFlip as RandomHorizontalFlipTV


from torchvision.transforms import RandomCrop as RandomCropTV


from typing import BinaryIO


from typing import IO


from typing import Collection


from typing import ContextManager


from torch.nn.modules import Module


from torch.nn.parallel import DistributedDataParallel


from torch.distributed import init_process_group


from torch.distributed import broadcast_object_list


from matplotlib.figure import Figure


from enum import Enum


import matplotlib.pyplot as plt


from matplotlib.axes import Axes


from numpy import arange


from torch.nn.functional import pad


import torch.distributed as dist


from torchvision.utils import make_grid


from matplotlib.pyplot import subplots


from torch import arange


from torch.utils.tensorboard import SummaryWriter


from matplotlib.pyplot import Figure


from torchvision.transforms.functional import to_tensor


from typing import TextIO


from numpy import array


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from torch import sigmoid


from torch.nn.functional import mse_loss


from torch.nn.functional import softmax


import torchvision.models as models


from matplotlib import transforms


from torch.nn import Sequential


from torch.nn import BatchNorm2d


from torch.nn import Conv2d


from torch.nn import ReLU


from torch.nn import ConstantPad3d


from torch.nn import Identity


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Linear


from torch.nn.init import zeros_


from torch.nn.init import kaiming_normal_


from torch.nn.modules.flatten import Flatten


import typing as t


from torch.nn import functional as F


from torch.nn.functional import relu


from torch.nn.functional import avg_pool2d


import torch.utils.checkpoint


from torch import default_generator


from torch.nn import BCELoss


from torch.optim.lr_scheduler import MultiStepLR


from torch.nn.functional import normalize


from torch.optim import SGD


from torchvision.models.feature_extraction import get_graph_node_names


from torchvision.models.feature_extraction import create_feature_extractor


from torch.nn.modules.batchnorm import _NormBase


import collections


from numpy import inf


from torch import cat


from torch.nn import CrossEntropyLoss


from torch.optim import Optimizer


from math import ceil


from torch.nn.parameter import Parameter


from torchvision.transforms import Lambda


from typing import OrderedDict


import functools


import inspect


from torch.optim.optimizer import Optimizer


import torchvision


from torchvision.datasets import MNIST


import torch.optim.lr_scheduler


from torch.optim import Adam


from torch.utils.data import random_split


from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from torch.utils.data import TensorDataset


from sklearn.datasets import make_classification


from sklearn.model_selection import train_test_split


import time


from matplotlib import pyplot as plt


from torchvision.transforms.functional import to_pil_image


from torch.utils.data.sampler import SequentialSampler


import torch.nn


from torchvision.datasets import CIFAR10


import torch.utils.data


import torchvision.models.detection.mask_rcnn


from itertools import repeat


from itertools import chain


from torch.utils.model_zoo import tqdm


import torchvision.models.detection


from torchvision.transforms import functional as F


from torchvision.transforms import transforms as T


from torch.optim.lr_scheduler import ExponentialLR


from torch.utils.data.sampler import SubsetRandomSampler


from numpy.testing import assert_almost_equal


import torch.distributed as dst


from types import SimpleNamespace


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch import tensor


from torch import zeros


from sklearn.datasets import make_blobs


class ScaleFrom_0_255_To_0_1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        default_float_dtype = torch.get_default_dtype()
        return input.div(255)


class BatchRenorm2D(Module):

    def __init__(self, num_features, gamma=None, beta=None, running_mean=None, running_var=None, eps=1e-05, momentum=0.01, r_d_max_inc_step=0.0001, r_max=1.0, d_max=0.0, max_r_max=3.0, max_d_max=5.0):
        super(BatchRenorm2D, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.momentum = torch.tensor(momentum, requires_grad=False)
        if gamma is None:
            self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        else:
            self.gamma = torch.nn.Parameter(gamma.view(1, -1, 1, 1))
        if beta is None:
            self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)
        else:
            self.beta = torch.nn.Parameter(beta.view(1, -1, 1, 1))
        if running_mean is None:
            self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
            self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)
        else:
            self.running_avg_mean = running_mean.view(1, -1, 1, 1)
            self.running_avg_std = torch.sqrt(running_var.view(1, -1, 1, 1))
        self.max_r_max = max_r_max
        self.max_d_max = max_d_max
        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step
        self.r_max = r_max
        self.d_max = d_max

    def forward(self, x):
        device = self.gamma.device
        self.r_max = self.r_max if isinstance(self.r_max, float) else self.r_max
        self.d_max = self.d_max if isinstance(self.d_max, float) else self.d_max
        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        batch_ch_std = torch.sqrt(torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False) + self.eps)
        batch_ch_std = batch_ch_std
        self.running_avg_std = self.running_avg_std
        self.running_avg_mean = self.running_avg_mean
        self.momentum = self.momentum
        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data
            x = (x - batch_ch_mean) * r / batch_ch_std + d
            x = self.gamma * x + self.beta
            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]
            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]
            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data - self.running_avg_std)
        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta
        return x


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters

    Bias layers used in Bias Correction (BiC) plugin.
    "Wu, Yue, et al. "Large scale incremental learning." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition. 2019"
    """

    def __init__(self, clss: 'Iterable[SupportsInt]'):
        """
        :param clss: list of classes of the current layer. This are use
            to identify the columns which are multiplied by the Bias
            correction Layer.
        """
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
        unique_classes = list(sorted(set(int(x) for x in clss)))
        self.register_buffer('clss', torch.tensor(unique_classes, dtype=torch.long))

    def forward(self, x):
        alpha = torch.ones_like(x)
        beta = torch.zeros_like(x)
        alpha[:, self.clss] = self.alpha
        beta[:, self.clss] = self.beta
        return alpha * x + beta


class CosineLinear(nn.Module):
    """
    Cosine layer defined in
    "Learning a Unified Classifier Incrementally via Rebalancing"
    by Saihui Hou et al.

    Implementation modified from https://github.com/G-U-N/PyCIL

    This layer is aimed at countering the task-recency bias by removing the bias
    in the classifier and normalizing the weight and the input feature before
    computing the weight-feature product
    """

    def __init__(self, in_features, out_features, sigma=True):
        """
        :param in_features: number of input features
        :param out_features: number of classes
        :param sigma: learnable output scaling factor
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(nn.Module):
    """
    This class keeps two Cosine Linear layers, without sigma scaling,
    and handles the sigma parameter that is common for the two of them.
    One CosineLinear is for the old classes and the other
    one is for the new classes
    """

    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)
        if self.sigma is not None:
            out = self.sigma * out
        return out

