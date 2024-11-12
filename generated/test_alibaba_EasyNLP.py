
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


import time


import torch


import torch.nn as nn


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from typing import Union


from typing import List


from torch.utils.data import Dataset


import torch.cuda


from math import factorial


from torchvision import transforms


from typing import Optional


from typing import Dict


from typing import Any


from typing import Tuple


import torch.nn.functional as F


import abc


from typing import Callable


from torchvision.utils import save_image


import random


from torchvision.io import read_image


from scipy import ndimage


import logging


import math


import torch.utils.checkpoint


import re


from torch import nn


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import roc_auc_score


from sklearn.metrics import classification_report


from sklearn.metrics import f1_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from torch import Tensor


import uuid


from typing import Iterable


from typing import Mapping


from typing import Type


import torch.distributed as dist


from collections import OrderedDict


import warnings


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from itertools import count


import pandas as pd


import copy


from itertools import accumulate


import functools


import collections


import torch.utils.data.dataloader as DataLoader


import torch.utils.data


from torch.utils.data.dataloader import default_collate


import string


from torch.nn.utils import clip_grad_norm_


from torch.optim import Adam


from torch.optim import Optimizer


from torch.optim.optimizer import required


from torch.optim.lr_scheduler import LambdaLR


from scipy.special import log_softmax


import types


from collections import UserDict


from enum import Enum


from functools import partial


from functools import wraps


from types import ModuleType


from typing import BinaryIO


from uuid import uuid4


from abc import ABC


from abc import abstractmethod


import inspect


from copy import deepcopy


from scipy.stats import poisson


from queue import Empty


from collections import defaultdict


from torch.utils import data


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch.multiprocessing import Lock


from torch.optim.lr_scheduler import _LRScheduler


import torch.distributed


from collections import namedtuple


import itertools


from torch import distributed as dist


from typing import Set


from torch import device


from torch import dtype


from torch.nn import CrossEntropyLoss


from torch.nn import functional as F


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


from typing import TYPE_CHECKING


from torch.nn import LayerNorm


from torch.nn import init


from torch.nn import Linear


from math import ceil


from inspect import isfunction


from torch import einsum


from torchvision.utils import make_grid


import torch as th


from collections import abc


from queue import Queue


from torch.nn.modules import Module


from torch.nn.parallel.distributed import DistributedDataParallel as DDP


import torch.nn


import torch.nn.init as init


from torch.utils.checkpoint import checkpoint


from typing import TypeVar


from typing import Generator


from torch.nn.modules.batchnorm import _BatchNorm


import torch.utils.checkpoint as torch_checkpoint


from logging import Filter


from logging.handlers import QueueHandler


from logging.handlers import QueueListener


from torch.multiprocessing import Queue


import torch.multiprocessing as mp


from torch.utils.data.sampler import SequentialSampler


from typing import NamedTuple


from typing import Sequence


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from functools import reduce


from math import sqrt


from abc import abstractstaticmethod


from torch.nn.utils.rnn import pad_sequence


from time import time


from typing import cast


from itertools import repeat


from numbers import Number


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import LinearLR


from logging import CRITICAL


from logging import DEBUG


from logging import ERROR


from logging import FATAL


from logging import INFO


from logging import NOTSET


from logging import WARN


from logging import WARNING


from typing import MutableMapping


from torch.utils.data import TensorDataset


from typing import Iterator


from torch.utils.data import DistributedSampler


from itertools import chain


from collections import Counter


from torch.utils.data import ConcatDataset


from typing import *


from sklearn.metrics import average_precision_score


from torchvision.transforms import RandomResizedCrop


from torch.utils.data import SubsetRandomSampler


import numpy


import torchvision.datasets as datasets


from time import gmtime


from time import strftime


from torch import optim


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torchvision.transforms.functional import InterpolationMode


from torchvision.datasets.utils import download_url


from collections import deque


from math import *


from torch.hub import download_url_to_file


from torch.hub import get_dir


from sklearn.metrics.pairwise import cosine_similarity


from re import S


import scipy.spatial.distance as distance


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Dataset


from types import MethodType


from torch.autograd import Function


from collections import defaultdict as ddict


import pandas


from numpy.random import choice


from torch.utils.data import Dataset as DS


from torch.nn import Embedding


from torch.optim.optimizer import Optimizer


from numpy.random import randint


from torch.utils import data as torch_data


from torch.nn import functional


from torch import cuda


from itertools import combinations


from sklearn.metrics import accuracy_score


import torchvision.transforms as transforms


from torchvision import datasets


import torch.optim as optim


import torchvision


from torch.utils.data import Dataset as TorchDataset


import torchvision.transforms as T


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class SmallEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicMotionEncoder(nn.Module):

    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):

    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


class AlternateCorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]
        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))
        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class CorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class RaftArgs:

    def __init__(self):
        pass

    def __iter__(self):
        return self.__dict__.__iter__()


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = 8 * flow.shape[2], 8 * flow.shape[3]
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class RAFT(nn.Module):

    def __init__(self, small=False, mixed_precision=False, alternate_corr=False):
        super(RAFT, self).__init__()
        args = RaftArgs()
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = alternate_corr
        self.args = args
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions

    @staticmethod
    def from_pretrained(model_path, **kwargs):
        model = torch.nn.DataParallel(RAFT(*kwargs))
        model.load_state_dict(torch.load(model_path))
        model = model.module
        model.eval()
        return model


class OLSSSchedulerModel(torch.nn.Module):

    def __init__(self, wx, we):
        super(OLSSSchedulerModel, self).__init__()
        assert len(wx.shape) == 1 and len(we.shape) == 2
        T = wx.shape[0]
        assert T == we.shape[0] and T == we.shape[1]
        self.register_parameter('wx', torch.nn.Parameter(wx))
        self.register_parameter('we', torch.nn.Parameter(we))

    def forward(self, t, xT, e_prev):
        assert t - len(e_prev) + 1 == 0
        x = xT * self.wx[t]
        for e, we in zip(e_prev, self.we[t]):
            x += e * we
        return x


CONFIG_NAME = 'wrapper_config.json'


_default_endpoint = 'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo'


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def push_to_hub(self, repo_path_or_name: 'Optional[str]'=None, repo_url: 'Optional[str]'=None, use_temp_dir: 'bool'=False, commit_message: 'Optional[str]'=None, organization: 'Optional[str]'=None, private: 'Optional[bool]'=None, use_auth_token: 'Optional[Union[bool, str]]'=None) ->str:
        """
        Upload model checkpoint or tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
        :obj:`repo_path_or_name`.

        Parameters:
            repo_path_or_name (:obj:`str`, `optional`):
                Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
                which case the repository will have the name of that local folder). If not specified, will default to
                the name given by :obj:`repo_url` and a local directory with that name will be created.
            repo_url (:obj:`str`, `optional`):
                Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
                repository will be created in your namespace (unless you specify an :obj:`organization`) with
                :obj:`repo_name`.
            use_temp_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clone the distant repo in a temporary directory or in :obj:`repo_path_or_name` inside
                the current working directory. This will slow things down if you are making changes in an existing repo
                since you will need to clone the repo before every push.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
                :obj:`"add model"` depending on the type of the class.
            organization (:obj:`str`, `optional`):
                Organization in which you want to push your model or tokenizer (you must be a member of this
                organization).
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
                :obj:`True` if :obj:`repo_url` is not specified.


        Returns:
            The url of the commit of your model in the given repository.
        """
        if use_temp_dir:
            if repo_url is None:
                if use_auth_token is None:
                    use_auth_token = True
                repo_name = Path(repo_path_or_name).name
                repo_url = self._get_repo_url_from_name(repo_name, organization=organization, private=private, use_auth_token=use_auth_token)
            repo_path_or_name = tempfile.mkdtemp()
        repo = self._create_or_get_repo(repo_path_or_name=repo_path_or_name, repo_url=repo_url, organization=organization, private=private, use_auth_token=use_auth_token)
        self.save_pretrained(repo_path_or_name)
        url = self._push_to_hub(repo, commit_message=commit_message)
        if use_temp_dir:
            shutil.rmtree(repo_path_or_name)
        return url

    @staticmethod
    def _get_repo_url_from_name(repo_name: 'str', organization: 'Optional[str]'=None, private: 'bool'=None, use_auth_token: 'Optional[Union[bool, str]]'=None) ->str:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        elif use_auth_token:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError('You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own token as the `use_auth_token` argument.')
        else:
            token = None
        return HfApi(endpoint=HUGGINGFACE_CO_RESOLVE_ENDPOINT).create_repo(token, repo_name, organization=organization, private=private, repo_type=None, exist_ok=True)

    @classmethod
    def _create_or_get_repo(cls, repo_path_or_name: 'Optional[str]'=None, repo_url: 'Optional[str]'=None, organization: 'Optional[str]'=None, private: 'bool'=None, use_auth_token: 'Optional[Union[bool, str]]'=None):
        repo = None
        raise Exception('creat/get repo not supported yet.')
        return repo

    @classmethod
    def _push_to_hub(cls, repo, commit_message):
        raise Exception('_push_to_hub not supported yet.')


ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}


def is_offline_mode():
    return _is_offline_mode


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')


class PretrainedConfig(PushToHubMixin):
    """
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    Note:
        A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
        initialize a model does **not** load the model weights. It only affects the model's configuration.

    Class attributes (overridden by derived classes)

        - **model_type** (:obj:`str`) -- An identifier for the model type, serialized into the JSON file, and used to
          recreate the correct object in :class:`~transformers.AutoConfig`.
        - **is_composition** (:obj:`bool`) -- Whether the config class is composed of multiple sub-configs. In this
          case the config has to be initialized from two or more configs of type
          :class:`~transformers.PretrainedConfig` like: :class:`~transformers.EncoderDecoderConfig` or
          :class:`~RagConfig`.
        - **keys_to_ignore_at_inference** (:obj:`List[str]`) -- A list of keys to ignore by default when looking at
          dictionary outputs of the model during inference.

    Common attributes (present in all subclasses)

        - **vocab_size** (:obj:`int`) -- The number of tokens in the vocabulary, which is also the first dimension of
          the embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
        - **hidden_size** (:obj:`int`) -- The hidden size of the model.
        - **num_attention_heads** (:obj:`int`) -- The number of attention heads used in the multi-head attention layers
          of the model.
        - **num_hidden_layers** (:obj:`int`) -- The number of blocks in the model.

    Args:
        name_or_path (:obj:`str`, `optional`, defaults to :obj:`""`):
            Store the string that was passed to :func:`~transformers.PreTrainedModel.from_pretrained` or
            :func:`~transformers.TFPreTrainedModel.from_pretrained` as ``pretrained_model_name_or_path`` if the
            configuration was created with such a method.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return all hidden-states.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should returns all attentions.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return a :class:`~transformers.file_utils.ModelOutput` instead of a plain
            tuple.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
        tie_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (:obj:`Dict[int, List[int]]`, `optional`, defaults to :obj:`{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance ``{1: [0, 2], 2: [2, 3]}`` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (:obj:`int`, `optional`, defaults to :obj:`0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of :obj:`0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            :obj:`n` < sequence_length embeddings at a time. For more information on feed forward chunking, see `How
            does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .

    Parameters for sequence generation

        - **max_length** (:obj:`int`, `optional`, defaults to 20) -- Maximum length that will be used by default in the
          :obj:`generate` method of the model.
        - **min_length** (:obj:`int`, `optional`, defaults to 10) -- Minimum length that will be used by default in the
          :obj:`generate` method of the model.
        - **do_sample** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default in the
          :obj:`generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
        - **early_stopping** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default
          in the :obj:`generate` method of the model. Whether to stop the beam search when at least ``num_beams``
          sentences are finished per batch or not.
        - **num_beams** (:obj:`int`, `optional`, defaults to 1) -- Number of beams for beam search that will be used by
          default in the :obj:`generate` method of the model. 1 means no beam search.
        - **num_beam_groups** (:obj:`int`, `optional`, defaults to 1) -- Number of groups to divide :obj:`num_beams`
          into in order to ensure diversity among different groups of beams that will be used by default in the
          :obj:`generate` method of the model. 1 means no group beam search.
        - **diversity_penalty** (:obj:`float`, `optional`, defaults to 0.0) -- Value to control diversity for group
          beam search. that will be used by default in the :obj:`generate` method of the model. 0 means no diversity
          penalty. The higher the penalty, the more diverse are the outputs.
        - **temperature** (:obj:`float`, `optional`, defaults to 1) -- The value used to module the next token
          probabilities that will be used by default in the :obj:`generate` method of the model. Must be strictly
          positive.
        - **top_k** (:obj:`int`, `optional`, defaults to 50) -- Number of highest probability vocabulary tokens to keep
          for top-k-filtering that will be used by default in the :obj:`generate` method of the model.
        - **top_p** (:obj:`float`, `optional`, defaults to 1) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``top_p``. If set to float < 1, only the most probable tokens with
          probabilities that add up to ``top_p`` or higher are kept for generation.
        - **repetition_penalty** (:obj:`float`, `optional`, defaults to 1) -- Parameter for repetition penalty that
          will be used by default in the :obj:`generate` method of the model. 1.0 means no penalty.
        - **length_penalty** (:obj:`float`, `optional`, defaults to 1) -- Exponential penalty to the length that will
          be used by default in the :obj:`generate` method of the model.
        - **no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``no_repeat_ngram_size``. If set to int > 0, all ngrams of that size
          can only occur once.
        - **encoder_no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by
          default in the :obj:`generate` method of the model for ``encoder_no_repeat_ngram_size``. If set to int > 0,
          all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the ``decoder_input_ids``.
        - **bad_words_ids** (:obj:`List[int]`, `optional`) -- List of token ids that are not allowed to be generated
          that will be used by default in the :obj:`generate` method of the model. In order to get the tokens of the
          words that should not appear in the generated text, use :obj:`tokenizer.encode(bad_word,
          add_prefix_space=True)`.
        - **num_return_sequences** (:obj:`int`, `optional`, defaults to 1) -- Number of independently computed returned
          sequences for each element in the batch that will be used by default in the :obj:`generate` method of the
          model.
        - **output_scores** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should return the
          logits when used for generation
        - **return_dict_in_generate** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should
          return a :class:`~transformers.file_utils.ModelOutput` instead of a :obj:`torch.LongTensor`
        - **forced_bos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the first generated token
          after the :obj:`decoder_start_token_id`. Useful for multilingual models like :doc:`mBART
          <../model_doc/mbart>` where the first generated token needs to be the target language token.
        - **forced_eos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the last generated token
          when :obj:`max_length` is reached.
        - **remove_invalid_values** (:obj:`bool`, `optional`) -- Whether to remove possible `nan` and `inf` outputs of
          the model to prevent the generation method to crash. Note that using ``remove_invalid_values`` can slow down
          generation.


    Parameters for fine-tuning tasks

        - **architectures** (:obj:`List[str]`, `optional`) -- Model architectures that can be used with the model
          pretrained weights.
        - **finetuning_task** (:obj:`str`, `optional`) -- Name of the task used to fine-tune the model. This can be
          used when converting from an original (TensorFlow or PyTorch) checkpoint.
        - **id2label** (:obj:`Dict[int, str]`, `optional`) -- A map from index (for instance prediction index, or
          target index) to label.
        - **label2id** (:obj:`Dict[str, int]`, `optional`) -- A map from label to index for the model.
        - **num_labels** (:obj:`int`, `optional`) -- Number of labels to use in the last layer added to the model,
          typically for a classification task.
        - **task_specific_params** (:obj:`Dict[str, Any]`, `optional`) -- Additional keyword arguments to store for the
          current task.
        - **problem_type** (:obj:`str`, `optional`) -- Problem type for :obj:`XxxForSequenceClassification` models. Can
          be one of (:obj:`"regression"`, :obj:`"single_label_classification"`, :obj:`"multi_label_classification"`).
          Please note that this parameter is only available in the following models: `AlbertForSequenceClassification`,
          `BertForSequenceClassification`, `BigBirdForSequenceClassification`, `ConvBertForSequenceClassification`,
          `DistilBertForSequenceClassification`, `ElectraForSequenceClassification`, `FunnelForSequenceClassification`,
          `LongformerForSequenceClassification`, `MobileBertForSequenceClassification`,
          `ReformerForSequenceClassification`, `RobertaForSequenceClassification`,
          `SqueezeBertForSequenceClassification`, `XLMForSequenceClassification` and `XLNetForSequenceClassification`.

    Parameters linked to the tokenizer

        - **tokenizer_class** (:obj:`str`, `optional`) -- The name of the associated tokenizer class to use (if none is
          set, will use the tokenizer associated to the model by default).
        - **prefix** (:obj:`str`, `optional`) -- A specific prompt that should be added at the beginning of each text
          before calling the model.
        - **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
        - **pad_token_id** (:obj:`int`, `optional`)) -- The id of the `padding` token.
        - **eos_token_id** (:obj:`int`, `optional`)) -- The id of the `end-of-stream` token.
        - **decoder_start_token_id** (:obj:`int`, `optional`)) -- If an encoder-decoder model starts decoding with a
          different token than `bos`, the id of that token.
        - **sep_token_id** (:obj:`int`, `optional`)) -- The id of the `separation` token.

    PyTorch specific parameters

        - **torchscript** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should be
          used with Torchscript.
        - **tie_word_embeddings** (:obj:`bool`, `optional`, defaults to :obj:`True`) -- Whether the model's input and
          output word embeddings should be tied. Note that this is only relevant if the model has a output word
          embedding layer.

    TensorFlow specific parameters

        - **use_bfloat16** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should use
          BFloat16 scalars (only used by some TensorFlow models).
    """
    model_type: 'str' = ''
    is_composition: 'bool' = False

    def __init__(self, **kwargs):
        self.return_dict = kwargs.pop('return_dict', True)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.torchscript = kwargs.pop('torchscript', False)
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})
        self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', True)
        self.is_encoder_decoder = kwargs.pop('is_encoder_decoder', False)
        self.is_decoder = kwargs.pop('is_decoder', False)
        self.add_cross_attention = kwargs.pop('add_cross_attention', False)
        self.tie_encoder_decoder = kwargs.pop('tie_encoder_decoder', False)
        self.max_length = kwargs.pop('max_length', 20)
        self.min_length = kwargs.pop('min_length', 0)
        self.do_sample = kwargs.pop('do_sample', False)
        self.early_stopping = kwargs.pop('early_stopping', False)
        self.num_beams = kwargs.pop('num_beams', 1)
        self.num_beam_groups = kwargs.pop('num_beam_groups', 1)
        self.diversity_penalty = kwargs.pop('diversity_penalty', 0.0)
        self.temperature = kwargs.pop('temperature', 1.0)
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 1.0)
        self.repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
        self.length_penalty = kwargs.pop('length_penalty', 1.0)
        self.no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop('encoder_no_repeat_ngram_size', 0)
        self.bad_words_ids = kwargs.pop('bad_words_ids', None)
        self.num_return_sequences = kwargs.pop('num_return_sequences', 1)
        self.chunk_size_feed_forward = kwargs.pop('chunk_size_feed_forward', 0)
        self.output_scores = kwargs.pop('output_scores', False)
        self.return_dict_in_generate = kwargs.pop('return_dict_in_generate', False)
        self.forced_bos_token_id = kwargs.pop('forced_bos_token_id', None)
        self.forced_eos_token_id = kwargs.pop('forced_eos_token_id', None)
        self.remove_invalid_values = kwargs.pop('remove_invalid_values', False)
        self.architectures = kwargs.pop('architectures', None)
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.id2label = kwargs.pop('id2label', None)
        self.label2id = kwargs.pop('label2id', None)
        if self.id2label is not None:
            kwargs.pop('num_labels', None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        else:
            self.num_labels = kwargs.pop('num_labels', 2)
        self.tokenizer_class = kwargs.pop('tokenizer_class', None)
        self.prefix = kwargs.pop('prefix', None)
        self.bos_token_id = kwargs.pop('bos_token_id', None)
        self.pad_token_id = kwargs.pop('pad_token_id', None)
        self.eos_token_id = kwargs.pop('eos_token_id', None)
        self.sep_token_id = kwargs.pop('sep_token_id', None)
        self.decoder_start_token_id = kwargs.pop('decoder_start_token_id', None)
        self.task_specific_params = kwargs.pop('task_specific_params', None)
        self.problem_type = kwargs.pop('problem_type', None)
        allowed_problem_types = 'regression', 'single_label_classification', 'multi_label_classification'
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(f"The config parameter `problem_type` wasnot understood: received {self.problem_type}but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid.")
        if kwargs.pop('xla_device', None) is not None:
            logger.warning('The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.')
        self._name_or_path = str(kwargs.pop('name_or_path', ''))
        self.easynlp_version = kwargs.pop('easynlp_version', None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) ->str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)

    @property
    def use_return_dict(self) ->bool:
        """
        :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
        """
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) ->int:
        """
        :obj:`int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: 'int'):
        if self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f'LABEL_{i}' for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory: 'Union[str, os.PathLike]', push_to_hub: 'bool'=False, **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f'Provided path ({save_directory}) should be a directory, not a file')
        if push_to_hub:
            commit_message = kwargs.pop('commit_message', None)
            repo = self._create_or_get_repo(save_directory, **kwargs)
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f'Configuration saved in {output_config_file}')
        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f'Configuration pushed to the hub in this commit: {url}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->'PretrainedConfig':
        """
        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pretrained model
        configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.,
                  ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from huggingface.co and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert config.output_attentions == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warn(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        local_files_only = kwargs.pop('local_files_only', False)
        revision = kwargs.pop('revision', None)
        from_pipeline = kwargs.pop('_from_pipeline', None)
        from_auto_class = kwargs.pop('_from_auto', False)
        user_agent = {'file_type': 'config', 'from_auto_class': from_auto_class}
        if from_pipeline is not None:
            user_agent['using_pipeline'] = from_pipeline
        if is_offline_mode() and not local_files_only:
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise Exception(f'{pretrained_model_name_or_path} is not a filer or folder.')
        config_dict = cls._dict_from_json_file(config_file)
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: 'Dict[str, Any]', **kwargs) ->'PretrainedConfig':
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        config = cls(**config_dict)
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info(f'Model config {config}')
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: 'Union[str, os.PathLike]') ->'PretrainedConfig':
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: 'Union[str, os.PathLike]'):
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f'{self.__class__.__name__} {self.to_json_string()}'

    def to_diff_dict(self) ->Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()
        default_config_dict = PretrainedConfig().to_dict()
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if key not in default_config_dict or key == 'easynlp_version' or value != default_config_dict[key] or key in class_config_dict and value != class_config_dict[key]:
                serializable_config_dict[key] = value
        return serializable_config_dict

    def to_dict(self) ->Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, 'model_type'):
            output['model_type'] = self.__class__.model_type
        output['easynlp_version'] = __version__
        return output

    def to_json_string(self, use_diff: 'bool'=True) ->str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path: 'Union[str, os.PathLike]', use_diff: 'bool'=True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: 'Dict[str, Any]'):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: 'str'):
        """
        Updates attributes of this class with attributes from ``update_str``.

        The expected format is ints, floats and strings as is, and for booleans use ``true`` or ``false``. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (:obj:`str`): String with attributes that should be updated for this class.

        """
        d = dict(x.split('=') for x in update_str.split(','))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")
            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ['true', '1', 'y', 'yes']:
                    v = True
                elif v.lower() in ['false', '0', 'n', 'no']:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(f'You can only update int, float, bool or string values in the config, got {v} for key {k}')
            setattr(self, k, v)


class ARTISTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`MinGPTModel`. 
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the GEEP model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`GEEPModel`.
    """
    model_type = 'artist'

    def __init__(self, vocab_size=37512, img_vocab_size=16384, text_vocab_size=21128, block_size=288, n_layer=12, n_head=12, n_embd=768, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_unmasked=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.img_vocab_size = img_vocab_size
        self.text_vocab_size = text_vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_unmasked = n_unmasked
        for k, v in kwargs.items():
            setattr(self, k, v)


class BartConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BartModel`. It is used to
    instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BART `facebook/bart-large
    <https://huggingface.co/facebook/bart-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import BartModel, BartConfig

        >>> # Initializing a BART facebook/bart-large style configuration
        >>> configuration = BartConfig()

        >>> # Initializing a model from the facebook/bart-large style configuration
        >>> model = BartModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'bart'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2, is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2, **kwargs):
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding
        if self.forced_bos_token_id is None and kwargs.get('force_bos_token_to_be_generated', False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(f'Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions.The config can simply be saved and uploaded again to be fixed.')

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


class DefaultIO(IO):
    __name__ = 'DefaultIO'

    def _check_path(self, path):
        if not self.islocal(path):
            raise RuntimeError('OSS Credentials must be provided to use oss_io_config. ')

    def open(self, path, mode='r', encoding='utf-8'):
        self._check_path(path)
        path = self.abspath(path)
        if mode.endswith('b'):
            return open(path, mode=mode)
        else:
            return open(path, mode=mode, encoding=encoding)

    def exists(self, path):
        self._check_path(path)
        path = self.abspath(path)
        return os.path.exists(path)

    def move(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        if src == dst:
            return
        shutil.move(src, dst)

    def copy(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass

    def copytree(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src).rstrip('/')
        dst = self.abspath(dst).rstrip('/')
        if src == dst:
            return
        self.makedirs(dst)
        created_dir = {dst}
        for file in self.listdir(src, recursive=True):
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            dst_dir = os.path.dirname(dst_file)
            if dst_dir not in created_dir:
                self.makedirs(dst_dir)
                created_dir.add(dst_dir)
            self.copy(src_file, dst_file)

    def makedirs(self, path, exist_ok=True):
        self._check_path(path)
        path = self.abspath(path)
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path):
        self._check_path(path)
        path = self.abspath(path)
        if os.path.isdir(path):
            self.rmtree(path)
        else:
            os.remove(path)

    def rmtree(self, path):
        shutil.rmtree(path)

    def listdir(self, path, recursive=False, full_path=False, contains: 'Union[str, List[str]]'=None):
        self._check_path(path)
        path = self.abspath(path)
        if isinstance(contains, str):
            contains = [contains]
        elif not contains:
            contains = ['']
        if recursive:
            files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
            if not full_path:
                prefix_len = len(path.rstrip('/')) + 1
                files = [file[prefix_len:] for file in files]
        else:
            files = os.listdir(path)
            if full_path:
                files = [os.path.join(path, file) for file in files]
        files = [file for file in files if any(keyword in file for keyword in contains)]
        return files

    def isdir(self, path):
        self._check_path(path)
        return os.path.isdir(path)

    def isfile(self, path):
        self._check_path(path)
        return os.path.isfile(path)

    def abspath(self, path):
        self._check_path(path)
        return os.path.abspath(path)

    def last_modified(self, path):
        return datetime.fromtimestamp(float(self.last_modified_str(path)))

    def last_modified_str(self, path):
        self._check_path(path)
        return str(os.path.getmtime(path))

    def size(self, path: 'str') ->int:
        return os.stat(path).st_size

    def download(self, oss_path, local_path):
        with open(oss_path, 'rb') as fin, open(local_path, 'wb') as fout:
            fout.write(fin.read())

    def download_dir(self, oss_dir, local_dir):
        created_dir = {local_dir}
        for file in self.listdir(oss_dir, recursive=True):
            src_file = os.path.join(oss_dir, file)
            dst_file = os.path.join(oss_dir, file)
            dst_dir = os.path.dirname(dst_file)
            if dst_dir not in created_dir:
                os.makedirs(dst_dir)
                created_dir.add(dst_dir)
            with open(src_file, 'rb') as fin, open(dst_file, 'wb') as fout:
                fout.write(fin.read())

    def upload(self, local_path, oss_path):
        with open(local_path, 'rb') as fin, open(oss_path, 'wb') as fout:
            fout.write(fin.read())


class _IOWrapper:

    def __init__(self):
        self._io = DefaultIO()

    def set_io(self, new_io):
        self._io = new_io

    def __getattr__(self, name):
        if hasattr(self._io, name):
            return getattr(self._io, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'io' object has no attribute '{name}'")

    def __str__(self):
        return self._io.__name__


io = _IOWrapper()


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, pre_trained='', training=''):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.model_type = 'bert'
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pre_trained = pre_trained
        self.training = training
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with io.open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with io.open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with io.open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BloomConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the Bloom model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BloomModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        skip_bias_add (`bool`, *optional*, defaults to `True`):
            If set to `True`, it will skip bias add for each linear layer in the transformer blocks
        skip_bias_add_qkv (`bool`, *optional*, defaults to `False`):
            If set to `True`, it will skip bias add for the first linear layer in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.

    Example:

    ```python
    >>> from transformers import BloomModel, BloomConfig

    >>> # Initializing a Bloom configuration
    >>> configuration = BloomConfig()

    >>> # Initializing a model from the configuration
    >>> model = BloomModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'bloom'
    keys_to_ignore_at_inference = ['past_key_values']
    attribute_map = {'num_hidden_layers': 'n_layer', 'num_attention_heads': 'n_head'}

    def __init__(self, vocab_size=250880, hidden_size=64, n_layer=2, n_head=8, layer_norm_epsilon=1e-05, initializer_range=0.02, use_cache=False, bos_token_id=1, eos_token_id=2, apply_residual_connection_post_layernorm=False, hidden_dropout=0.0, attention_dropout=0.0, pretraining_tp=1, slow_but_exact=False, **kwargs):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop('n_embed', None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class CLIPTextConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate an CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported. layer_norm_eps (`float`, *optional*,
            defaults to 1e-5): The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLIPTextModel, CLIPTextConfig

    >>> # Initializing a CLIPTextModel with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPTextConfig()

    >>> # Initializing a CLIPTextConfig from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'clip_text_model'

    def __init__(self, vocab_size=21128, hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, max_position_embeddings=512, hidden_act='gelu', layer_norm_eps=1e-12, dropout=0.0, attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0, pad_token_id=0, bos_token_id=0, eos_token_id=2, type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'clip':
            config_dict = config_dict['text_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class CLIPVisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate an CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported. layer_norm_eps (`float`, *optional*,
            defaults to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLIPVisionModel, CLIPVisionConfig

    >>> # Initializing a CLIPVisionModel with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel model from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'clip_vision_model'

    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, image_size=224, patch_size=32, hidden_act='quick_gelu', layer_norm_eps=1e-05, dropout=0.0, attention_dropout=0.0, initializer_range=0.02, initializer_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'clip':
            config_dict = config_dict['vision_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.")
        return cls.from_dict(config_dict, **kwargs)

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class CLIPConfig(PretrainedConfig):
    """
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    CLIP model according to the specified arguments, defining the text model and vision model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config_dict (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config_dict (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'clip'
    is_composition = True

    def __init__(self, text_config_dict=None, vision_config_dict=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(text_config_dict=text_config_dict, vision_config_dict=vision_config_dict, **kwargs)
        if text_config_dict is None:
            text_config_dict = {}
            logger.info('text_config_dict is None. Initializing the CLIPTextConfig with default values.')
        if vision_config_dict is None:
            vision_config_dict = {}
            logger.info('vision_config_dict is None. initializing the CLIPVisionConfig with default values.')
        self.text_config = CLIPTextConfig(**text_config_dict).__dict__
        self.vision_config = CLIPVisionConfig(**vision_config_dict).__dict__
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: 'CLIPTextConfig', vision_config: 'CLIPVisionConfig', **kwargs):
        """
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """
        return cls(text_config_dict=text_config.to_dict(), vision_config_dict=vision_config.to_dict(), **kwargs)

    def to_json_string(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['text_config'] = self.text_config
        output['vision_config'] = self.vision_config
        output['model_type'] = self.__class__.model_type
        return json.dumps(output)


class DkplmConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.DkplmModel` or a
    :class:`~transformers.TFDkplmModel`. It is used to instantiate a DKPLM model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the DKPLM `dkplm-base-uncased <https://huggingface.co/dkplm-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DKPLM model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.DkplmModel` or
            :class:`~transformers.TFDkplmModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DkplmModel` or
            :class:`~transformers.TFDkplmModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    """
    model_type = 'dkplm'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model` or a
    :class:`~transformers.TFGPT2Model`. It is used to instantiate a GPT-2 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.GPT2Model` or
            :class:`~transformers.TFGPT2Model`.
        n_positions (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, `optional`, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (:obj:`string`, `optional`, defaults to :obj:`"cls_index"`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.GPT2DoubleHeadsModel`.

            Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    """
    model_type = 'gpt2'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, scale_attn_weights=True, gradient_checkpointing=False, use_cache=True, bos_token_id=50256, eos_token_id=50256, **kwargs):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


class KBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    """
    model_type = 'kbert'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class KangarooConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    """
    model_type = 'kangaroo'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class MT5Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.MT5Model` or a
    :class:`~transformers.TFMT5Model`. It is used to instantiate a mT5 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the mT5 `google/mt5-small <https://huggingface.co/google/mt5-small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.T5Model` or :class:`~transformers.TFT5Model`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head. :obj:`d_kv` has to be equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 1024):
            Size of the intermediate feed forward layer in each :obj:`T5Block`.
        num_layers (:obj:`int`, `optional`, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (:obj:`string`, `optional`, defaults to :obj:`"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = 'mt5'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=250112, d_model=512, d_kv=64, d_ff=1024, num_layers=8, num_decoder_layers=None, num_heads=6, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='gated-gelu', is_encoder_decoder=True, use_cache=True, tokenizer_class='T5Tokenizer', tie_word_embeddings=False, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0, **kwargs):
        super().__init__(is_encoder_decoder=is_encoder_decoder, tokenizer_class=tokenizer_class, tie_word_embeddings=tie_word_embeddings, pad_token_id=pad_token_id, eos_token_id=eos_token_id, decoder_start_token_id=decoder_start_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class MegatronBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.MegatronBertModel`. It is
    used to instantiate a MEGATRON_BERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the MEGATRON_BERT
    `megatron-bert-uncased-345m <https://huggingface.co/nvidia/megatron-bert-uncased-345m>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 29056):
            Vocabulary size of the MEGATRON_BERT model. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~transformers.MegatronBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling
            :class:`~transformers.MegatronBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    Examples::

        >>> from transformers import MegatronBertModel, MegatronBertConfig

        >>> # Initializing a MEGATRON_BERT bert-base-uncased style configuration
        >>> configuration = MegatronBertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = MegatronBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'megatron-bert'

    def __init__(self, vocab_size=29056, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class MinGPTI2TConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`MinGPT`. 
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the GEEP model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`GEEPModel`.
    """
    model_type = 'mingpt_i2t'

    def __init__(self, vocab_size=37512, block_size=288, n_layer=12, n_head=12, n_embd=768, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_unmasked=0, decode_vocab_size=21128, model_type=model_type, prefix_encoder_type=None, prefix_encoder_ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_unmasked = n_unmasked
        self.decode_vocab_size = decode_vocab_size
        self.model_type = model_type
        self.prefix_encoder_type = prefix_encoder_type
        self.prefix_encoder_ckpt_path = prefix_encoder_ckpt_path
        for k, v in kwargs.items():
            setattr(self, k, v)


class PegasusConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.PegasusModel`. It is used to
    instantiate an PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PEGASUS `google/pegasus-large
    <https://huggingface.co/google/pegasus-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.PegasusModel` or
            :class:`~transformers.TFPegasusModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 1):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import PegasusModel, PegasusConfig

        >>> # Initializing a PEGASUS google/pegasus-large style configuration
        >>> configuration = PegasusConfig()

        >>> # Initializing a model from the google/pegasus-large style configuration
        >>> model = PegasusModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'pegasus'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=0, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, pad_token_id=0, eos_token_id=1, forced_eos_token_id=1, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


class RandengConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.PegasusModel`. It is used to
    instantiate an PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PEGASUS `google/pegasus-large
    <https://huggingface.co/google/pegasus-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.PegasusModel` or
            :class:`~transformers.TFPegasusModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 1):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import PegasusModel, PegasusConfig

        >>> # Initializing a PEGASUS google/pegasus-large style configuration
        >>> configuration = PegasusConfig()

        >>> # Initializing a model from the google/pegasus-large style configuration
        >>> model = PegasusModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'randeng'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=0, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, pad_token_id=0, eos_token_id=1, forced_eos_token_id=1, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


class RobertaConfig(BertConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    """
    model_type = 'roberta'

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class T5Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.T5Model` or a
    :class:`~transformers.TFT5Model`. It is used to instantiate a T5 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the T5 `t5-small <https://huggingface.co/t5-small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.T5Model` or :class:`~transformers.TFT5Model`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head. :obj:`d_kv` has to be equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 2048):
            Size of the intermediate feed forward layer in each :obj:`T5Block`.
        num_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (:obj:`string`, `optional`, defaults to :obj:`"relu"`):
            Type of feed forward layer to be used. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"`. T5v1.1 uses
            the :obj:`"gated-gelu"` feed forward projection. Original T5 uses :obj:`"relu"`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """
    model_type = 't5'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=None, num_heads=8, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1, gradient_checkpointing=False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class TextCNNConfig(BertConfig):
    """
    This is the configuration class to store the configuration of a :class:TextCNNClassify`. It is used to instantiate a
    CNN model according to the specified arguments, defining the model architecture.


    Args:
        conv_dim (:obj:`int`, `optional`, defaults to 100):
            The output dimemsion of the convolution layer
        kernal_sizes (:obj:`string`, `optional`, defaults to 1,2,3,4):
            Specify the number of convolutional layers and kerval size for each layer.
        linear_hidden_size (:obj:`int`, `optional`, defaults to 512):
            number of neurals for fead-forward layers after each convolutional layer
        embed_size (:obj:`int`, `optional`, defaults to 300):
           embedding dimension for input tokens
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the CNN model.The defalut setting is to use BERTTokenizer so the vocab size is 30522 for
            english tasks.
        sequence_length (:obj:`int`, `optional`, defaults to 128):
           max sequence length for of the input text
    Examples::

        >>> from easynlp.modelzoo.models.cnn import TextCNNConfig
        >>> from easynlp.appzoo.classification import CNNTextClassify

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = TextCNNConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = CNNTextClassify(configuration)
    """
    model_type = 'cnn'

    def __init__(self, conv_dim=100, kernel_sizes=[1, 2, 3], embed_size=300, vocab_size=21128, sequence_length=128, linear_hidden_size=None, **kwargs):
        super(TextCNNConfig, self).__init__()
        self.conv_dim = conv_dim
        self.kernel_sizes = kernel_sizes
        if linear_hidden_size:
            self.hidden_size = linear_hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    @classmethod
    def from_dict(cls, config_dict, **kwargs) ->'PretrainedConfig':
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        config = cls(conv_dim=config_dict['conv_dim'], kernel_sizes=config_dict['kernel_sizes'], linear_hidden_size=config_dict['hidden_size'], embed_size=config_dict['embed_size'], vocab_size=config_dict['vocab_size'], sequence_length=config_dict['sequence_length'])
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config


class TransformerConfig(PretrainedConfig):
    model_type = 'transformer'

    def __init__(self, vocab_size=54944, pad_token_id=0, bos_token_id=1, eos_token_id=2, embedding_size=512, ffn_size=2048, n_encoder_layers=-1, n_decoder_layers=-1, n_layers=8, n_heads=16, dropout=0.1, activation='gelu', variant='xlm', learn_positional_embeddings=True, n_positions=512, truncate=-1, text_truncate=512, label_truncate=128, attention_dropout=0.0, relu_dropout=0.0, embeddings_scale=True, output_scaling=1.0, n_segments=0, checkpoint_activations=False, beam_min_length=20, beam_block_ngram=3, tokenizer='bpe', **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_encoder_layers = n_encoder_layers
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation
        self.variant = variant
        self.learn_positional_embeddings = learn_positional_embeddings
        self.n_decoder_layers = n_decoder_layers
        self.n_positions = n_positions
        self.truncate = truncate
        self.text_truncate = text_truncate
        self.label_truncate = label_truncate
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.embeddings_scale = embeddings_scale
        self.output_scaling = output_scaling
        self.n_segments = n_segments
        self.checkpoint_activations = checkpoint_activations
        self.beam_min_length = beam_min_length
        self.beam_block_ngram = beam_block_ngram
        self.tokenizer = tokenizer


CONFIG_MAPPING = OrderedDict([('roberta', RobertaConfig), ('bert', BertConfig), ('dkplm', DkplmConfig), ('megatron-bert', MegatronBertConfig), ('gpt2', GPT2Config), ('cnn', TextCNNConfig), ('artist', ARTISTConfig), ('artist_i2t', MinGPTI2TConfig), ('mingpt_i2t', MinGPTI2TConfig), ('clip', CLIPConfig), ('kbert', KBertConfig), ('mt5', MT5Config), ('t5', T5Config), ('pegasus', PegasusConfig), ('bart', BartConfig), ('bloom', BloomConfig), ('randeng', RandengConfig), ('kangaroo', KangarooConfig), ('transformer', TransformerConfig)])


MODEL_NAMES_MAPPING = OrderedDict([('roberta', 'RoBERTa'), ('bert', 'BERT'), ('dkplm', 'DKPLM'), ('megatron-bert', 'MEGATRON-BERT'), ('gpt2', 'OpenAI GPT-2'), ('cnn', 'TextCNN'), ('artist', 'ARTIST'), ('artist_i2t', 'MinGPTI2TConfig'), ('mingpt_i2t', 'MinGPTI2TConfig'), ('clip', 'CLIP'), ('kbert', 'KBERT'), ('t5', 'T5'), ('pegasus', 'Pegasus'), ('bart', 'BART'), ('mt5', 'mT5'), ('bloom', 'Bloom'), ('randeng', 'Randeng'), ('kangaroo', 'KANGAROO'), ('transformer', 'Transformer')])


def _get_class_name(model_class):
    if isinstance(model_class, (list, tuple)):
        return ' or '.join([f':class:`~transformers.{c.__name__}`' for c in model_class])
    return f':class:`~transformers.{model_class.__name__}`'


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError('Using `use_model_types=False` requires a `config_to_class` dictionary.')
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f':class:`~transformers.{config.__name__}`' for model_type, config in CONFIG_MAPPING.items()}
        else:
            model_type_to_name = {model_type: _get_class_name(config_to_class[config]) for model_type, config in CONFIG_MAPPING.items() if config in config_to_class}
        lines = [f'{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)' for model_type in sorted(model_type_to_name.keys())]
    else:
        config_to_name = {config.__name__: _get_class_name(clas) for config, clas in config_to_class.items()}
        config_to_model_name = {config.__name__: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING.items()}
        lines = [f'{indent}- :class:`~transformers.{config_name}` configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)' for config_name in sorted(config_to_name.keys())]
    return '\n'.join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):

    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split('\n')
        i = 0
        while i < len(lines) and re.search('^(\\s*)List options\\s*$', lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search('^(\\s*)List options\\s*$', lines[i]).groups()[0]
            if use_model_types:
                indent = f'{indent}    '
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = '\n'.join(lines)
        else:
            raise ValueError(f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}")
        fn.__doc__ = docstrings
        return fn
    return docstring_decorator


class AutoConfig:
    """
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError('AutoConfig is designed to be instantiated using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method.')

    @classmethod
    def for_model(cls, model_type: 'str', *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}")

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config object
        that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                      namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        """
        kwargs['_from_auto'] = True
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if 'model_type' in config_dict:
            config_class = CONFIG_MAPPING[config_dict['model_type']]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in str(pretrained_model_name_or_path):
                    return config_class.from_dict(config_dict, **kwargs)
        raise ValueError(f"Unrecognized model in {pretrained_model_name_or_path}. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: {', '.join(CONFIG_MAPPING.keys())}")


def print_init_keys_info(loaded_keys=None, expected_keys=None, unexpected_keys=None, missing_keys=None, error_msgs=None):
    if loaded_keys is None:
        logger.warning(f'\n Inited keys of the model: None. Probably .bin file not exist.\n')
    if loaded_keys is not None and expected_keys is not None:
        init_keys = set.intersection(set(loaded_keys), set(expected_keys))
        if len(init_keys) > 0:
            key_info = '[' + ','.join(init_keys) + ']'
            logger.warning(f'\n Inited keys of the model:\n {key_info}.\n')
    if unexpected_keys is not None:
        if len(unexpected_keys) > 0:
            key_info = '[' + ','.join(unexpected_keys) + ']'
            logger.warning(f'\n Unloaded keys of the model:\n {key_info}. \n' + ' This IS expected if you initialize A model from B.\n' + ' This IS NOT expected if you initialize A model from A.\n')
        else:
            logger.warning(f'All keys are initialized.\n')
    if missing_keys is not None and len(missing_keys) > 0:
        key_info = '[' + ','.join(missing_keys) + ']'
        logger.warning(f'\n New keys are added:\n {key_info}.\n')
    if error_msgs is not None and len(error_msgs) > 0:
        error_msg = '\n\t'.join(error_msgs)


class Application(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if 'modelzoo' in pretrained_model_name_or_path:
            return cls(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(from_config=config)
        state_dict = None
        weights_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        if not io.exists(weights_path):
            print_init_keys_info()
            return model
        with io.open(weights_path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        logger.info('Loading model...')
        load(model, prefix=start_prefix)
        logger.info('Load finished!')
        expected_keys = list(model.state_dict().keys())
        loaded_keys = list(state_dict.keys())
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        print_init_keys_info(loaded_keys, expected_keys, unexpected_keys, missing_keys, error_msgs)
        return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_att=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu}


class BertIntermediate(nn.Module):

    def __init__(self, config, intermediate_size=-1):
        super(BertIntermediate, self).__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            self.dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config, intermediate_size=-1):
        super(BertOutput, self).__init__()
        if intermediate_size < 0:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, layer_att = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, layer_att


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_atts = []
        for _, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(hidden_states, attention_mask)
            all_encoder_atts.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


class BertPooler(nn.Module):

    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        pooled_output = hidden_states[-1][:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if from_tf:
            weights_path = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, output_att=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if not output_att:
            return encoded_layers, pooled_output
        return encoded_layers, layer_atts, pooled_output


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor'):
        layers_output = {}
        for idx, m in enumerate(self.resblocks):
            x = m(x)
            layers_output[idx] = x
        return x, layers_output


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CHINESE_CLIP(nn.Module):

    def __init__(self, model_type: 'str', embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', vocab_size: 'int', text_attention_probs_dropout_prob: 'float', text_hidden_act: 'str', text_hidden_dropout_prob: 'float', text_hidden_size: 'int', text_initializer_range: 'float', text_intermediate_size: 'int', text_max_position_embeddings: 'int', text_num_attention_heads: 'int', text_num_hidden_layers: 'int', text_type_vocab_size: 'int'):
        super().__init__()
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.bert_config = BertConfig(vocab_size=vocab_size, hidden_size=text_hidden_size, num_hidden_layers=text_num_hidden_layers, num_attention_heads=text_num_attention_heads, intermediate_size=text_intermediate_size, hidden_act=text_hidden_act, hidden_dropout_prob=text_hidden_dropout_prob, attention_probs_dropout_prob=text_attention_probs_dropout_prob, max_position_embeddings=text_max_position_embeddings, type_vocab_size=text_type_vocab_size, initializer_range=text_initializer_range, layer_norm_eps=1e-12)
        self.bert = BertModel(self.bert_config)
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        pad_index = 0
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype)
        return x[:, 0, :] @ self.text_projection

    def forward(self, image, text):
        assert image is not None or text is not None, 'text and image cannot both be None!'
        image_features = None
        text_features = None
        if image is not None:
            image_features = self.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text is not None:
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features


_flax_available = False


def is_flax_available():
    return _flax_available


_tf_available = False


def is_tf_available():
    return _tf_available


def is_torch_available():
    return _torch_available


_torch_fx_available = False


def is_torch_fx_available():
    return _torch_fx_available


def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx
        return isinstance(x, torch.fx.Proxy)
    return False


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jaxlib.xla_extension.DeviceArray` or
    :obj:`np.ndarray`.
    """
    if is_torch_fx_proxy(x):
        return True
    if is_torch_available():
        import torch
        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return True
    if is_flax_available():
        if isinstance(x, (jax_xla.DeviceArray, Tracer)):
            return True
    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)
        assert len(class_fields), f'{self.__class__.__name__} has no fields.'
        assert all(field.default is None for field in class_fields[1:]), f'{self.__class__.__name__} should not have more than one required field.'
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False
            if first_field_iterator:
                for element in iterator:
                    if not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.')

    def setdefault(self, *args, **kwargs):
        raise Exception(f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.')

    def pop(self, *args, **kwargs):
        raise Exception(f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        raise Exception(f'You cannot use ``update`` on a {self.__class__.__name__} instance.')

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, causal_attention_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(self, config: 'CLIPConfig'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', causal_attention_mask: 'torch.Tensor', output_attentions: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs

