
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


import torch.nn as nn


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


from torchvision import datasets


from torchvision import models


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Pad


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import ToTensor


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


from typing import Any


from typing import Optional


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.nn import CrossEntropyLoss


from torch.optim import SGD


from torchvision.models import wide_resnet50_2


import random


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torchvision.datasets.cifar import CIFAR100


from torchvision.transforms import RandomErasing


from collections import OrderedDict


import numpy as np


from torch.optim import Adam


from torchvision import transforms


from collections import namedtuple


from torchvision.models.vgg import VGG16_Weights


import warnings


import torch.utils.data as data


import torch.nn.functional as F


from torch import nn


from torchvision.datasets import MNIST


from torch.utils.tensorboard import SummaryWriter


from functools import partial


import torch.optim.lr_scheduler as lrs


from torchvision.models.resnet import resnet50


from typing import Callable


from typing import Tuple


from torch.utils.data.dataset import Subset


from torchvision.datasets import ImageFolder


from torchvision.models.segmentation import deeplabv3_resnet101


from torch.utils.data import Dataset


from torchvision.datasets.sbd import SBDataset


from torchvision.datasets.voc import VOCSegmentation


from collections import deque


from torch.distributions import Categorical


import torchvision


from torchvision.transforms.functional import center_crop


from torchvision.transforms.functional import resize


from torchvision.transforms.functional import to_tensor


import torch.nn.init as init


import numbers


from typing import cast


from typing import Dict


from typing import Iterable


from typing import Mapping


from typing import Sequence


from typing import Union


from torch.optim.optimizer import Optimizer


from torch.utils.data.distributed import DistributedSampler


import collections.abc as collections


from abc import abstractmethod


from typing import Iterator


from typing import List


from torch.utils.data import IterableDataset


from torch.utils.data.sampler import Sampler


from abc import ABCMeta


from numbers import Number


import re


import torch.distributed as dist


import torch.multiprocessing as mp


import itertools


from functools import wraps


from torch import distributed as dist


from collections.abc import Mapping


from typing import Generator


from torch.utils.data.sampler import BatchSampler


import functools


import logging


import math


import time


from collections import defaultdict


from collections.abc import Sequence


from enum import Enum


from types import DynamicClassAttribute


from typing import TYPE_CHECKING


from torch.optim import Optimizer


from typing import NamedTuple


from typing import DefaultDict


from typing import Type


from copy import deepcopy


from math import ceil


from torch.optim.lr_scheduler import _LRScheduler


from copy import copy


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Collection


from torch import Tensor


from torch.nn.functional import pairwise_distance


from typing import TextIO


from typing import TypeVar


from torch.utils.data.dataloader import _InfiniteConstantSampler


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataset import IterableDataset


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import WeightedRandomSampler


from torch.nn.functional import mse_loss


from torch.utils.data import BatchSampler


from torch.utils.data import RandomSampler


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import copy


import matplotlib


from torch.optim.lr_scheduler import ExponentialLR


from sklearn.metrics import calinski_harabasz_score


from sklearn.metrics import davies_bouldin_score


from sklearn.metrics import silhouette_score


import scipy


from numpy import cov


from collections import Counter


from sklearn.metrics import DistanceMetric


from scipy.stats import kendalltau


from scipy.stats import pearsonr


from sklearn.metrics import r2_score


from scipy.stats import spearmanr


from torch.nn import Linear


from sklearn.metrics import accuracy_score


import sklearn


from sklearn.metrics import average_precision_score


from sklearn.metrics import cohen_kappa_score


from sklearn.metrics import confusion_matrix


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from scipy.special import softmax


from scipy.stats import entropy as scipy_entropy


from sklearn.metrics import fbeta_score


from scipy.spatial.distance import jensenshannon


from scipy.stats import entropy


from numpy.testing import assert_almost_equal


from torch.nn.functional import nll_loss


from sklearn.metrics import f1_score


from sklearn.metrics import multilabel_confusion_matrix


from sklearn.exceptions import UndefinedMetricWarning


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import roc_auc_score


from sklearn.metrics import roc_curve


from itertools import accumulate


class PACTClip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, 0, alpha.data)

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        dx = dy.clone()
        dx[x < 0] = 0
        dx[x > alpha] = 0
        dalpha = dy.clone()
        dalpha[x <= alpha] = 0
        return dx, torch.sum(dalpha)


class PACTReLU(nn.Module):

    def __init__(self, alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return PACTClip.apply(x, self.alpha)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, weight_bit_width=8):
    """3x3 convolution with padding"""
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, weight_bit_width=weight_bit_width)


def make_PACT_relu(bit_width=8):
    relu = qnn.QuantReLU(bit_width=bit_width)
    relu.act_impl = PACTReLU()
    return relu


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, weight_bit_width=bit_width)
        self.bn1 = norm_layer(planes)
        self.relu = make_PACT_relu(bit_width=bit_width)
        self.conv2 = conv3x3(planes, planes, weight_bit_width=bit_width)
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
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1, weight_bit_width=8):
    """1x1 convolution"""
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, weight_bit_width=weight_bit_width)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, weight_bit_width=bit_width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, weight_bit_width=bit_width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, weight_bit_width=bit_width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = make_PACT_relu(bit_width=bit_width)
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
        out = self.relu(out)
        return out


class ResNet_QAT_Xb(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = qnn.QuantConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = make_PACT_relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bit_width=bit_width)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], bit_width=bit_width)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], bit_width=bit_width)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], bit_width=bit_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bit_width=8):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride, weight_bit_width=bit_width), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, bit_width=bit_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, bit_width=bit_width))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Net(nn.Module):

    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


class Generator(Net):
    """Generator network.

    Args:
        nf (int): Number of filters in the second-to-last deconv layer
    """

    def __init__(self, z_dim, nf, nc):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_channels=z_dim, out_channels=nf * 8, kernel_size=4, stride=1, padding=0, bias=False), nn.BatchNorm2d(nf * 8), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        self.weights_init()

    def forward(self, x):
        return self.net(x)


class Discriminator(Net):
    """Discriminator network.

    Args:
        nf (int): Number of filters in the first conv layer.
    """

    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.weights_init()

    def forward(self, x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class SiameseNetwork(nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer.
    The output of the linear layer passed through a sigmoid function.
    `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
    This implementation varies from FaceNet as we use the `ResNet-18` model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`
    as our feature extractor.
    In addition we use CIFAR10 dataset along with TripletMarginLoss
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet34(weights=None)
        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(fc_in_features, 256), nn.ReLU(inplace=True), nn.Linear(256, 10), nn.ReLU(inplace=True))
        self.relu = nn.ReLU()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output1 = self.fc(output1)
        output2 = self.fc(output2)
        output3 = self.fc(output3)
        return output1, output2, output3


class TransformerModel(nn.Module):

    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes, output_hidden_states=n_fc, classifier_dropout=dropout, output_attentions=True)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir, config=self.config)

    def forward(self, inputs):
        output = self.transformer(**inputs)['logits']
        return output


class InceptionModel(torch.nn.Module):
    """Inception Model pre-trained on the ImageNet Dataset.

    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def __init__(self, return_features: 'bool', device: 'Union[str, torch.device]'='cpu') ->None:
        try:
            import torchvision
            from torchvision import models
        except ImportError:
            raise ModuleNotFoundError('This module requires torchvision to be installed.')
        super(InceptionModel, self).__init__()
        self._device = device
        if Version(torchvision.__version__) < Version('0.13.0'):
            model_kwargs = {'pretrained': True}
        else:
            model_kwargs = {'weights': models.Inception_V3_Weights.DEFAULT}
        self.model = models.inception_v3(**model_kwargs)
        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: 'torch.Tensor') ->torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f'Inputs should be a tensor of dim 4, got {data.dim()}')
        if data.shape[1] != 3:
            raise ValueError(f'Inputs should be a tensor with 3 channels, got {data.shape}')
        if data.device != torch.device(self._device):
            data = data
        return self.model(data)


class DummyModel(nn.Module):

    def __init__(self, n_channels=10, out_channels=1, flatten_input=False):
        super(DummyModel, self).__init__()
        self.net = nn.Sequential(nn.Flatten() if flatten_input else nn.Identity(), nn.Linear(n_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class DummyPretrainedModel(nn.Module):

    def __init__(self):
        super(DummyPretrainedModel, self).__init__()
        self.features = nn.Linear(4, 2, bias=False)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class DummyModelMulipleParamGroups(nn.Module):

    def __init__(self):
        super(DummyModelMulipleParamGroups, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CustomMultiMSELoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, y_pred: 'Tuple[torch.Tensor, torch.Tensor]', y_true: 'Tuple[torch.Tensor, torch.Tensor]') ->torch.Tensor:
        a_true, b_true = y_true
        a_pred, b_pred = y_pred
        return mse_loss(a_pred, a_true) + mse_loss(b_pred, b_true)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CustomMultiMSELoss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), (torch.rand([4, 4]), torch.rand([4, 4, 4, 4]))], {})),
    (DummyPretrainedModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PACTReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Policy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SiameseNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

