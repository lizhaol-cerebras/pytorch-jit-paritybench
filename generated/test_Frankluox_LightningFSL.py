
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


import torchvision.transforms as transforms


import torch


import torch.nn as nn


from sklearn.cluster import KMeans


from torchvision.datasets import ImageFolder


import numbers


from collections.abc import Sequence


import numpy as np


import torchvision.transforms.functional as functional


import math


from torchvision.transforms import RandomResizedCrop


import torch.nn.functional as F


from torch.nn.utils.weight_norm import WeightNorm


from torch import Tensor


import torch.nn.init as init


import random


import torch.distributed as dist


from typing import Optional


from typing import Dict


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torchvision import transforms


from typing import List


from typing import Union


from typing import TypeVar


from typing import Iterator


from torch.utils.data import Sampler


from typing import Tuple


from torch.nn import Module


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim import Adam


from torch.optim import SGD


def weight_norm(module, name='weight', dim=0):
    """Applies weight normalization to a parameter in the given module.

    .. math::
         \\mathbf{w} = g \\dfrac{\\mathbf{v}}{\\|\\mathbf{v}\\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


class CC_head(nn.Module):

    def __init__(self, indim, outdim, scale_cls=10.0, learn_scale=True, normalize=True):
        super().__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale)
        self.normalize = normalize

    def forward(self, features):
        if features.dim() == 4:
            if self.normalize:
                features = F.normalize(features, p=2, dim=1, eps=1e-12)
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        assert features.dim() == 2
        x_normalized = F.normalize(features, p=2, dim=1, eps=1e-12)
        self.L.weight.data = F.normalize(self.L.weight.data, p=2, dim=1, eps=1e-12)
        cos_dist = self.L(x_normalized)
        classification_scores = self.scale_cls * cos_dist
        return classification_scores


class SOC(nn.Module):

    def __init__(self, num_patch, alpha, beta):
        super().__init__()
        self.num_patch = num_patch
        self.alpha = alpha
        self.beta = beta

    def forward(self, feature_extractor, data, way, shot, batch_size):
        num_support_samples = way * shot
        num_patch = data.size(1)
        data = data.reshape([-1] + list(data.shape[-3:]))
        data = feature_extractor(data)
        data = nn.functional.normalize(data, dim=1)
        data = F.adaptive_avg_pool2d(data, 1)
        data = data.reshape([batch_size, -1, num_patch] + list(data.shape[-3:]))
        data = data.permute(0, 1, 3, 2, 4, 5).squeeze(-1)
        features_train = data[:, :num_support_samples]
        features_test = data[:, num_support_samples:]
        M = features_train.shape[1]
        N = features_test.shape[1]
        c = features_train.size(2)
        b = features_train.size(0)
        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
        features_train = features_train.reshape(list(features_train.shape[:3]) + [-1])
        num = features_train.size(3)
        patch_num = self.num_patch
        if shot == 1:
            features_focus = features_train
        else:
            features_focus = []
            features_train = features_train.reshape([b, shot, way] + list(features_train.shape[2:]))
            features_train = torch.transpose(features_train, 1, 2)
            count = 1.0
            for l in range(patch_num - 1):
                features_train_ = list(torch.split(features_train, 1, dim=2))
                for i in range(shot):
                    features_train_[i] = features_train_[i].squeeze(2)
                    repeat_dim = [1, 1, 1]
                    for j in range(i):
                        features_train_[i] = features_train_[i].unsqueeze(3)
                        repeat_dim.append(num)
                    repeat_dim.append(1)
                    for j in range(shot - i - 1):
                        features_train_[i] = features_train_[i].unsqueeze(-1)
                        repeat_dim.append(num)
                    features_train_[i] = features_train_[i].repeat(repeat_dim)
                features_train_ = torch.stack(features_train_, dim=shot + 3)
                repeat_dim = []
                for _ in range(shot + 4):
                    repeat_dim.append(1)
                repeat_dim.append(shot)
                features_train_ = features_train_.unsqueeze(-1).repeat(repeat_dim)
                features_train_ = (features_train_ * torch.transpose(features_train_, shot + 3, shot + 4)).sum(2)
                features_train_ = features_train_.reshape(b, way, -1, shot, shot)
                for i in range(shot):
                    features_train_[:, :, :, i, i] = 0
                sim = features_train_.sum(-1).sum(-1)
                _, idx = torch.max(sim, dim=2)
                best_idx = torch.LongTensor(b, way, shot)
                for i in range(shot):
                    best_idx[:, :, shot - i - 1] = idx % num
                    idx = idx // num
                feature_train_ = features_train.reshape(-1, c, num)
                best_idx_ = best_idx.reshape(-1)
                b_index = torch.LongTensor(range(b * way * shot)).unsqueeze(1).repeat(1, c).unsqueeze(-1)
                c_index = torch.LongTensor(range(c)).unsqueeze(0).repeat(b * way * shot, 1).unsqueeze(-1)
                num_index = best_idx_.unsqueeze(-1).repeat(1, c).unsqueeze(-1)
                feature_pick = feature_train_[b_index, c_index, num_index].squeeze().reshape(b, way, shot, c)
                feature_avg = torch.mean(feature_pick, dim=2)
                feature_avg = F.normalize(feature_avg, p=2, dim=2, eps=1e-12)
                features_focus.append(count * feature_avg)
                count *= self.alpha
                temp = torch.FloatTensor(b, way, shot, c, num - 1)
                for q in range(b):
                    for w in range(way):
                        for r in range(shot):
                            temp[q, w, r, :, :] = features_train[q, w, r, :, torch.arange(num) != best_idx[q, w, r].item()]
                features_train = temp
                num = num - 1
            features_train = torch.mean(features_train.squeeze(-1), dim=2)
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_focus.append(count * feature_avg)
            features_focus = torch.stack(features_focus, dim=3)
        M = way
        features_focus = features_focus.unsqueeze(2)
        features_test = features_test.unsqueeze(1)
        features_test = features_test.reshape(list(features_test.shape[:4]) + [-1])
        features_focus = features_focus.repeat(1, 1, N, 1, 1)
        features_test = features_test.repeat(1, M, 1, 1, 1)
        sim = torch.einsum('bmnch,bmncw->bmnhw', features_focus, features_test)
        combination = []
        count = 1.0
        for i in range(patch_num - 1):
            combination_, idx_1 = torch.max(sim, dim=3)
            combination_, idx_2 = torch.max(combination_, dim=3)
            combination.append(F.relu(combination_) * count)
            count *= self.beta
            temp = torch.FloatTensor(b, M, N, sim.size(3) - 1, sim.size(4) - 1)
            for q in range(b):
                for w in range(M):
                    for e in range(N):
                        temp[q, w, e, :, :] = sim[q, w, e, torch.arange(sim.size(3)) != idx_1[q, w, e, idx_2[q, w, e]].item(), torch.arange(sim.size(4)) != idx_2[q, w, e].item()]
            sim = temp
        sim = sim.reshape(b, M, N)
        combination.append(F.relu(sim) * count)
        combination = torch.stack(combination, dim=-1).sum(-1)
        classification_scores = torch.transpose(combination, 1, 2)
        return classification_scores


def L2SquareDist(A: 'Tensor', B: 'Tensor', average: 'bool'=True) ->Tensor:
    """calculate parwise euclidean distance between two batchs of features.

    Args:
        A: Torch feature tensor. size:[Batch_size, Na, nC]
        B: Torch feature tensor. size:[Batch_size, Nb, nC]
    Output:
        dist: The calculated distance tensor. size:[Batch_size, Na, Nb]
    """
    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)
    AB = torch.bmm(A, B.transpose(1, 2))
    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1)
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb)
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB
    if average:
        dist = dist / nC
    return dist


class PN_head(nn.Module):
    """The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.

    Args:
        metric: Whether use cosine or enclidean distance.
        scale_cls: The initial scale number which affects the following softmax function.
        learn_scale: Whether make scale number learnable.
        normalize: Whether normalize each spatial dimension of image features before average pooling.
    """

    def __init__(self, metric: 'str'='cosine', scale_cls: 'int'=10.0, learn_scale: 'bool'=True, normalize: 'bool'=True) ->None:
        super().__init__()
        assert metric in ['cosine', 'enclidean']
        if learn_scale:
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        else:
            self.scale_cls = scale_cls
        self.metric = metric
        self.normalize = normalize

    def forward(self, features_test: 'Tensor', features_train: 'Tensor', way: 'int', shot: 'int') ->Tensor:
        """Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        if features_train.dim() == 5:
            if self.normalize:
                features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3
        batch_size = features_train.size(0)
        if self.metric == 'cosine':
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1), dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)
        if self.normalize:
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        assert features_test.dim() == 3
        if self.metric == 'cosine':
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
            classification_scores = self.scale_cls * torch.bmm(features_test, prototypes.transpose(1, 2))
        elif self.metric == 'euclidean':
            classification_scores = -self.scale_cls * L2SquareDist(features_test, prototypes)
        return classification_scores


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def mixup_data(x, y, lam):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_b = y[index]
    return mixed_x, y_b, lam


class WideResNet(nn.Module):

    def __init__(self, depth=28, widen_factor=10, stride=1):
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.outdim = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, target=None, mixup=False, mixup_hidden=True, mixup_alpha=None, lam=0.4):
        if target is not None:
            if mixup_hidden:
                layer_mix = random.randint(0, 3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None
            out = x
            target_a = target_b = target
            if layer_mix == 0:
                out, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.conv1(out)
            out = self.block1(out)
            if layer_mix == 1:
                out, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.block2(out)
            if layer_mix == 2:
                out, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.block3(out)
            if layer_mix == 3:
                out, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.relu(self.bn1(out))
            return out, target_b
        else:
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            return out


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        return out


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


class ResNet12(nn.Module):
    """The standard popular ResNet12 Model used in Few-Shot Learning.
    """

    def __init__(self, channels):
        super().__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.outdim = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(conv1x1(self.inplanes, planes), norm_layer(planes))
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'downsample': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CC_head,
     lambda: ([], {'indim': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': torch.nn.ReLU, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet12,
     lambda: ([], {'channels': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (WideResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

