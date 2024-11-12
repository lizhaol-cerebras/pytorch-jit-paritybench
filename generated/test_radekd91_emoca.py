
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


from enum import Enum


import numpy as np


import scipy as sp


import torch


import pandas as pd


from torch.utils.data.dataloader import DataLoader


from torchvision.transforms import Resize


from torchvision.transforms import Compose


from sklearn.neighbors import NearestNeighbors


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.sampler import WeightedRandomSampler


from collections import OrderedDict


import warnings


from torch.utils.data._utils.collate import *


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


import torchvision.transforms as transforms


import scipy


import scipy.io


from typing import Optional


from typing import Union


from typing import List


from typing import Any


from typing import overload


import copy


from torchvision.transforms import Normalize


import types


import torch.nn.functional as F


from torchvision.transforms import ToTensor


from types import GeneratorType


import torch.nn as nn


from torch import nn


from torch import optim


from functools import reduce


import torchvision.models as models


import inspect


from torch.autograd import Variable


import math


import torchvision


import torchvision.transforms.functional as F_v


import torchvision.models.vgg as vgg


from torch.nn import Linear


from torch.nn import BatchNorm1d


from torch.nn import LayerNorm


from torch.nn import InstanceNorm1d


from torch.nn.functional import mse_loss


from torch.nn.functional import cross_entropy


from torch.nn.functional import nll_loss


from torch.nn.functional import l1_loss


from torch.nn.functional import log_softmax


from torch.nn import functional


from torch.nn import LeakyReLU


from torch.nn import Sequential


from torch.nn.parameter import Parameter


import torch.optim as optim


from torch import nn as nn


from torch.hub import load_state_dict_from_url


from scipy.ndimage import morphology


from abc import abstractmethod


from abc import ABC


import matplotlib.pyplot as plt


from torchvision import transforms as tf


import torch.functional as F


from torch.functional import F


from torchvision import models


from torchvision import transforms


from matplotlib.colors import LinearSegmentedColormap


class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ArcFace(nn.Module):

    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: 'torch.Tensor', label=None):
        if label is not None:
            index = torch.where(label != -1)[0]
        else:
            index = torch.ones()
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLossHeadless(nn.Module):

    def __init__(self, feature_size, batch_size=None, lambd=0.005, final_reduction='mean_on_diag'):
        super().__init__()
        self.bn = nn.BatchNorm1d(feature_size, affine=False)
        self.lambd = lambd
        self.batch_size = batch_size
        if final_reduction not in ['sum', 'mean', 'mean_on_diag', 'mean_off_diag']:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        self.final_reduction = final_reduction

    def forward(self, z1, z2, batch_size=None, ring_size=None):
        assert not (batch_size is not None and self.batch_size is not None)
        if ring_size is not None and ring_size > 1:
            raise NotImplementedError('Barlow Twins with rings are not yet supported.')
        if batch_size is None:
            if self.batch_size is not None:
                batch_size = self.batch_size
            else:
                None
                batch_size = z1.shape[0]
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(batch_size)
        if torch.distributed.is_initialized():
            torch.distributed.nn.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2)
        off_diag = off_diagonal(c).pow_(2)
        if self.final_reduction == 'sum':
            on_diag = on_diag.sum()
            off_diag = off_diag.sum()
        elif self.final_reduction == 'mean':
            on_diag = on_diag.mean()
            off_diag = off_diag.mean()
        elif self.final_reduction == 'mean_on_diag':
            n = on_diag.numel()
            on_diag = on_diag.mean()
            off_diag = off_diag.sum() / n
        elif self.final_reduction == 'mean_off_diag':
            n = off_diag.numel()
            on_diag = on_diag.sum() / n
            off_diag = off_diag.mean()
        else:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        loss = on_diag + self.lambd * off_diag
        return loss


class BarlowTwinsLoss(nn.Module):

    def __init__(self, feature_size=2048, layer_sizes=None, final_reduction='mean_on_diag'):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = 3 * [8192]
        sizes = [feature_size] + layer_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bt_loss_headless = BarlowTwinsLossHeadless(sizes[-1], final_reduction=final_reduction)

    def forward(self, y1, y2, batch_size=None, ring_size=None):
        if self.projector is not None:
            z1 = self.projector(y1)
            z2 = self.projector(y2)
        else:
            z1 = y1
            z2 = y2
        loss = self.bt_loss_headless(z1, z2, batch_size=batch_size, ring_size=ring_size)
        return loss


class BarlowTwins(nn.Module):

    def __init__(self, args, backbone=None):
        super().__init__()
        self.args = args
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        self.bt_loss = BarlowTwinsLoss(self.args)

    def forward(self, y1, y2):
        loss = self.bt_loss(self.backbone(y1), self.backbone(y2))
        return loss


class VGG19FeatLayer(nn.Module):

    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        return out


class IDMRFLoss(nn.Module):

    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.size(0)
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [(self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        content_loss_list = [(self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        return self.style_loss + self.content_loss

    def train(self, b=True):
        return super().train(False)


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        self.register_buffer('mean', torch.Tensor(np.array([129.1863, 104.7624, 93.594]) / 255.0).float().view(1, 3, 1, 1))

    def load_weights(self, path='pretrained/VGG_FACE.t7'):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, 'conv_%d_%d' % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, 'fc%d' % block)
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        out['relu3_2'] = x
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        out['relu4_2'] = x
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out['last'] = x
        return out


class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().float()
        self.featlayer.load_weights(path='data/face_recognition_model/vgg_face_torch/VGG_FACE.t7')
        self.featlayer = self.featlayer.eval()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.size(0)
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [(self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style
        content_loss_list = [(self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
        return self.style_loss + self.content_loss


class IdentityLoss(nn.Module):

    def __init__(self, pretrained_data='vggface2'):
        super(IdentityLoss, self).__init__()
        self.reg_model = InceptionResnetV1(pretrained=pretrained_data).eval()
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def _cos_metric(self, x, y, dim=1):
        return F.cosine_similarity(x, y)

    def _l2_metric(self, x, y):
        return ((x - y) ** 2).mean()

    def reg_features(self, x):
        out = []
        x = F.interpolate(x * 2.0 - 1.0, [160, 160])
        x = self.reg_model.conv2d_1a(x)
        x = self.reg_model.conv2d_2a(x)
        x = self.reg_model.conv2d_2b(x)
        x = self.reg_model.maxpool_3a(x)
        x = self.reg_model.conv2d_3b(x)
        x = self.reg_model.conv2d_4a(x)
        x = self.reg_model.conv2d_4b(x)
        x = self.reg_model.repeat_1(x)
        x = self.reg_model.mixed_6a(x)
        x = self.reg_model.repeat_2(x)
        out.append(x)
        x = self.reg_model.mixed_7a(x)
        x = self.reg_model.repeat_3(x)
        x = self.reg_model.block8(x)
        out.append(x)
        x = self.reg_model.avgpool_1a(x)
        x = self.reg_model.dropout(x)
        x = self.reg_model.last_linear(x.view(x.shape[0], -1))
        x = self.reg_model.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        out.append(x)
        return out

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-05
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT
        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm
        cosine_dist_l = []
        BatchSize = tar.size(0)
        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)
            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = -(cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar, content_loss=False, identity_loss=True, content_type='mrf', identity_type='l2'):
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        if identity_loss:
            if identity_type == 'l2':
                loss = ((gen_out[-1] - tar_out[-1]) ** 2).mean()
            else:
                loss = 1 - F.cosine_similarity(gen_out[-1], tar_out[-1]).mean()
        else:
            loss = 0.0
        if content_loss:
            weight = [1, 1, 1, 1]
            for i in range(len(gen_out) - 1):
                if content_type == 'mrf':
                    loss_curr = self.mrf_loss(gen_out[i], tar_out[i]) * 0.0001
                elif content_type == 'l2':
                    loss_curr = self._l2_metric(gen_out[i], tar_out[i]) * 0.02
                loss = loss + loss_curr * weight[i]
        return loss


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        return x2


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class VGGFace2Loss(nn.Module):

    def __init__(self, pretrained_checkpoint_path=None, metric='cosine_similarity', trainable=False):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        checkpoint = pretrained_checkpoint_path or '/ps/scratch/rdanecek/FaceRecognition/resnet50_ft_weight.pkl'
        load_state_dict(self.reg_model, checkpoint)
        self.register_buffer('mean_bgr', torch.tensor([91.4953, 103.8827, 131.0912]))
        self.trainable = trainable
        if metric is None:
            metric = 'cosine_similarity'
        if metric not in ['l1', 'l1_loss', 'l2', 'mse', 'mse_loss', 'cosine_similarity', 'barlow_twins', 'barlow_twins_headless']:
            raise ValueError(f'Invalid metric for face recognition feature loss: {metric}')
        if metric == 'barlow_twins_headless':
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLossHeadless(feature_size)
        elif metric == 'barlow_twins':
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLoss(feature_size)
        else:
            self.bt_loss = None
        self.metric = metric

    def _get_trainable_params(self):
        params = []
        if self.trainable:
            params += list(self.reg_model.parameters())
        if self.bt_loss is not None:
            params += list(self.bt_loss.parameters())
        return params

    def train(self, b=True):
        if not self.trainable:
            ret = super().train(False)
        else:
            ret = super().train(b)
        if self.bt_loss is not None:
            self.bt_loss.train(b)
        return ret

    def requires_grad_(self, b):
        super().requires_grad_(False)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(b)

    def freeze_nontrainable_layers(self):
        if not self.trainable:
            super().requires_grad_(False)
        else:
            super().requires_grad_(True)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(True)

    def reg_features(self, x):
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2.0 - 1.0, [224, 224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_bgr
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True, batch_size=None, ring_size=None):
        gen = self.transform(gen)
        tar = self.transform(tar)
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        if self.metric == 'cosine_similarity':
            loss = self._cos_metric(gen_out, tar_out).mean()
        elif self.metric in ['l1', 'l1_loss', 'mae']:
            loss = torch.nn.functional.l1_loss(gen_out, tar_out)
        elif self.metric in ['mse', 'mse_loss', 'l2', 'l2_loss']:
            loss = torch.nn.functional.mse_loss(gen_out, tar_out)
        elif self.metric in ['barlow_twins_headless', 'barlow_twins']:
            loss = self.bt_loss(gen_out, tar_out, batch_size=batch_size, ring_size=ring_size)
        else:
            raise ValueError(f'Invalid metric for face recognition feature loss: {self.metric}')
        return loss


def class_from_str(str, module=None, none_on_fail=False) ->type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")


def cosine_sim_negative(*args, **kwargs):
    return (1.0 - F.cosine_similarity(*args, **kwargs)).mean()


def metric_from_cfg(metric):
    if metric.type == 'cosine_similarity':
        return cosine_sim_negative
    elif metric.type in ['l1', 'l1_loss', 'mae']:
        return torch.nn.functional.l1_loss
    elif metric.type in ['mse', 'mse_loss', 'l2', 'l2_loss']:
        return torch.nn.functional.mse_loss
    elif metric.type == 'barlow_twins_headless':
        return BarlowTwinsLossHeadless(metric.feature_size)
    elif metric.type == 'barlow_twins':
        layer_sizes = metric.layer_sizes if 'layer_sizes' in metric.keys() else None
        return BarlowTwinsLoss(metric.feature_size, layer_sizes)
    else:
        raise ValueError(f'Invalid metric for deep feature loss: {metric}')


def metric_from_str(metric, **kwargs):
    if metric == 'cosine_similarity':
        return cosine_sim_negative
    elif metric in ['l1', 'l1_loss', 'mae']:
        return torch.nn.functional.l1_loss
    elif metric in ['mse', 'mse_loss', 'l2', 'l2_loss']:
        return torch.nn.functional.mse_loss
    elif metric == 'barlow_twins_headless':
        return BarlowTwinsLossHeadless(**kwargs)
    elif metric == 'barlow_twins':
        return BarlowTwinsLoss(**kwargs)
    else:
        raise ValueError(f'Invalid metric for deep feature loss: {metric}')


def get_metric(metric):
    if isinstance(metric, str):
        return metric_from_str(metric)
    if isinstance(metric, (DictConfig, Munch)):
        return metric_from_cfg(metric)
    if isinstance(metric, dict):
        return metric_from_cfg(Munch(metric))
    raise ValueError(f"invalid type: '{type(metric)}'")


class EmoLossBase(torch.nn.Module):

    def __init__(self, trainable=False, normalize_features=False, emo_feat_loss=None, au_loss=None, last_feature_size=None):
        super().__init__()
        self.last_feature_size = last_feature_size
        if emo_feat_loss is not None:
            if isinstance(emo_feat_loss, str) and 'barlow_twins' in emo_feat_loss:
                emo_feat_loss_type = emo_feat_loss
                emo_feat_loss = {}
                emo_feat_loss['type'] = emo_feat_loss_type
            if isinstance(emo_feat_loss, dict) and 'barlow_twins' in emo_feat_loss['type']:
                emo_feat_loss['feature_size'] = last_feature_size
            emo_feat_loss = get_metric(emo_feat_loss)
        if isinstance(au_loss, str):
            au_loss = class_from_str(au_loss, F)
        self.emo_feat_loss = emo_feat_loss or F.l1_loss
        self.normalize_features = normalize_features
        self.valence_loss = F.l1_loss
        self.arousal_loss = F.l1_loss
        self.expression_loss = F.l1_loss
        self.au_loss = au_loss or F.l1_loss
        self.input_emotion = None
        self.output_emotion = None
        self.trainable = trainable

    @property
    def input_emo(self):
        return self.input_emotion

    @property
    def output_emo(self):
        return self.output_emotion

    def _forward_input(self, images):
        with torch.no_grad():
            result = self(images)
        return result

    def _forward_output(self, images):
        return self(images)

    def compute_loss(self, input_images, output_images, batch_size=None, ring_size=None):
        input_emotion = self._forward_input(input_images)
        output_emotion = self._forward_output(output_images)
        self.input_emotion = input_emotion
        self.output_emotion = output_emotion
        if 'emo_feat' in input_emotion.keys():
            input_emofeat = input_emotion['emo_feat']
            output_emofeat = output_emotion['emo_feat']
            if self.normalize_features:
                input_emofeat = input_emofeat / input_emofeat.view(input_images.shape[0], -1).norm(dim=1).view(-1, *((len(input_emofeat.shape) - 1) * [1]))
                output_emofeat = output_emofeat / output_emofeat.view(output_images.shape[0], -1).norm(dim=1).view(-1, *((len(input_emofeat.shape) - 1) * [1]))
            if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
                emo_feat_loss_1 = self.emo_feat_loss(input_emofeat, output_emofeat, batch_size=batch_size, ring_size=ring_size).mean()
            else:
                emo_feat_loss_1 = self.emo_feat_loss(input_emofeat, output_emofeat).mean()
        else:
            emo_feat_loss_1 = None
        input_emofeat_2 = input_emotion['emo_feat_2']
        output_emofeat_2 = output_emotion['emo_feat_2']
        if self.normalize_features:
            input_emofeat_2 = input_emofeat_2 / input_emofeat_2.view(input_images.shape[0], -1).norm(dim=1).view(-1, *((len(input_emofeat_2.shape) - 1) * [1]))
            output_emofeat_2 = output_emofeat_2 / output_emofeat_2.view(output_images.shape[0], -1).norm(dim=1).view(-1, *((len(input_emofeat_2.shape) - 1) * [1]))
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            emo_feat_loss_2 = self.emo_feat_loss(input_emofeat_2, output_emofeat_2, batch_size=batch_size, ring_size=ring_size).mean()
        else:
            emo_feat_loss_2 = self.emo_feat_loss(input_emofeat_2, output_emofeat_2).mean()
        if 'valence' in input_emotion.keys() and input_emotion['valence'] is not None:
            valence_loss = self.valence_loss(input_emotion['valence'], output_emotion['valence'])
        else:
            valence_loss = None
        if 'arousal' in input_emotion.keys() and input_emotion['arousal'] is not None:
            arousal_loss = self.arousal_loss(input_emotion['arousal'], output_emotion['arousal'])
        else:
            arousal_loss = None
        if 'expression' in input_emotion.keys() and input_emotion['expression'] is not None:
            expression_loss = self.expression_loss(input_emotion['expression'], output_emotion['expression'])
        elif 'expr_classification' in input_emotion.keys() and input_emotion['expr_classification'] is not None:
            expression_loss = self.expression_loss(input_emotion['expr_classification'], output_emotion['expr_classification'])
        else:
            expression_loss = None
        if 'AUs' in input_emotion.keys() and input_emotion['AUs'] is not None:
            au_loss = self.au_loss(input_emotion['AUs'], output_emotion['AUs'])
        else:
            au_loss = None
        return emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, au_loss

    def _get_trainable_params(self):
        params = []
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            params += list(self.emo_feat_loss.parameters())
        return params

    def is_trainable(self):
        return len(self._get_trainable_params()) != 0

    def train(self, b=True):
        super().train(False)
        if isinstance(self.emo_feat_loss, (BarlowTwinsLossHeadless, BarlowTwinsLoss)):
            self.emo_feat_loss.train(b)
        return self


def get_emonet(device=None, load_pretrained=True):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_to_emonet = get_path_to_externals() / 'emonet'
    if not (str(path_to_emonet) in sys.path or str(path_to_emonet.absolute()) in sys.path):
        sys.path += [str(path_to_emonet)]
    n_expression = 8
    net = EmoNet(n_expression=n_expression)
    state_dict_path = Path(inspect.getfile(EmoNet)).parent.parent.parent / 'pretrained' / f'emonet_{n_expression}.pth'
    None
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    if not load_pretrained:
        None
        net.reset_emo_parameters()
    net.eval()
    return net


class EmoNetLoss(EmoLossBase):

    def __init__(self, device, emonet=None, trainable=False, normalize_features=False, emo_feat_loss=None, au_loss=None):
        if emonet is None:
            emonet = get_emonet(device).eval()
        last_feature_size = 256
        if isinstance(emo_feat_loss, dict) and 'barlow_twins' in emo_feat_loss['type']:
            emo_feat_loss['feature_size'] = last_feature_size
        super().__init__(trainable, normalize_features=normalize_features, emo_feat_loss=emo_feat_loss, au_loss=au_loss, last_feature_size=last_feature_size)
        self.emonet = emonet
        if not trainable:
            self.emonet.eval()
            self.emonet.requires_grad_(False)
        else:
            self.emonet.train()
            self.emonet.emo_parameters_requires_grad(True)
        self.size = 256, 256

    @property
    def network(self):
        return self.emonet

    def to(self, *args, **kwargs):
        self.emonet = self.emonet

    def eval(self):
        self.emonet = self.emonet.eval()

    def train(self, mode: 'bool'=True):
        super().train(mode)
        if hasattr(self, 'emonet'):
            self.emonet = self.emonet.eval()

    def forward(self, images):
        return self.emonet_out(images)

    def emonet_out(self, images):
        images = F.interpolate(images, self.size, mode='bilinear')
        return self.emonet(images, intermediate_features=True)

    def _get_trainable_params(self):
        if self.trainable:
            return self.emonet.emo_parameters
        return []


class EmoBackboneLoss(EmoLossBase):

    def __init__(self, device, backbone, trainable=False, normalize_features=False, emo_feat_loss=None, au_loss=None):
        if isinstance(emo_feat_loss, dict) and 'barlow_twins' in emo_feat_loss['type']:
            emo_feat_loss['feature_size'] = backbone.get_last_feature_size()
        super().__init__(trainable, normalize_features=normalize_features, emo_feat_loss=emo_feat_loss, au_loss=au_loss, last_feature_size=backbone.get_last_feature_size())
        self.backbone = backbone
        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        else:
            self.backbone.requires_grad_(True)
        self.backbone

    def _get_trainable_params(self):
        params = super()._get_trainable_params()
        if self.trainable:
            params += list(self.backbone.parameters())
        return params

    def forward(self, images):
        return self.backbone._forward(images)

    def train(self, b=True):
        super().train(b)
        if not self.trainable:
            self.backbone.eval()
        else:
            self.backbone.train(b)
        return self


class EmoBackboneDualLoss(EmoBackboneLoss):

    def __init__(self, device, backbone, trainable=False, clone_is_trainable=True, normalize_features=False, emo_feat_loss=None, au_loss=None):
        super().__init__(device, backbone, trainable, normalize_features=normalize_features, emo_feat_loss=emo_feat_loss, au_loss=au_loss)
        assert not trainable
        if not clone_is_trainable:
            raise ValueError('The second cloned backbone (used to be finetuned on renderings) is not trainable. Probably not what you want.')
        self.clone_is_trainable = clone_is_trainable
        self.trainable_backbone = copy.deepcopy(backbone)
        if not clone_is_trainable:
            self.trainable_backbone.requires_grad_(False)
            self.trainable_backbone.eval()
        else:
            self.trainable_backbone.requires_grad_(True)
        self.trainable_backbone

    def _get_trainable_params(self):
        trainable_params = super()._get_trainable_params()
        if self.clone_is_trainable:
            trainable_params += list(self.trainable_backbone.parameters())
        return trainable_params

    def _forward_output(self, images):
        return self.trainable_backbone._forward(images)

    def train(self, b=True):
        super().train(b)
        if not self.clone_is_trainable:
            self.trainable_backbone.eval()
        else:
            self.trainable_backbone.train(b)
        return self


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class VGGFace2(nn.Module):

    def __init__(self, pretrained_data='vggface2'):
        super(VGGFace2, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        checkpoint = '/ps/scratch/rdanecek/FaceRecognition/resnet50_ft_weight.pkl'
        load_state_dict(self.reg_model, checkpoint)
        self._freeze_layer(self.reg_model)

    def _freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.reg_model(x)
        out = out.view(out.size(0), -1)
        return out


cfgs = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [nn.ModuleList([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])]
            else:
                layers += [nn.ModuleList([conv2d, nn.ReLU(inplace=True)])]
            in_channels = v
    return nn.ModuleList(layers)


model_urls = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers(cfgs[cfg], batch_norm=batch_norm)
    if pretrained:
        archname = arch
        if batch_norm:
            archname += '_bn'
        state_dict = load_state_dict_from_url(model_urls[archname], progress=progress)
        state_dict2 = OrderedDict()
        for key in state_dict.keys():
            if 'features' in key:
                state_dict2[key[len('features.'):]] = state_dict[key]
        layers_ = []
        for bi, block in enumerate(layers):
            for layer in block:
                layers_ += [layer]
        net = nn.Sequential(*layers_)
        net.load_state_dict(state_dict2)
    return layers


class VGG19(nn.Module):

    def __init__(self, layer_activation_indices, batch_norm=False):
        super().__init__()
        self.layer_activation_indices = layer_activation_indices
        self.blocks = _vgg('vgg19', 'E', batch_norm=batch_norm, pretrained=True, progress=True)
        self.conv_block_indices = []
        self.layers = []
        for bi, block in enumerate(self.blocks):
            for layer in block:
                self.layers += [layer]
                if isinstance(layer, nn.Conv2d):
                    self.conv_block_indices += [bi]
        if len(self.layer_activation_indices) != len(set(layer_activation_indices).intersection(set(self.conv_block_indices))):
            raise ValueError('The specified layer indices are not of a conv block')
        self.net = nn.Sequential(*self.layers)
        self.net.eval()
        self.net.requires_grad_(False)

    def requires_grad_(self, requires_grad: 'bool'=True):
        return super().requires_grad_(False)

    def train(self, mode: 'bool'=True):
        return super().train(False)

    def forward(self, x):
        layer_outputs = {}
        for bi, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            if bi in self.layer_activation_indices:
                layer_outputs[bi] = x
        layer_outputs['final'] = x
        return layer_outputs


class VGG19Loss(nn.Module):

    def __init__(self, layer_activation_indices_weights, diff=torch.nn.functional.l1_loss, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.vgg19 = VGG19(sorted(layer_activation_indices_weights.keys()), batch_norm=batch_norm)
        self.layer_activation_indices_weights = layer_activation_indices_weights
        self.diff = diff

    def forward(self, x, y):
        feat_x = self.vgg19(x)
        feat_y = self.vgg19(y)
        out = {}
        loss = 0
        for idx, weight in self.layer_activation_indices_weights.items():
            d = self.diff(feat_x[idx], feat_y[idx])
            out[idx] = d
            loss += d * weight
        return loss, out


class Coma(torch.nn.Module):

    def __init__(self, config: 'dict', downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        config = copy.deepcopy(config)
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, config['num_input_features'])
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), num_nodes[i]) for i in range(len(num_nodes))])
        self.with_edge_weights = True
        self.conv_type = config['conv_type']
        self.conv_type_name = config['conv_type']['class']
        if 'params' not in config['conv_type'].keys() or not bool(config['conv_type']['params']):
            conv_kwargs = None
        else:
            conv_kwargs = config['conv_type']['params']
        if self.conv_type_name == 'ChebConv_Coma':
            None
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i + 1], self.K[i], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i - 1], self.filters[-i - 2], self.K[i], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'ChebConv':
            None
            conv_kwargs = conv_kwargs or {'normalization': 'sym'}
            self.conv_enc = torch.nn.ModuleList([ChebConv(self.filters[i], self.filters[i + 1], self.K[i], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([ChebConv(self.filters[-i - 1], self.filters[-i - 2], self.K[i], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GCNConv':
            None
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([GCNConv(self.filters[i], self.filters[i + 1], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([GCNConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'FeaStConv':
            None
            conv_kwargs = conv_kwargs or {}
            self.with_edge_weights = False
            self.conv_enc = torch.nn.ModuleList([FeaStConv(self.filters[i], self.filters[i + 1], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([FeaStConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GATConv':
            None
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([GATConv(self.filters[i], self.filters[i + 1], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([GATConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'SAGEConv':
            None
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([SAGEConv(self.filters[i], self.filters[i + 1], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([SAGEConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs) for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GraphConv':
            None
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([GraphConv(self.filters[i], self.filters[i + 1], **conv_kwargs) for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList([GraphConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs) for i in range(len(self.filters) - 1)])
        else:
            raise ValueError("Invalid convolution type: '%s'" % self.conv_type)
        self.conv_dec[-1].bias = None
        self.pool = Pool(treat_batch_dim_separately=self.conv_type_name == 'ChebConv_Coma')
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0] * self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1] * self.upsample_matrices[-1].shape[1])
        self.reset_parameters()
        self._D_edge_batch = None
        self._D_norm_batch = None
        self._D_sizes = None
        self._U_matrices_batch = None
        self._U_norm_batch = None
        self._U_sizes = None
        self._batch_size = None

    def _create_batched_edges(self, batch_size):
        if self.conv_type_name != 'ChebConv_Coma':
            if self._batch_size == batch_size:
                return
            self._batch_size = batch_size
            self._A_edge_index_batch = []
            self._A_norm_batch = []
            for i in range(len(self.A_edge_index)):
                num_vertices = self.adjacency_matrices[i].size()[0]
                repeated_edges = self.A_edge_index[i][:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_vertices * torch.arange(batch_size, device=self.A_edge_index[i].device, dtype=self.A_edge_index[i].dtype).reshape((1, batch_size, 1))
                self._A_edge_index_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                self._A_norm_batch += [None]
            self._D_edge_batch = []
            self._D_norm_batch = []
            self._D_sizes = []
            for i in range(len(self.downsample_matrices)):
                num_downsampled_vertices = self.downsample_matrices[i].size()[0]
                repeated_edges = self.downsample_matrices[i]._indices()[:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_downsampled_vertices * torch.arange(batch_size, device=self.downsample_matrices[i].device, dtype=torch.int64).reshape((1, batch_size, 1))
                self._D_edge_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                self._D_norm_batch += [self.downsample_matrices[i]._values().repeat(batch_size)]
                self._D_sizes += [[(self.downsample_matrices[i].size()[j] * batch_size) for j in range(self.downsample_matrices[i].ndim)]]
            self._U_edge_batch = []
            self._U_norm_batch = []
            self._U_sizes = []
            for i in range(len(self.upsample_matrices)):
                num_upsampled_vertices = self.upsample_matrices[i].size()[1]
                repeated_edges = self.upsample_matrices[i]._indices()[:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_upsampled_vertices * torch.arange(batch_size, device=self.upsample_matrices[i].device, dtype=torch.int64).reshape((1, batch_size, 1))
                self._U_edge_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                self._U_norm_batch += [self.upsample_matrices[i]._values().repeat(batch_size)]
                self._U_sizes += [[(self.upsample_matrices[i].size()[j] * batch_size) for j in range(self.upsample_matrices[i].ndim)]]
        else:
            self._A_edge_index_batch = self.A_edge_index
            self._A_norm_batch = self.A_norm
            self._D_edge_batch = [self.downsample_matrices[i]._indices() for i in range(len(self.downsample_matrices))]
            self._D_norm_batch = [self.downsample_matrices[i]._values() for i in range(len(self.downsample_matrices))]
            self._D_sizes = [list(self.downsample_matrices[i].size()) for i in range(len(self.downsample_matrices))]
            self._U_edge_batch = [self.upsample_matrices[i]._indices() for i in range(len(self.upsample_matrices))]
            self._U_norm_batch = [self.upsample_matrices[i]._values() for i in range(len(self.upsample_matrices))]
            self._U_sizes = [list(self.upsample_matrices[i].size()) for i in range(len(self.upsample_matrices))]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        self._create_batched_edges(batch_size)
        if self.conv_type_name == 'ChebConv_Coma':
            x = x.reshape(batch_size, -1, self.filters[0])
        x = self.encoder(x, batch_size)
        x = self.decoder(x)
        if self.conv_type_name == 'ChebConv_Coma':
            x = x.reshape(-1, self.filters[0])
        return x

    def encoder(self, x, batch_size=None):
        batch_size = batch_size or x.shape[0]
        if self._A_edge_index_batch is None:
            if self.conv_type_name == 'ChebConv_Coma':
                self._create_batched_edges(x.shape[0])
            else:
                self._create_batched_edges(1)
        for i in range(self.n_layers):
            if self.with_edge_weights:
                x = self.conv_enc[i](x, self._A_edge_index_batch[i], self._A_norm_batch[i])
            else:
                x = self.conv_enc[i](x, self._A_edge_index_batch[i])
            x = F.relu(x)
            x = self.pool(x, self._D_edge_batch[i], self._D_norm_batch[i], self._D_sizes[i])
        x = x.reshape(batch_size, self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        if self.conv_type_name == 'ChebConv_Coma':
            x = x.reshape(x.shape[0], -1, self.filters[-1])
        else:
            x = x.reshape(-1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self._U_edge_batch[-i - 1], self._U_norm_batch[-i - 1], self._U_sizes[-i - 1])
            if self.with_edge_weights:
                x = self.conv_dec[i](x, self._A_edge_index_batch[self.n_layers - i - 1], self._A_norm_batch[self.n_layers - i - 1])
            else:
                x = self.conv_dec[i](x, self._A_edge_index_batch[self.n_layers - i - 1])
            x = F.relu(x)
        if self.with_edge_weights:
            x = self.conv_dec[-1](x, self._A_edge_index_batch[-1], self._A_norm_batch[-1])
        else:
            x = self.conv_dec[-1](x, self._A_edge_index_batch[-1])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


class DecaMode(Enum):
    COARSE = 1
    DETAIL = 2


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=torch.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    angle = torch.norm(rot_vecs + 1e-08, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True, dtype=torch.float32, detach_pose_correctives=True):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    if detach_pose_correctives:
        pose_offsets = pose_offsets.detach()
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """ Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(self, config, v_template=None):
        super(FLAME, self).__init__()
        None
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        if v_template is not None:
            self.flame_model.v_template = v_template
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        default_shape = torch.zeros([self.batch_size, 300 - config.shape_params], dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape, requires_grad=False))
        default_exp = torch.zeros([self.batch_size, 100 - config.flame_expression_params], dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp, requires_grad=False))
        default_eyball_pose = torch.zeros([self.batch_size, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        self.use_3D_translation = config.use_3D_translation
        default_transl = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=False))
        self.register_buffer('v_template', to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype))
        shapedirs = self.flame_model.shapedirs
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=self.dtype))
        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
        with open(config.static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))
        lmk_faces_idx = static_embeddings.lmk_face_idx.astype(np.int64)
        self.register_buffer('lmk_faces_idx', torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords', torch.tensor(lmk_bary_coords, dtype=self.dtype))
        if self.use_face_contour:
            conture_embeddings = np.load(config.dynamic_landmark_embedding_path, allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx', dynamic_lmk_faces_idx)
            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords', dynamic_lmk_bary_coords)
            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """
        batch_size = vertices.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39))
        neg_mask = y_rot_angle.lt(0)
        mask = y_rot_angle.lt(-39)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None, transl=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        betas = torch.cat([shape_params, self.shape_betas, expression_params, self.expression_betas], dim=1)
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights)
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(vertices, full_pose, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords, self.neck_kin_chain, dtype=self.dtype)
            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        return vertices, landmarks


class FLAMETex(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            None
            exit()
        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


class FLAME_mediapipe(FLAME):

    def __init__(self, config):
        super().__init__(config)
        lmk_embeddings_mediapipe = np.load(config.flame_mediapipe_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        self.register_buffer('lmk_faces_idx_mediapipe', torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64), dtype=torch.long))
        self.register_buffer('lmk_bary_coords_mediapipe', torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'], dtype=self.dtype))

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        vertices, landmarks2d, landmarks3d = super().forward(shape_params, expression_params, pose_params, eye_pose_params)
        batch_size = shape_params.shape[0]
        lmk_faces_idx_mediapipe = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords_mediapipe = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(batch_size, -1, -1).contiguous()
        landmarks2d_mediapipe = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe)
        return vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe


class Generator(nn.Module):

    def __init__(self, latent_dim=100, out_channels=1, out_scale=1, sample_mode='bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(32, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, out_channels, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale


class AdaInUpConvBlock(nn.Module):

    def __init__(self, dim_in, dim_out, cond_dim, kernel_size=3, scale_factor=2, sample_mode='bilinear'):
        super().__init__()
        self.norm = AdaIN(cond_dim, dim_in)
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        if scale_factor > 0:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=sample_mode)
        else:
            self.upsample = None
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride=1, padding=1)

    def forward(self, x, condition):
        x = self.norm(x, condition)
        x = self.actv(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        return x


class GeneratorAdaIn(nn.Module):

    def __init__(self, latent_dim, condition_dim, out_channels=1, out_scale=1, sample_mode='bilinear'):
        super().__init__()
        self.out_scale = out_scale
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_block1 = AdaInUpConvBlock(128, 128, condition_dim, sample_mode=sample_mode)
        self.conv_block2 = AdaInUpConvBlock(128, 64, condition_dim, sample_mode=sample_mode)
        self.conv_block3 = AdaInUpConvBlock(64, 64, condition_dim, sample_mode=sample_mode)
        self.conv_block4 = AdaInUpConvBlock(64, 32, condition_dim, sample_mode=sample_mode)
        self.conv_block5 = AdaInUpConvBlock(32, 16, condition_dim, sample_mode=sample_mode)
        self.conv_block6 = AdaInUpConvBlock(16, out_channels, condition_dim, scale_factor=0)
        self.conv_blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6]
        self.out_actv = nn.Tanh()

    def forward(self, z, cond):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        for i, block in enumerate(self.conv_blocks):
            out = block(out, cond)
        img = self.out_actv(out)
        return img * self.out_scale


class BaseEncoder(nn.Module):

    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.feature_size = 2048
        self.outsize = outsize
        self._create_encoder()
        self.layers = nn.Sequential(nn.Linear(self.feature_size, 1024), nn.ReLU(), nn.Linear(1024, self.outsize))
        self.last_op = last_op

    def forward_features(self, inputs):
        return self.encoder(inputs)

    def forward_features_to_output(self, features):
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs, output_features=False):
        features = self.forward_features(inputs)
        parameters = self.forward_features_to_output(features)
        if not output_features:
            return parameters
        return parameters, features

    def _create_encoder(self):
        raise NotImplementedError()

    def reset_last_layer(self):
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)


class ResnetEncoder(BaseEncoder):

    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__(outsize, last_op)

    def _create_encoder(self):
        self.encoder = resnet.load_ResNet50Model()


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {'image_size': image_size, 'blur_radius': 0.0, 'faces_per_pixel': 1, 'bin_size': None, 'max_faces_per_bin': None, 'perspective_correct': False}
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(meshes_screen, image_size=raster_settings.image_size, blur_radius=raster_settings.blur_radius, faces_per_pixel=raster_settings.faces_per_pixel, bin_size=raster_settings.bin_size, max_faces_per_bin=raster_settings.max_faces_per_bin, perspective_correct=raster_settings.perspective_correct)
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):

    def __init__(self, image_size, obj_filename, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]
        uvfaces = faces.textures_idx[None, ...]
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.0
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)
        pi = np.pi
        constant_factor = torch.tensor([1 / np.sqrt(4 * pi), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), 2 * pi / 3 * np.sqrt(3 / (4 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * 3 * np.sqrt(5 / (12 * pi)), pi / 4 * (3 / 2) * np.sqrt(5 / (12 * pi)), pi / 4 * (1 / 2) * np.sqrt(5 / (4 * pi))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        """
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], rnage:[-1,1], projected vertices, in image space, for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        """
        batch_size = vertices.shape[0]
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(), face_normals], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        uvcoords_images = rendering[:, :3, :, :]
        grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            elif light_type == 'point':
                vertice_images = rendering[:, 6:9, :, :].detach()
                shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
                shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            else:
                shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
                shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.0
        outputs = {'images': images * alpha_images, 'albedo_images': albedo_images, 'alpha_images': alpha_images, 'pos_mask': pos_mask, 'shading_images': shading_images, 'grid': grid, 'normals': normals, 'normal_images': normal_images, 'transformed_normals': transformed_normals}
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        """
            sh_coeff: [bz, 9, 3]
        """
        N = normal_images
        sh = torch.stack([N[:, 0] * 0.0 + 1.0, N[:, 0], N[:, 1], N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2], N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * N[:, 2] ** 2 - 1], 1)
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)
        return shading

    def add_pointlight(self, vertices, normals, lights):
        """
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        """
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0.0, 1.0)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, images=None, detail_normal_images=None, lights=None):
        """
        -- rendering shape with detail normal map
        """
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1]])[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_colors.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(), face_normals], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        albedo_images = rendering[:, :3, :, :]
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0).float()
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images
        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2).contiguous()
        shaded_images = albedo_images * shading_images
        if images is None:
            shape_images = shaded_images * alpha_images + torch.zeros_like(shaded_images) * (1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)
        return shape_images

    def render_depth(self, transformed_vertices):
        """
        -- rendering depth
        """
        batch_size = transformed_vertices.shape[0]
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_normal(self, transformed_vertices, normals):
        """
        -- rendering normal
        """
        batch_size = normals.shape[0]
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        """
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices


class StarGANWrapper(torch.nn.Module):

    def __init__(self, cfg, stargan_repo=None):
        super().__init__()
        if isinstance(cfg, (str, Path)):
            self.args = OmegaConf.load(cfg)
            if self.args.wing_path is not None:
                self.args.wing_path = str(Path(stargan_repo) / self.args.wing_path)
                self.args.checkpoint_dir = str(Path(stargan_repo) / self.args.checkpoint_dir)
        else:
            self.args = cfg
        generator = build_generator(self.args)
        style_encoder = build_style_encoder(self.args)
        generator.requires_grad_(False)
        style_encoder.requires_grad_(False)
        self.nets_ema = Munch(generator=generator, style_encoder=style_encoder)
        fan = build_FAN(self.args)
        if fan is not None:
            self.nets_ema.fan = fan
        self._load_checkpoint('latest')

    @property
    def background_mode(self):
        return self.args.deca_background

    def _load_checkpoint(self, step):
        self.ckptios = [CheckpointIO(str(Path(self.args.checkpoint_dir) / '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]
        if isinstance(step, str):
            if step == 'latest':
                path = Path(self.ckptios[0].fname_template)
                ckpts = list(path.parent.glob('*.ckpt'))
                ckpts.sort(reverse=True)
                found = False
                for ckpt in ckpts:
                    split_name = ckpt.name.split('_')
                    if len(split_name) < 1:
                        None
                        continue
                    num = split_name[0]
                    step = int(num)
                    None
                    found = True
                    break
                if not found:
                    raise RuntimeError(f"Checkpoint not found in '{path.parent}'")
            else:
                raise ValueError(f"Invalid resume_iter value: '{step}'")
        if step is not None and not isinstance(step, int):
            raise ValueError(f"Invalid resume_iter value: '{step}' or type: '{type(step)}'")
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _normalize(self, img, mean, std, max_pixel_value=1.0):
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device=img.device)
        mean *= max_pixel_value
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1, 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32, device=img.device)
        std *= max_pixel_value
        if img.ndim == 4:
            std = std.view(1, std.numel(), 1, 1)
        else:
            std = std.view(std.size(), 1, 1)
        mean = mean
        std = std
        denominator = torch.reciprocal(std)
        img = img
        img = img - mean
        img = img * denominator
        return img

    def _denormalize(self, img, mean, std, max_pixel_value=1.0):
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device=img.device)
        mean *= max_pixel_value
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1, 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32, device=img.device)
        std *= max_pixel_value
        if img.ndim == 4:
            std = std.view(1, std.numel(), 1, 1)
        else:
            std = std.view(std.size(), 1, 1)
        denominator = torch.reciprocal(std)
        img = img / denominator
        img = img + mean
        return img

    def forward(self, sample):
        input_image = sample['input_image']
        input_image = F.interpolate(input_image, (self.args.img_size, self.args.img_size), mode='bilinear')
        image = self._normalize(input_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_ref_image = F.interpolate(sample['ref_image'], (self.args.img_size, self.args.img_size), mode='bilinear')
        ref_image = self._normalize(input_ref_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        target_domain_label = sample['target_domain']
        target_style = self.nets_ema.style_encoder(ref_image, target_domain_label)
        if hasattr(self.nets_ema, 'fan'):
            masks = self.nets_ema.fan.get_heatmap(image)
        else:
            masks = None
        translated_image = self.nets_ema.generator(image, target_style, masks=masks)
        translated_image = self._denormalize(translated_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        output_size = sample['input_image'].shape[2:4]
        translated_image = F.interpolate(translated_image, output_size, mode='bilinear')
        return translated_image


def create_swin_backbone(swin_cfg, num_classes, img_size, load_pretrained_swin=False, pretrained_model=None):
    """
    Returns a SWIN backbone with a head of size num_classes.
    """
    with open_dict(swin_cfg):
        swin_cfg.MODEL.NUM_CLASSES = num_classes
        swin_cfg.MODEL.SWIN.PATCH_SIZE = 4
        swin_cfg.MODEL.SWIN.IN_CHANS = 3
        swin_cfg.MODEL.SWIN.MLP_RATIO = 4.0
        swin_cfg.MODEL.SWIN.QKV_BIAS = True
        swin_cfg.MODEL.SWIN.QK_SCALE = None
        swin_cfg.MODEL.SWIN.APE = False
        swin_cfg.MODEL.SWIN.PATCH_NORM = True
        if 'DROP_RATE' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.DROP_RATE = 0.0
        if 'DROP_PATH_RATE' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.DROP_PATH_RATE = 0.1
        if 'DROP_PATH_RATE' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.LABEL_SMOOTHING = 0.1
        swin_cfg.DATA = {}
        swin_cfg.DATA.IMG_SIZE = img_size
        swin_cfg.TRAIN = {}
        swin_cfg.TRAIN.USE_CHECKPOINT = False
    swin = build_model(swin_cfg)
    if load_pretrained_swin:
        path_to_model = swin_path / 'pretrained_models' / (pretrained_model + '.pth')
        state_dict = torch.load(path_to_model)
        del state_dict['model']['head.weight']
        del state_dict['model']['head.bias']
        swin.load_state_dict(state_dict['model'], strict=False)
        None
    return swin


def swin_cfg_from_name(name):
    swin_cfg = OmegaConf.load(Path(swin_path) / 'configs' / (name + '.yaml'))
    OmegaConf.set_struct(swin_cfg, True)
    return swin_cfg


class SwinEncoder(BaseEncoder):

    def __init__(self, swin_type, img_size, outsize, last_op=None):
        self.swin_type = swin_type
        self.img_size = img_size
        super().__init__(outsize, last_op)

    def _create_encoder(self):
        swin_cfg = swin_cfg_from_name(self.swin_type)
        self.encoder = create_swin_backbone(swin_cfg, self.feature_size, self.img_size, load_pretrained_swin=True, pretrained_model=self.swin_type)


class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()
        self.perceptual_loss = None
        self.id_loss = None
        self.vgg_loss = None
        self._reconfigure(config)
        self._reinitialize()

    def get_input_image_size(self):
        return self.config.image_size, self.config.image_size

    def _reconfigure(self, config):
        self.config = config
        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        self.n_detail = config.n_detail
        self.n_detail_emo = config.n_detail_emo if 'n_detail_emo' in config.keys() else 0
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp
        self.mode = DecaMode[str(config.mode).upper()]
        self._create_detail_generator()
        self._init_deep_losses()
        self._setup_neural_rendering()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self._init_deep_losses()
        self.face_attr_mask = util.load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _get_num_shape_params(self):
        return self.config.n_shape

    def _init_deep_losses(self):
        """
        Initialize networks for deep losses
        """
        if 'mrfwr' not in self.config.keys() or self.config.mrfwr == 0:
            self.perceptual_loss = None
        elif self.perceptual_loss is None:
            self.perceptual_loss = lossfunc.IDMRFLoss().eval()
            self.perceptual_loss.requires_grad_(False)
        if 'idw' not in self.config.keys() or self.config.idw == 0:
            self.id_loss = None
        elif self.id_loss is None:
            id_metric = self.config.id_metric if 'id_metric' in self.config.keys() else None
            id_trainable = self.config.id_trainable if 'id_trainable' in self.config.keys() else False
            self.id_loss_start_step = self.config.id_loss_start_step if 'id_loss_start_step' in self.config.keys() else 0
            self.id_loss = lossfunc.VGGFace2Loss(self.config.pretrained_vgg_face_path, id_metric, id_trainable)
            self.id_loss.freeze_nontrainable_layers()
        if 'vggw' not in self.config.keys() or self.config.vggw == 0:
            self.vgg_loss = None
        elif self.vgg_loss is None:
            vgg_loss_batch_norm = 'vgg_loss_batch_norm' in self.config.keys() and self.config.vgg_loss_batch_norm
            self.vgg_loss = VGG19Loss(dict(zip(self.config.vgg_loss_layers, self.config.lambda_vgg_layers)), batch_norm=vgg_loss_batch_norm).eval()
            self.vgg_loss.requires_grad_(False)

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path, uv_size=self.config.uv_size)
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)
        if 'displacement_mask' in self.config.keys():
            displacement_mask_ = 1 - np.load(self.config.displacement_mask).astype(np.float32)
            displacement_mask_ = torch.from_numpy(displacement_mask_)[None, None, ...].contiguous()
            displacement_mask_ = F.interpolate(displacement_mask_, [self.config.uv_size, self.config.uv_size])
            self.register_buffer('displacement_mask', displacement_mask_)
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
            None
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def uses_texture(self):
        if 'use_texture' in self.config.keys():
            return self.config.use_texture
        return True

    def _disable_texture(self, remove_from_model=False):
        self.config.use_texture = False
        if remove_from_model:
            self.flametex = None

    def _enable_texture(self):
        self.config.use_texture = True

    def _has_neural_rendering(self):
        return hasattr(self.config, 'neural_renderer') and bool(self.config.neural_renderer)

    def _setup_neural_rendering(self):
        if self._has_neural_rendering():
            if self.config.neural_renderer.class_ == 'StarGAN':
                None
                self.image_translator = StarGANWrapper(self.config.neural_renderer.cfg, self.config.neural_renderer.stargan_repo)
            else:
                raise ValueError(f"Unsupported neural renderer class '{self.config.neural_renderer.class_}'")
            if self.image_translator.background_mode == 'input':
                if self.config.background_from_input not in [True, 'input']:
                    raise NotImplementedError('The background mode of the neural renderer and deca is not synchronized. Background should be inpainted from the input')
            elif self.image_translator.background_mode == 'black':
                if self.config.background_from_input not in [False, 'black']:
                    raise NotImplementedError('The background mode of the neural renderer and deca is not synchronized. Background should be black.')
            elif self.image_translator.background_mode == 'none':
                if self.config.background_from_input not in ['none']:
                    raise NotImplementedError('The background mode of the neural renderer and deca is not synchronized. The background should not be handled')
            else:
                raise NotImplementedError(f"Unsupported mode of the neural renderer backroungd: '{self.image_translator.background_mode}'")

    def _create_detail_generator(self):
        if hasattr(self, 'D_detail'):
            if (not 'detail_conditioning_type' in self.config.keys() or self.config.detail_conditioning_type == 'concat') and isinstance(self.D_detail, Generator):
                return
            if self.config.detail_conditioning_type == 'adain' and isinstance(self.D_detail, GeneratorAdaIn):
                return
            None
            del self.D_detail
        if not 'detail_conditioning_type' in self.config.keys() or str(self.config.detail_conditioning_type).lower() == 'concat':
            None
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_detail_emo + self.n_cond, out_channels=1, out_scale=0.01, sample_mode='bilinear')
        elif str(self.config.detail_conditioning_type).lower() == 'adain':
            None
            self.D_detail = GeneratorAdaIn(self.n_detail + self.n_detail_emo, self.n_cond, out_channels=1, out_scale=0.01, sample_mode='bilinear')
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _create_model(self):
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type
        if e_flame_type == 'ResnetEncoder':
            self.E_flame = ResnetEncoder(outsize=self.n_param)
        elif e_flame_type[:4] == 'swin':
            self.E_flame = SwinEncoder(outsize=self.n_param, img_size=self.config.image_size, swin_type=e_flame_type)
        else:
            raise ValueError(f"Invalid 'e_flame_type' = {e_flame_type}")
        import copy
        flame_cfg = copy.deepcopy(self.config)
        flame_cfg.n_shape = self._get_num_shape_params()
        if 'flame_mediapipe_lmk_embedding_path' not in flame_cfg.keys():
            self.flame = FLAME(flame_cfg)
        else:
            self.flame = FLAME_mediapipe(flame_cfg)
        if self.uses_texture():
            self.flametex = FLAMETex(self.config)
        else:
            self.flametex = None
        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in self.config.keys():
            e_detail_type = self.config.e_detail_type
        if e_detail_type == 'ResnetEncoder':
            self.E_detail = ResnetEncoder(outsize=self.n_detail + self.n_detail_emo)
        elif e_flame_type[:4] == 'swin':
            self.E_detail = SwinEncoder(outsize=self.n_detail + self.n_detail_emo, img_size=self.config.image_size, swin_type=e_detail_type)
        else:
            raise ValueError(f"Invalid 'e_detail_type'={e_detail_type}")
        self._create_detail_generator()

    def _get_coarse_trainable_parameters(self):
        None
        return list(self.E_flame.parameters())

    def _get_detail_trainable_parameters(self):
        trainable_params = []
        if self.config.train_coarse:
            trainable_params += self._get_coarse_trainable_parameters()
            None
        trainable_params += list(self.E_detail.parameters())
        None
        trainable_params += list(self.D_detail.parameters())
        None
        return trainable_params

    def train(self, mode: 'bool'=True):
        super().train(mode)
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_flame.train()
                self.E_detail.eval()
                self.D_detail.eval()
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    self.E_flame.train()
                else:
                    self.E_flame.eval()
                self.E_detail.train()
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_flame.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        self.flame.eval()
        if self.flametex is not None:
            self.flametex.eval()
        return self

    def _load_old_checkpoint(self):
        """
        Loads the DECA model weights from the original DECA implementation: 
        https://github.com/YadiraF/DECA 
        """
        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            None
            checkpoint = torch.load(model_path)
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            if 'E_detail' in checkpoint.keys():
                util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
                util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            self.start_epoch = 0
            self.start_iter = 0
        else:
            None
            self.start_epoch = 0
            self.start_iter = 0

    def _encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        """
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        """
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam, self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list, None

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals. 
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()
        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :, :] * uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        return uv_detail_normals, uv_coarse_vertices

    def visualize(self, visdict, savepath, catdim=1):
        grids = {}
        for key in visdict:
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), catdim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image

    def create_mesh(self, opdict, dense_template):
        """
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        if 'uv_texture_gt' in opdict.keys():
            texture = util.tensor2image(opdict['uv_texture_gt'][i])
        else:
            texture = None
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        if 'uv_detail_normals' in opdict.keys():
            normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
            texture = texture[:, :, [2, 1, 0]]
            normals = opdict['normals'][i].cpu().numpy()
            displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
            dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, dense_template)
        else:
            normal_map = None
            dense_vertices = None
            dense_colors = None
            dense_faces = None
        return vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors

    def save_obj(self, filename, opdict, dense_template, mode='detail'):
        if mode not in ['coarse', 'detail', 'both']:
            raise ValueError(f"Invalid mode '{mode}. Expected modes are: 'coarse', 'detail', 'both'")
        vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors = self.create_mesh(opdict, dense_template)
        if mode == 'both':
            if isinstance(filename, list):
                filename_coarse = filename[0]
                filename_detail = filename[1]
            else:
                filename_coarse = filename
                filename_detail = filename.replace('.obj', '_detail.obj')
        elif mode == 'coarse':
            filename_coarse = filename
        else:
            filename_detail = filename
        if mode in ['coarse', 'both']:
            util.write_obj(str(filename_coarse), vertices, faces, texture=texture, uvcoords=uvcoords, uvfaces=uvfaces, normal_map=normal_map)
        if mode in ['detail', 'both']:
            util.write_obj(str(filename_detail), dense_vertices, dense_faces, colors=dense_colors, inverse_face_order=True)


class EmoNetRegressor(torch.nn.Module):

    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.emonet = get_emonet().eval()
        self.input_image_size = 256, 256
        self.feature_to_use = 'emo_feat_2'
        if self.feature_to_use == 'emo_feat_2':
            self.emonet_feature_size = 256
            self.fc_size = 256
        else:
            raise NotImplementedError(f"Not yet implemented for feature '{self.feature_to_use}'")
        self.layers = torch.nn.Sequential(torch.nn.Linear(self.emonet_feature_size, self.fc_size), torch.nn.ReLU(), torch.nn.Linear(self.fc_size, outsize))
        self.last_op = last_op

    def forward(self, images):
        images = F.interpolate(images, self.input_image_size, mode='bilinear')
        out = self.emonet(images, intermediate_features=True)
        out = self.layers(out[self.feature_to_use])
        return out


class EmonetRegressorStatic(EmoNetRegressor):

    def __init__(self, outsize, last_op=None):
        super().__init__(outsize, last_op)
        self.emonet.requires_grad_(False)
        self.emonet.eval()

    def train(self, mode=True):
        self.emonet.eval()
        self.layers.train(mode)
        return self


class SecondHeadResnet(nn.Module):

    def __init__(self, enc: 'BaseEncoder', outsize, last_op=None):
        super().__init__()
        self.resnet = enc
        self.layers = nn.Sequential(nn.Linear(self.resnet.feature_size, 1024), nn.ReLU(), nn.Linear(1024, outsize))
        if last_op == 'same':
            self.last_op = self.resnet.last_op
        else:
            self.last_op = last_op

    def forward_features(self, inputs):
        out1, features = self.resnet(inputs, output_features=True)
        return out1, features

    def forward_features_to_output(self, features):
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs):
        out1, features = self.forward_features()
        out2 = self.forward_features_to_output(features)
        return out1, out2

    def train(self, mode: 'bool'=True):
        self.layers.train(mode)
        return self

    def reset_last_layer(self):
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality. 
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        super()._create_model()
        self.E_flame.requires_grad_(False)
        if self.config.expression_backbone == 'deca_parallel':
            self.E_expression = SecondHeadResnet(self.E_flame, self.n_exp_param, 'same')
        elif self.config.expression_backbone == 'deca_clone':
            self.E_expression = ResnetEncoder(self.n_exp_param)
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        elif self.config.expression_backbone == 'emonet_trainable':
            self.E_expression = EmoNetRegressor(self.n_exp_param)
        elif self.config.expression_backbone == 'emonet_static':
            self.E_expression = EmonetRegressorStatic(self.n_exp_param)
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")
        if self.config.get('zero_out_last_enc_layer', False):
            self.E_expression.reset_last_layer()

    def _get_coarse_trainable_parameters(self):
        None
        return list(self.E_expression.parameters())

    def _reconfigure(self, config):
        super()._reconfigure(config)
        self.n_exp_param = self.config.n_exp
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def _encode_flame(self, images):
        if self.config.expression_backbone == 'deca_parallel':
            return self.E_expression(images)
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]
        deca_code_list, _ = super().decompose_code(deca_code)
        exp_idx = 2
        pose_idx = 3
        deca_code_list_copy = deca_code_list.copy()
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:, 3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code
        return deca_code_list, deca_code_list_copy

    def train(self, mode: 'bool'=True):
        super().train(mode)
        self.E_flame.eval()
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_expression.train()
                self.E_detail.eval()
                self.D_detail.eval()
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    self.E_expression.train()
                else:
                    self.E_expression.eval()
                self.E_detail.train()
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_expression.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        return self


class EMICA(ExpDECA):

    def __init__(self, config):
        self.use_mica_shape_dim = True
        self.mica_cfg = get_cfg_defaults()
        super().__init__(config)

    def _create_model(self):
        super()._create_model()
        if Path(self.config.mica_model_path).exists():
            mica_path = self.config.mica_model_path
        else:
            mica_path = get_path_to_assets() / self.config.mica_model_path
            assert mica_path.exists(), f"MICA model path does not exist: '{mica_path}'"
        self.mica_cfg.pretrained_model_path = str(mica_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.E_mica = MICA(self.mica_cfg, device, str(mica_path), instantiate_flame=False)
        self.E_mica.requires_grad_(False)
        self.E_mica.testing = True
        if self.config.mica_preprocessing:
            self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(224, 224))

    def _get_num_shape_params(self):
        if self.use_mica_shape_dim:
            return self.mica_cfg.model.n_shape
        return self.config.n_shape

    def _get_coarse_trainable_parameters(self):
        return super()._get_coarse_trainable_parameters()

    def train(self, mode: 'bool'=True):
        super().train(mode)
        self.E_mica.train(False)

    def _encode_flame(self, images):
        if self.config.mica_preprocessing:
            mica_image = self._dirty_image_preprocessing(images)
        else:
            mica_image = F.interpolate(images, (112, 112), mode='bilinear', align_corners=False)
        deca_code, exp_deca_code = super()._encode_flame(images)
        mica_code = self.E_mica.encode(images, mica_image)
        mica_code = self.E_mica.decode(mica_code, predict_vertices=False)
        return deca_code, exp_deca_code, mica_code['pred_shape_code']

    def _dirty_image_preprocessing(self, input_image):
        image = input_image.detach().clone().cpu().numpy() * 255.0
        image = image.transpose((0, 2, 3, 1))
        min_det_score = 0.5
        image_list = list(image)
        aligned_image_list = []
        for i, img in enumerate(image_list):
            bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                aimg = resize(img, output_shape=(112, 112), preserve_range=True)
                aligned_image_list.append(aimg)
                raise RuntimeError('No faces detected')
                continue
            i = get_center(bboxes, image)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)
            aligned_image_list.append(aimg)
        aligned_images = np.array(aligned_image_list)
        aligned_images = aligned_images.transpose((0, 3, 1, 2))
        aligned_images = torch.from_numpy(aligned_images)
        return aligned_images

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]
        mica_code = code[2]
        code_list, deca_code_list_copy = super().decompose_code((deca_code, expdeca_code))
        id_idx = 0
        if self.use_mica_shape_dim:
            code_list[id_idx] = mica_code
        else:
            code_list[id_idx] = mica_code[..., :self.config.n_shape]
        return code_list, deca_code_list_copy


class FLAMETex_trainable(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(self, config):
        super(FLAMETex_trainable, self).__init__()
        tex_params = config.tex_params
        texture_model = np.load(config.tex_path)
        num_tex_pc = texture_model['PC'].shape[-1]
        tex_shape = texture_model['MU'].shape
        MU = torch.from_numpy(np.reshape(texture_model['MU'], (1, -1))).float()[None, ...]
        PC = torch.from_numpy(np.reshape(texture_model['PC'], (-1, num_tex_pc))[:, :tex_params]).float()[None, ...]
        self.register_buffer('MU', MU)
        self.register_buffer('PC', PC)
        if 'specMU' in texture_model.files:
            specMU = torch.from_numpy(np.reshape(texture_model['specMU'], (1, -1))).float()[None, ...]
            specPC = torch.from_numpy(np.reshape(texture_model['specPC'], (-1, num_tex_pc)))[:, :tex_params].float()[None, ...]
            self.register_buffer('specMU', specMU)
            self.register_buffer('specPC', specPC)
            self.isspec = True
        else:
            self.isspec = False
        self.register_parameter('PC_correction', nn.Parameter(torch.zeros_like(PC)))

    def forward(self, texcode):
        diff_albedo = self.MU + (self.PC * texcode[:, None, :]).sum(-1) + (self.PC_correction * texcode[:, None, :]).sum(-1)
        if self.isspec:
            spec_albedo = self.specMU + (self.specPC * texcode[:, None, :]).sum(-1)
            texture = diff_albedo + spec_albedo
        else:
            texture = diff_albedo
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.InstanceNorm2d(in_planes), nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class EmoNetHead(nn.Module):

    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, input_image_only=False, temporal_smoothing=False):
        super().__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.n_blocks = n_blocks
        self.input_image_only = input_image_only
        self.attention = attention
        if input_image_only and attention:
            raise ValueError("Options 'input_image_only' and 'attention' cannot be both activated")
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False
        if self.temporal_smoothing:
            self.n_temporal_states = 5
            self.init_smoothing = True
            self.temporal_weights = torch.Tensor([0.1, 0.1, 0.15, 0.25, 0.4]).unsqueeze(0).unsqueeze(2)
        self._create_Emo()

    def _create_Emo(self):
        if self.input_image_only:
            n_in_features = 3
        elif self.attention:
            n_in_features = 256 * (self.num_modules + 1)
        else:
            n_in_features = 256 * (self.num_modules + 1) + 68
        n_features = [(256, 256)] * self.n_blocks
        self.emo_convs = []
        self.conv1x1_input_emo_2 = nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, self.n_expression + self.n_reg))
        self.emo_parameters = list(set(self.parameters()).difference(set(self.fan_parameters)))
        self.emo_modules = list(set(self.modules()).difference(set(self.fan_modules)))

    def forward(self, x, hg_features=None, tmp_out=None, reset_smoothing=False, intermediate_features=False):
        if self.init_smoothing:
            self.init_smoothing = False
            self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, self.n_expression + self.n_reg)
        if reset_smoothing:
            self.temporal_state = self.temporal_state.zeros_()
        hg_features = torch.cat(tuple(hg_features), dim=1)
        if self.input_image_only:
            assert hg_features is None and tmp_out is None
            emo_feat = x
        elif self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features *= mask
            emo_feat = torch.cat((x, hg_features), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features, tmp_out], dim=1)
        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        if intermediate_features:
            emo_feat2 = final_features
        final_features = self.emo_fc_2(final_features)
        if self.temporal_smoothing:
            with torch.no_grad():
                self.temporal_state[:, :-1, :] = self.temporal_state[:, 1:, :]
                self.temporal_state[:, -1, :] = final_features
                final_features = torch.sum(self.temporal_weights * self.temporal_state, dim=1)
        res = {'heatmap': tmp_out, 'expression': final_features[:, :-2], 'valence': final_features[:, -2], 'arousal': final_features[:, -1]}
        if intermediate_features:
            res['emo_feat'] = emo_feat
            res['emo_feat_2'] = emo_feat2
        return res


class MLP(torch.nn.Module):

    def __init__(self, in_size: 'int', out_size: 'int', hidden_layer_sizes: 'list', hidden_activation=None, batch_norm=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.batch_norm = batch_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        hidden_activation = hidden_activation or LeakyReLU(0.2)
        self.hidden_activation = hidden_activation
        self._build_network()

    def _build_network(self):
        layers = []
        layer_sizes = [self.in_size] + self.hidden_layer_sizes
        for i in range(1, len(layer_sizes)):
            layers += [Linear(layer_sizes[i - 1], layer_sizes[i])]
            if self.batch_norm is not None:
                layers += [self.batch_norm(layer_sizes[i])]
            layers += [self.hidden_activation]
        layers += [Linear(layer_sizes[-1], self.out_size)]
        self.model = Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y


def _get_step_loss_weights(v_loss, a_loss, va_loss, scheme, training):
    va_loss_weights = {}
    for key in v_loss:
        va_loss_weights[key] = v_loss[key]
    for key in a_loss:
        va_loss_weights[key] = a_loss[key]
    for key in va_loss:
        va_loss_weights[key] = va_loss[key]
    if scheme is not None:
        if training and scheme == 'shake':
            for key in va_loss_weights:
                va_loss_weights[key] = np.random.rand(1)[0]
            total_w = 0.0
            for key in va_loss_weights:
                total_w += va_loss_weights[key]
            for key in va_loss_weights:
                va_loss_weights[key] /= total_w
        elif scheme == 'norm':
            total_w = 0.0
            for key in va_loss_weights:
                total_w += va_loss_weights[key]
            for key in va_loss_weights:
                va_loss_weights[key] /= total_w
    return va_loss_weights


def add_cfg_if_missing(cfg, name, default):
    if name not in cfg.keys():
        cfg[name] = default
    return cfg


def ACC_torch(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean(torch.eq(ground_truth.int(), predictions.int()).float())


def exp_loss(loss, pred, gt, class_weight, metrics, losses, expression_balancing, num_classes, pred_prefix=''):
    if pred[pred_prefix + 'expr_classification'] is not None:
        if class_weight.shape[0] != num_classes:
            weight = None
        elif expression_balancing:
            weight = class_weight
        else:
            weight = torch.ones_like(class_weight)
        if (num_classes <= gt['expr_classification'].max()).any():
            None
            gt['expr_classification'][gt['expr_classification'] >= num_classes] = num_classes - 1
        metrics[pred_prefix + 'expr_nll'] = F.nll_loss(pred[pred_prefix + 'expr_classification'], gt['expr_classification'][:, 0], None)
        if weight is not None:
            metrics[pred_prefix + 'expr_weighted_nll'] = F.nll_loss(pred[pred_prefix + 'expr_classification'], gt['expr_classification'][:, 0], class_weight)
        else:
            metrics[pred_prefix + 'expr_weighted_nll'] = F.nll_loss(pred[pred_prefix + 'expr_classification'], gt['expr_classification'][:, 0], None)
        metrics[pred_prefix + 'expr_acc'] = ACC_torch(torch.argmax(pred[pred_prefix + 'expr_classification'], dim=1), gt['expr_classification'][:, 0])
        if loss is not None:
            if callable(loss):
                losses[pred_prefix + 'expr'] = loss(pred[pred_prefix + 'expr_classification'], gt['expr_classification'][:, 0], weight)
            elif isinstance(loss, dict):
                for name, weight in loss.items():
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * weight
            else:
                raise RuntimeError(f"Uknown expression loss '{loss}'")
    return losses, metrics


def loss_from_cfg(config, loss_name):
    if loss_name in config.keys():
        if isinstance(config[loss_name], str):
            loss = class_from_str(config[loss_name], sys.modules[__name__])
        else:
            cont = OmegaConf.to_container(config[loss_name])
            if isinstance(cont, list):
                loss = {name: (1.0) for name in cont}
            elif isinstance(cont, dict):
                loss = cont
            else:
                raise ValueError(f"Unkown type of loss '{type(cont)}' for loss '{loss_name}'")
    else:
        loss = None
    return loss


def weighted_avg_and_std_torch(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    weighted_mean = torch.sum(weights * values)
    weighted_std = torch.mean(weights * (values - weighted_mean) ** 2)
    return weighted_mean, torch.sqrt(weighted_std)


def PCC_torch(ground_truth, predictions, batch_first=True, weights=None):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2
    if batch_first:
        dim = -1
    else:
        dim = 0
    if weights is None:
        centered_x = ground_truth - ground_truth.mean(dim=dim, keepdim=True)
        centered_y = predictions - predictions.mean(dim=dim, keepdim=True)
        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        x_std = ground_truth.std(dim=dim, keepdim=True)
        y_std = predictions.std(dim=dim, keepdim=True)
    else:
        weights = weights / weights.sum()
        centered_x, x_std = weighted_avg_and_std_torch(ground_truth, weights)
        centered_y, y_std = weighted_avg_and_std_torch(predictions, weights)
        covariance = (weights * centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (ground_truth.shape[dim] - 1)
    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr


def CCC_torch(ground_truth, predictions, batch_first=False, weights=None):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2
    if weights is not None:
        weights = weights / weights.sum()
        mean_pred, std_pred = weighted_avg_and_std_torch(predictions, weights)
        mean_gt, std_gt = weighted_avg_and_std_torch(ground_truth, weights)
    else:
        mean_pred = torch.mean(predictions)
        mean_gt = torch.mean(ground_truth)
        std_pred = torch.std(predictions)
        std_gt = torch.std(ground_truth)
    pearson = PCC_torch(ground_truth, predictions, batch_first=batch_first)
    return 2.0 * pearson * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)


def SAGR_torch(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean(torch.eq(torch.sign(ground_truth), torch.sign(predictions)).float())


def v_or_a_loss(loss, pred, gt, term_weights, metrics, losses, measure, pred_prefix='', permit_dropping_corr=False, sample_weights=None):
    if measure not in ['valence', 'arousal']:
        raise ValueError(f'Invalid measure {measure}')
    measure_label = pred_prefix + measure
    if pred[pred_prefix + measure] is not None:
        metrics[pred_prefix + f'{measure[0]}_mae'] = F.l1_loss(pred[measure_label], gt[measure])
        metrics[pred_prefix + f'{measure[0]}_mse'] = F.mse_loss(pred[measure_label], gt[measure])
        metrics[pred_prefix + f'{measure[0]}_rmse'] = torch.sqrt(metrics[pred_prefix + f'{measure[0]}_mse'])
        if sample_weights is not None:
            metrics[pred_prefix + f'{measure[0]}_mse_weighted'] = (sample_weights * F.mse_loss(pred[measure_label], gt[measure], reduction='none')).mean()
            metrics[pred_prefix + f'{measure[0]}_rmse_weighted'] = torch.sqrt(metrics[pred_prefix + f'{measure[0]}_mse_weighted'])
        if gt[measure].numel() >= 2:
            metrics[pred_prefix + f'{measure[0]}_pcc'] = PCC_torch(pred[measure_label], gt[measure], batch_first=False)[0]
            metrics[pred_prefix + f'{measure[0]}_ccc'] = CCC_torch(pred[measure_label], gt[measure], batch_first=False)[0]
            if metrics[pred_prefix + f'{measure[0]}_pcc'].isnan().any().item():
                None
                metrics[pred_prefix + f'{measure[0]}_pcc'] = torch.zeros_like(metrics[pred_prefix + f'{measure[0]}_pcc'])
            if metrics[pred_prefix + f'{measure[0]}_ccc'].isnan().any().item():
                None
                metrics[pred_prefix + f'{measure[0]}_ccc'] = torch.zeros_like(metrics[pred_prefix + f'{measure[0]}_pcc'])
            if sample_weights is not None:
                metrics[pred_prefix + f'{measure[0]}_pcc_weighted'] = PCC_torch(pred[measure_label], gt[measure], batch_first=False, weights=sample_weights)
                metrics[pred_prefix + f'{measure[0]}_ccc_weighted'] = CCC_torch(pred[measure_label], gt[measure], batch_first=False, weights=sample_weights)
                if metrics[pred_prefix + f'{measure[0]}_pcc_weighted'].isnan().any().item():
                    metrics[pred_prefix + f'{measure[0]}_pcc_weighted'] = torch.zeros_like(metrics[pred_prefix + f'{measure[0]}_pcc_weighted'])
                    None
                if metrics[pred_prefix + f'{measure[0]}_ccc_weighted'].isnan().any().item():
                    metrics[pred_prefix + f'{measure[0]}_ccc_weighted'] = torch.zeros_like(metrics[pred_prefix + f'{measure[0]}_pcc_weighted'])
                    None
        elif permit_dropping_corr:
            pass
        else:
            raise RuntimeError('Cannot compute correlation for a single sample')
        metrics[pred_prefix + f'{measure[0]}_sagr'] = SAGR_torch(pred[measure_label], gt[measure])
        if loss is not None:
            if callable(loss):
                losses[pred_prefix + measure[0]] = loss(pred[measure_label], gt[measure])
            elif isinstance(loss, dict):
                for name, weight in loss.items():
                    if permit_dropping_corr and pred_prefix + name not in metrics.keys():
                        continue
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * term_weights[name]
            else:
                raise RuntimeError(f"Uknown {measure} loss '{loss}'")
    return losses, metrics


def va_loss(loss, pred, gt, weights, metrics, losses, pred_prefix='', permit_dropping_corr=False, sample_weights=None):
    if pred[pred_prefix + 'valence'] is not None and pred[pred_prefix + 'arousal'] is not None:
        va_pred = torch.cat([pred[pred_prefix + 'valence'], pred[pred_prefix + 'arousal']], dim=1)
        va_gt = torch.cat([gt['valence'], gt['arousal']], dim=1)
        metrics[pred_prefix + 'va_mae'] = F.l1_loss(va_pred, va_gt)
        metrics[pred_prefix + 'va_mse'] = F.mse_loss(va_pred, va_gt)
        metrics[pred_prefix + 'va_rmse'] = torch.sqrt(metrics[pred_prefix + 'va_mse'])
        if pred_prefix + 'a_pcc' in metrics.keys():
            metrics[pred_prefix + 'va_lpcc'] = (1.0 - 0.5 * (metrics[pred_prefix + 'a_pcc'] + metrics[pred_prefix + 'v_pcc']))[0]
            metrics[pred_prefix + 'va_lccc'] = (1.0 - 0.5 * (metrics[pred_prefix + 'a_ccc'] + metrics[pred_prefix + 'v_ccc']))[0]
        elif permit_dropping_corr:
            pass
        else:
            raise RuntimeError(f"Missing computed correlation for the combined correlation loss: '{pred_prefix + 'a_pcc'}'")
        if sample_weights is not None:
            if pred_prefix + 'a_pcc_weighted' in metrics.keys():
                metrics[pred_prefix + 'va_lpcc_weighted'] = (1.0 - 0.5 * (metrics[pred_prefix + 'a_pcc_weighted'] + metrics[pred_prefix + 'v_pcc_weighted']))[0]
                metrics[pred_prefix + 'va_lccc_weighted'] = (1.0 - 0.5 * (metrics[pred_prefix + 'a_ccc_weighted'] + metrics[pred_prefix + 'v_ccc_weighted']))[0]
            elif permit_dropping_corr:
                pass
            else:
                raise RuntimeError(f"Missing computed correlation for the combined weighted correlation loss: '{pred_prefix + 'a_pcc'}'")
        if loss is not None:
            if callable(loss):
                losses[pred_prefix + 'va'] = loss(va_pred, va_gt)
            elif isinstance(loss, dict):
                for name, weight in loss.items():
                    if permit_dropping_corr and pred_prefix + name not in metrics.keys():
                        continue
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * weights[name]
            else:
                raise RuntimeError(f"Uknown expression loss '{loss}'")
    return losses, metrics


class EmotionMLP(torch.nn.Module):

    def __init__(self, config, deca_cfg):
        super().__init__()
        self.config = config
        in_size = 0
        if self.config.use_identity:
            in_size += deca_cfg.n_shape
        if self.config.use_expression:
            in_size += deca_cfg.n_exp
        if self.config.use_global_pose:
            in_size += 3
        if self.config.use_jaw_pose:
            in_size += 3
        if self.config.use_detail_code:
            self.n_detail = deca_cfg.n_detail
            in_size += deca_cfg.n_detail
        else:
            self.n_detail = None
        if 'use_detail_emo_code' in self.config.keys() and self.config.use_detail_emo_code:
            self.n_detail_emo = deca_cfg.n_detail_emo
            in_size += deca_cfg.n_detail_emo
        else:
            self.n_detail_emo = None
        hidden_layer_sizes = config.num_mlp_layers * [in_size]
        out_size = 0
        if self.config.predict_expression:
            self.num_classes = self.config.data.n_expression if 'n_expression' in self.config.data.keys() else 9
            out_size += self.num_classes
        if self.config.predict_valence:
            out_size += 1
        if self.config.predict_arousal:
            out_size += 1
        if 'mlp_norm_layer' in self.config.keys():
            batch_norm = class_from_str(self.config.mlp_norm_layer, sys.modules[__name__])
        else:
            batch_norm = None
        self.mlp = MLP(in_size, out_size, hidden_layer_sizes, batch_norm=batch_norm)
        if 'v_activation' in config.keys():
            self.v_activation = class_from_str(self.config.v_activation, sys.modules[__name__])
        else:
            self.v_activation = None
        if 'a_activation' in config.keys():
            self.a_activation = class_from_str(self.config.a_activation, sys.modules[__name__])
        else:
            self.a_activation = None
        if 'exp_activation' in config.keys():
            self.exp_activation = class_from_str(self.config.exp_activation, sys.modules[__name__])
        else:
            self.exp_activation = F.log_softmax
        self.va_loss = loss_from_cfg(config, 'va_loss')
        self.v_loss = loss_from_cfg(config, 'v_loss')
        self.a_loss = loss_from_cfg(config, 'a_loss')
        self.exp_loss = loss_from_cfg(config, 'exp_loss')
        self.config = add_cfg_if_missing(self.config, 'detach_shape', False)
        self.config = add_cfg_if_missing(self.config, 'detach_expression', False)
        self.config = add_cfg_if_missing(self.config, 'detach_detailcode', False)
        self.config = add_cfg_if_missing(self.config, 'detach_jaw', False)
        self.config = add_cfg_if_missing(self.config, 'detach_global_pose', False)

    def forward(self, values, result_prefix=''):
        shapecode = values['shapecode']
        if self.config.detach_shape:
            shapecode = shapecode.detach()
        expcode = values['expcode']
        if self.config.detach_expression:
            expcode = expcode.detach()
        posecode = values['posecode']
        if self.config.use_detail_code:
            if 'detailcode' in values.keys() and values['detailcode'] is not None:
                detailcode = values['detailcode']
                if self.config.detach_detailcode:
                    detailcode = detailcode.detach()
            else:
                detailcode = torch.zeros((posecode.shape[0], self.n_detail), dtype=posecode.dtype, device=posecode.device)
        else:
            detailcode = None
        if 'use_detailemo_code' in self.config.keys() and self.config.use_detailemo_code:
            if 'detailemocode' in values.keys() and values['detailemocode'] is not None:
                detailemocode = values['detailemocode']
                if 'detach_detailemocode' in self.config.keys() and self.config.detach_detailemocode:
                    detailemocode = detailemocode.detach()
            else:
                detailemocode = torch.zeros((posecode.shape[0], self.n_detail_emo), dtype=posecode.dtype, device=posecode.device)
        else:
            detailemocode = None
        global_pose = posecode[:, :3]
        if self.config.detach_global_pose:
            global_pose = global_pose.detach()
        jaw_pose = posecode[:, 3:]
        if self.config.detach_jaw:
            jaw_pose = jaw_pose.detach()
        input_list = []
        if self.config.use_identity:
            input_list += [shapecode]
        if self.config.use_expression:
            input_list += [expcode]
        if self.config.use_global_pose:
            input_list += [global_pose]
        if self.config.use_jaw_pose:
            input_list += [jaw_pose]
        if self.config.use_detail_code:
            input_list += [detailcode]
        if 'use_detail_emo_code' in self.config.keys() and self.config.use_detail_emo_code:
            input_list += [detailemocode]
        input = torch.cat(input_list, dim=1)
        output = self.mlp(input)
        out_idx = 0
        if self.config.predict_expression:
            expr_classification = output[:, out_idx:out_idx + self.num_classes]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(output[:, out_idx:out_idx + self.num_classes], dim=1)
            out_idx += self.num_classes
        else:
            expr_classification = None
        if self.config.predict_valence:
            valence = output[:, out_idx:out_idx + 1]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None
        if self.config.predict_arousal:
            arousal = output[:, out_idx:out_idx + 1]
            if self.a_activation is not None:
                arousal = self.a_activation(output[:, out_idx:out_idx + 1])
            out_idx += 1
        else:
            arousal = None
        values[result_prefix + 'valence'] = valence
        values[result_prefix + 'arousal'] = arousal
        values[result_prefix + 'expr_classification'] = expr_classification
        return values

    def compute_loss(self, pred, batch, training, pred_prefix=''):
        valence_gt = pred['va'][:, 0:1]
        arousal_gt = pred['va'][:, 1:2]
        expr_classification_gt = pred['affectnetexp']
        if 'expression_weight' in pred.keys():
            class_weight = pred['expression_weight'][0]
        else:
            class_weight = None
        gt = {}
        gt['valence'] = valence_gt
        gt['arousal'] = arousal_gt
        gt['expr_classification'] = expr_classification_gt
        scheme = None if 'va_loss_scheme' not in self.config.keys() else self.config.va_loss_scheme
        loss_term_weights = _get_step_loss_weights(self.v_loss, self.a_loss, self.va_loss, scheme, training)
        valence_sample_weight = batch['valence_sample_weight'] if 'valence_sample_weight' in batch.keys() else None
        arousal_sample_weight = batch['arousal_sample_weight'] if 'arousal_sample_weight' in batch.keys() else None
        va_sample_weight = batch['va_sample_weight'] if 'va_sample_weight' in batch.keys() else None
        expression_sample_weight = batch['expression_sample_weight'] if 'expression_sample_weight' in batch.keys() else None
        if 'continuous_va_balancing' in self.config.keys():
            if self.config.continuous_va_balancing == '1d':
                v_weight = valence_sample_weight
                a_weight = arousal_sample_weight
            elif self.config.continuous_va_balancing == '2d':
                v_weight = va_sample_weight
                a_weight = va_sample_weight
            elif self.config.continuous_va_balancing == 'expr':
                v_weight = expression_sample_weight
                a_weight = expression_sample_weight
            else:
                raise RuntimeError(f"Invalid continuous affect balancing '{self.config.continuous_va_balancing}'")
            if len(v_weight.shape) > 1:
                v_weight = v_weight.view(-1)
            if len(a_weight.shape) > 1:
                a_weight = a_weight.view(-1)
        else:
            v_weight = None
            a_weight = None
        losses, metrics = {}, {}
        losses, metrics = v_or_a_loss(self.v_loss, pred, gt, loss_term_weights, metrics, losses, 'valence', pred_prefix=pred_prefix, permit_dropping_corr=not training, sample_weights=v_weight)
        losses, metrics = v_or_a_loss(self.a_loss, pred, gt, loss_term_weights, metrics, losses, 'arousal', pred_prefix=pred_prefix, permit_dropping_corr=not training, sample_weights=a_weight)
        losses, metrics = va_loss(self.va_loss, pred, gt, loss_term_weights, metrics, losses, pred_prefix=pred_prefix, permit_dropping_corr=not training)
        losses, metrics = exp_loss(self.exp_loss, pred, gt, class_weight, metrics, losses, self.config.expression_balancing, self.num_classes, pred_prefix=pred_prefix)
        return losses, metrics


class StandardRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height
        self.w = w = width

    def forward(self, vertices, faces, attributes=None):
        device = vertices.device
        h = self.h
        w = self.h
        bz = vertices.shape[0]
        None
        None
        None
        depth_buffer = torch.zeros([bz, h, w]).float() + 1000000.0
        triangle_buffer = torch.zeros([bz, h, w]).int() - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float()
        vert_vis = torch.zeros([bz, vertices.shape[1]]).float()
        vertices = vertices.clone().float()
        vertices[..., 0] = vertices[..., 0] * w / 2 + w / 2
        vertices[..., 1] = vertices[..., 1] * h / 2 + h / 2
        vertices[..., 2] = vertices[..., 2] * w / 2
        f_vs = util.face_vertices(vertices, faces)
        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        None
        pix_to_face = triangle_buffer[:, :, :, None].long()
        bary_coords = baryw_buffer[:, :, :, None, :]
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = F.normalize(x)
        return x


def _parse_param_batch(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """
    n = param.shape[1]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')
    bs = param.shape[0]
    R_ = param[:, :trans_dim].reshape(bs, 3, -1)
    R = R_[:, :, :3]
    offset = R_[:, :, -1].reshape(bs, 3, 1)
    alpha_shp = param[:, trans_dim:trans_dim + shape_dim].reshape(bs, -1)
    alpha_exp = param[:, trans_dim + shape_dim:].reshape(bs, -1)
    return R, offset, alpha_shp, alpha_exp


def load_mesh(filename):
    fname, ext = os.path.splitext(filename)
    if ext == '.ply':
        vertices, faces = load_ply(filename)
    elif ext == '.obj':
        vertices, face_data, _ = load_obj(filename)
        faces = face_data[0]
    else:
        raise ValueError("Unknown extension '%s'" % ext)
    return vertices, faces


def render(mesh, device, renderer='flat') ->torch.Tensor:
    if isinstance(mesh, str):
        verts, faces = load_mesh(mesh)
    elif isinstance(mesh, list) or isinstance(mesh, tuple):
        verts = mesh[0]
        faces = mesh[1]
    else:
        raise ValueError("Unexpected mesh input of type '%s'. Pass in either a path to a mesh or its vertices and faces in a list or tuple" % str(type(mesh)))
    verts_rgb = torch.ones_like(verts)[None]
    verts_rgb[:, :, 0] = 135 / 255
    verts_rgb[:, :, 1] = 206 / 255
    verts_rgb[:, :, 2] = 250 / 255
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes([verts], [faces], textures)
    mesh = mesh
    batch_size = 5
    azim = torch.linspace(-90, 90, batch_size)
    R, T = look_at_view_transform(0.35, elev=0, azim=azim, at=((0, -0.025, 0),))
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=((0.0, 1, 1),), ambient_color=((0.5, 0.5, 0.5),), diffuse_color=((0.7, 0.7, 0.7),), specular_color=((0.8, 0.8, 0.8),))
    materials = Materials(device=device, specular_color=[[1.0, 1.0, 1.0]], shininess=65)
    if renderer == 'smooth':
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=HardPhongShader(device=device, lights=lights))
    elif renderer == 'flat':
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=HardFlatShader(device=device, lights=lights))
    else:
        raise ValueError("Invalid renderer specification '%s'" % renderer)
    meshes = mesh.extend(batch_size)
    images = renderer(meshes, materials=materials)
    return images


class Face3DDFAv2Wrapper(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        cfg_ = OmegaConf.to_container(cfg)
        if not Path(cfg_['bfm_fp']).is_file():
            cfg_['bfm_fp'] = str(Path(tddfa_v2_dir) / cfg_['bfm_fp'])
            assert Path(cfg_['bfm_fp']).is_file()
        if not Path(cfg_['param_mean_std_fp']).is_file():
            cfg_['param_mean_std_fp'] = str(Path(tddfa_v2_dir) / cfg_['param_mean_std_fp'])
            assert Path(cfg_['param_mean_std_fp']).is_file()
        if not Path(cfg_['checkpoint_fp']).is_file():
            cfg_['checkpoint_fp'] = str(Path(tddfa_v2_dir) / cfg_['checkpoint_fp'])
            assert Path(cfg_['checkpoint_fp']).is_file()
        self.tddfa = TDDFA(**cfg_)
        self.crop_images = False
        self.param_std = torch.from_numpy(self.tddfa.param_std)
        self.param_mean = torch.from_numpy(self.tddfa.param_mean)
        self.face_boxes = FaceBoxes()

    def forward(self, batch):
        values = self.encode(batch)
        values = self.decode(values)
        return values

    def encode(self, batch):
        img = batch['image']
        if not self.crop_images:
            resized_img = F.interpolate(img, (self.tddfa.size, self.tddfa.size), mode='bilinear') * 255.0
            resized_img = resized_img[:, [2, 1, 0], ...]
            transformed = self.tddfa.transform(resized_img)
            param = self.tddfa.model(transformed)
            param = param * self.param_std + self.param_mean
            R, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
            values = {'image': transformed, 'posecode': R, 'offset': offset, 'shapecode': alpha_shp, 'expcode': alpha_exp, 'bboxes': torch.tensor([0, 0, batch['image'].shape[2], batch['image'].shape[3]]).unsqueeze(0).repeat((img.shape[0], 1)), 'params': param}
        else:
            bboxes = []
            params = []
            shapecodes = []
            expcodes = []
            posecodes = []
            offsets = []
            imgs = (img.detach().cpu().numpy() * 255.0).transpose([0, 2, 3, 1]).astype(np.uint8)
            for i in range(imgs.shape[0]):
                img = imgs[i][:, :, ::-1]
                boxes = self.face_boxes(img)
                n = len(boxes)
                if n == 0:
                    None
                    boxes = [[0, 0, batch['image'].shape[2], batch['image'].shape[3]]]
                param_lst, roi_box_lst = self.tddfa(img, [boxes[0]])
                bboxes += [roi_box_lst[0]]
                params += [param_lst[0]]
                R, offset, alpha_shp, alpha_exp = _parse_param(param_lst[0])
                shapecodes += [alpha_shp]
                expcodes += [alpha_exp]
                posecodes += [R]
                offsets += [offset]
            values = {'bboxes': bboxes, 'params': params, 'shapecode': torch.from_numpy(np.hstack(shapecodes).T), 'expcode': torch.from_numpy(np.hstack(expcodes).T), 'posecode': torch.from_numpy(np.hstack(posecodes).T), 'offset': torch.from_numpy(np.hstack(offsets).T), 'image': img}
        return values

    def decode(self, values):
        image = values['image']
        param_lst = values['params']
        roi_box_lst = values['bboxes']
        wfp = None
        dense_flag = True
        if not self.crop_images:
            ver_lst = self.tddfa.recon_vers(param_lst.detach().cpu().numpy(), roi_box_lst.detach().cpu().numpy(), dense_flag=dense_flag)
        else:
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        imgs = image.detach().cpu().numpy()
        geometry_imgs = []
        landmark_imgs = []
        overlays_imgs = []
        for i, img in enumerate(imgs):
            if self.crop_images:
                img = (img.transpose([1, 2, 0])[:, :, [2, 1, 0]] * 255.0).astype(np.uint8)
            else:
                img = (img.transpose([1, 2, 0]) * 255.0).astype(np.uint8)
            show_flag = False
            dense_flag = False
            img_geometry = render(img, [ver_lst[i].copy()], self.tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
            img_overlay = render(img, [ver_lst[i].copy()], self.tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=wfp, with_bg_flag=True)
            geometry_imgs += [img_geometry]
            overlays_imgs += [img_overlay]
        values['geometry_coarse'] = geometry_imgs
        values['overlays_img'] = overlays_imgs
        return values


class LipReadingNet(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        cfg_path = get_path_to_externals() / 'spectre' / 'configs' / 'lipread_config.ini'
        config = ConfigParser()
        config.read(cfg_path)
        model_path = str(get_path_to_externals() / 'spectre' / config.get('model', 'model_path'))
        model_conf = str(get_path_to_externals() / 'spectre' / config.get('model', 'model_conf'))
        config.set('model', 'model_path', model_path)
        config.set('model', 'model_conf', model_conf)
        self.lip_reader = Lipreading(config, device=device)
        crop_size = 88, 88
        mean, std = 0.421, 0.165
        self.mouth_transform = Compose([Normalize(0.0, 1.0), CenterCrop(crop_size), Normalize(mean, std), Identity()])

    def forward(self, lip_images):
        """
        :param lip_images: (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88))
        """
        ndim = lip_images.ndim
        B, T = lip_images.shape[:2]
        rest = lip_images.shape[2:]
        if ndim == 5:
            lip_images = lip_images.view(B * T, *rest)
        elif ndim == 4:
            pass
        else:
            raise ValueError('Lip images should be of shape (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88)')
        channel_dim = 1
        lip_images = self.mouth_transform(lip_images.squeeze(channel_dim)).unsqueeze(channel_dim)
        if ndim == 5:
            lip_images = lip_images.view(B, T, *lip_images.shape[2:])
        elif ndim == 4:
            lip_images = lip_images.unsqueeze(0)
            lip_images = lip_images.squeeze(2)
        lip_features = self.lip_reader.model.encoder(lip_images, None, extract_resnet_feats=True)
        return lip_features

    def forward_old(self, lip_images):
        channel_dim = 1
        lip_images = self.mouth_transform(lip_images.squeeze(channel_dim)).unsqueeze(channel_dim)
        lip_images = lip_images.view(-1, lip_images.shape[1], lip_images.shape[-2], lip_images.shape[-1])
        lip_features = self.lip_reader.model.encoder(lip_images, None, extract_resnet_feats=True)
        return lip_features


class LipReadingLoss(torch.nn.Module):

    def __init__(self, device, loss='cosine_similarity'):
        super().__init__()
        self.loss = loss
        assert loss in ['cosine_similarity', 'l1_loss', 'mse_loss']
        self.model = LipReadingNet(device)
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _forward_input(self, images):
        with torch.no_grad():
            result = self.model(images)
        return result

    def _forward_output(self, images):
        return self.model(images)

    def compute_loss(self, mouth_images_gt, mouth_images_pred, mask=None):
        lip_features_gt = self._forward_input(mouth_images_gt)
        lip_features_pred = self._forward_output(mouth_images_pred)
        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        if mask is not None:
            lip_features_gt = lip_features_gt[mask.view(-1)]
            lip_features_pred = lip_features_pred[mask.view(-1)]
        return self._compute_feature_loss(lip_features_gt, lip_features_pred)

    def _compute_feature_loss(self, lip_features_gt, lip_features_pred):
        if self.loss == 'cosine_similarity':
            lr = 1 - torch.nn.functional.cosine_similarity(lip_features_gt, lip_features_pred, dim=1).mean()
        elif self.loss == 'l1_loss':
            lr = torch.nn.functional.l1_loss(lip_features_gt, lip_features_pred)
        elif self.loss == 'mse_loss':
            lr = torch.nn.functional.mse_loss(lip_features_gt, lip_features_pred)
        else:
            raise ValueError(f'Unknown loss function: {self.loss}')
        return lr


class KeypointTransform(torch.nn.Module):

    def __init__(self, scale_x=1.0, scale_y=1.0):
        super().__init__()
        self.scale_x = scale_x
        self.scale_y = scale_y

    def set_scale(self, scale_x=1.0, scale_y=1.0):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, points):
        raise NotImplementedError()


class KeypointScale(KeypointTransform):

    def __init__(self, scale_x=1.0, scale_y=1.0):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        points_ = points.clone()
        points_[..., 0] *= self.scale_x
        points_[..., 1] *= self.scale_y
        return points_


class KeypointNormalization(KeypointTransform):

    def __init__(self, scale_x=1.0, scale_y=1.0):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f'Invalid type of points {str(type(points))}')
        points_[..., 0] -= self.scale_x / 2
        points_[..., 0] /= self.scale_x / 2
        points_[..., 1] -= self.scale_y / 2
        points_[..., 1] /= self.scale_y / 2
        return points_

    def inv(self, points):
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f'Invalid type of points {str(type(points))}')
        points_[..., 0] *= self.scale_x / 2
        points_[..., 0] += self.scale_x / 2
        points_[..., 1] *= self.scale_y / 2
        points_[..., 1] += self.scale_y / 2
        return points_


def load_segmentation(filename):
    with open(filename, 'rb') as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
    return seg_image, seg_type


face_parsing_labels = {(0): 'background', (1): 'skin', (2): 'nose', (3): 'eye_g', (4): 'l_eye', (5): 'r_eye', (6): 'l_brow', (7): 'r_brow', (8): 'l_ear', (9): 'r_ear', (10): 'mouth', (11): 'u_lip', (12): 'l_lip', (13): 'hair', (14): 'hat', (15): 'ear_r', (16): 'neck_l', (17): 'neck', (18): 'cloth'}


face_parsin_inv_labels = {v: k for k, v in face_parsing_labels.items()}


default_discarded_labels = [face_parsin_inv_labels['background'], face_parsin_inv_labels['l_ear'], face_parsin_inv_labels['r_ear'], face_parsin_inv_labels['hair'], face_parsin_inv_labels['hat'], face_parsin_inv_labels['neck'], face_parsin_inv_labels['neck_l']]


def process_segmentation(segmentation, seg_type, discarded_labels=None):
    if seg_type == 'face_parsing':
        discarded_labels = discarded_labels or default_discarded_labels
        segmentation_proc = np.isin(segmentation, discarded_labels)
        segmentation_proc = np.logical_not(segmentation_proc)
        segmentation_proc = segmentation_proc.astype(np.float32)
        return segmentation_proc
    else:
        raise ValueError(f"Invalid segmentation type '{seg_type}'")


def load_and_process_segmentation(path):
    seg_image, seg_type = load_segmentation(path)
    seg_image = seg_image[np.newaxis, :, :, np.newaxis]
    seg_image = process_segmentation(seg_image, seg_type).astype(np.uint8)
    return seg_image


def load_landmark(fname):
    with open(fname, 'rb') as f:
        landmark_type = pkl.load(f)
        landmark = pkl.load(f)
    return landmark_type, landmark


def numpy_image_to_torch(img: 'np.ndarray') ->torch.Tensor:
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img)


def load_image_to_batch(image):
    batch = {}
    if isinstance(image, str) or isinstance(image, Path):
        image_path = image
        image = imread(image_path)[:, :, :3]
        if 'detections' in str(image_path):
            lmk_path = str(image_path).replace('detections', 'landmarks')
            lmk_path = str(lmk_path).replace('.png', '.pkl')
            if Path(lmk_path).is_file():
                landmark_type, landmark = load_landmark(lmk_path)
                landmark = landmark[np.newaxis, ...]
                landmark /= image.shape[0]
                landmark -= 0.5
                landmark *= 2
            else:
                landmark = None
            seg_path = str(image_path).replace('detections', 'segmentations')
            seg_path = str(seg_path).replace('.png', '.pkl')
            if Path(seg_path).is_file():
                seg_im = load_and_process_segmentation(seg_path)[0, ...]
            else:
                seg_im = None
        else:
            landmark = None
            seg_im = None
    if isinstance(image, np.ndarray):
        image = np.transpose(image, [2, 0, 1])[None, ...]
        if image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            image = image.astype(np.float32)
            image /= 255.0
    image = torch.from_numpy(image)
    batch['image'] = image
    if landmark is not None:
        batch['landmark'] = torch.from_numpy(landmark)
    if seg_im is not None:
        batch['mask'] = numpy_image_to_torch(seg_im)[None, ...]
    return batch


class TargetEmotionCriterion(torch.nn.Module):

    def __init__(self, target_image, use_feat_1=False, use_feat_2=True, use_valence=False, use_arousal=False, use_expression=False, emonet_loss_instance=None):
        super().__init__()
        if emonet_loss_instance is None:
            None
        self.emonet_loss = emonet_loss_instance or EmoNetLoss('cuda')
        self.emonet_loss.eval()
        target_image = load_image_to_batch(target_image)['image']
        self.register_buffer('target_image', target_image)
        self.target_emotion = self.emonet_loss(target_image)
        self.use_feat_1 = use_feat_1
        self.use_feat_2 = use_feat_2
        self.use_valence = use_valence
        self.use_arousal = use_arousal
        self.use_expression = use_expression

    def __call__(self, image):
        return self.forward(image)

    def forward(self, image):
        return self.compute(image)

    def compute(self, image):
        input_emotion = self.emonet_loss(image)
        emo_feat_loss_2 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat_2'], self.target_emotion['emo_feat_2'])
        valence_loss = self.emonet_loss.valence_loss(input_emotion['valence'], self.target_emotion['valence'])
        arousal_loss = self.emonet_loss.arousal_loss(input_emotion['arousal'], self.target_emotion['arousal'])
        if 'expression' in input_emotion.keys():
            expression_loss = self.emonet_loss.expression_loss(input_emotion['expression'], self.target_emotion['expression'])
        else:
            expression_loss = self.emonet_loss.expression_loss(input_emotion['expr_classification'], self.target_emotion['expr_classification'])
        total_loss = torch.zeros_like(emo_feat_loss_2)
        if self.use_feat_1:
            emo_feat_loss_1 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat'], self.target_emotion['emo_feat'])
            total_loss = total_loss + emo_feat_loss_1
        if self.use_feat_2:
            total_loss = total_loss + emo_feat_loss_2
        if self.use_valence:
            total_loss = total_loss + valence_loss
        if self.use_arousal:
            total_loss = total_loss + arousal_loss
        if self.use_expression:
            total_loss = total_loss + expression_loss
        return total_loss

    @property
    def name(self):
        return 'EmotionLoss'

    def get_target_image(self):
        im = np.transpose(self.target_image.detach().cpu().numpy()[0, ...], [1, 2, 0])
        return im

    def save_target_image(self, path):
        im = self.get_target_image()
        None
        imsave(path, im)


class DecaTermCriterion(torch.nn.Module):

    def __init__(self, keyword):
        super().__init__()
        self.keyword = keyword

    def forward(self, loss_dict):
        return loss_dict[self.keyword]


def convert_rotation(input, rot_type):
    if rot_type == 'aa':
        pass
    elif rot_type == 'quat':
        jaw_pose = trans.axis_angle_to_quaternion(input)
    elif rot_type == 'euler':
        jaw_pose = trans.matrix_to_euler_angles(trans.axis_angle_to_matrix(input), 'XYZ')
    else:
        raise ValueError(f"Invalid rotaion reference type: '{rot_type}'")
    return jaw_pose


class TargetJawCriterion(torch.nn.Module):

    def __init__(self, reference_pose, reference_type, loss_type='l1'):
        super().__init__()
        self.reference_pose = torch.tensor(reference_pose)
        self.reference_type = reference_type
        self.loss_type = loss_type

    def __call__(self, posecode):
        return self.forward(posecode)

    def forward(self, posecode):
        return self.compute(posecode)

    @property
    def name(self):
        return f'JawReg_{self.reference_type}_{self.loss_type}'

    def compute(self, posecode):
        jaw_pose = posecode[:, 3:]
        jaw_pose = convert_rotation(jaw_pose, self.reference_type)
        if self.loss_type == 'l1':
            reg = torch.abs(jaw_pose - self.reference_pose).sum()
        elif self.loss_type == 'l2':
            reg = torch.square(jaw_pose - self.reference_pose).sum()
        else:
            raise NotImplementedError(f"Invalid loss: '{self.loss_type}'")
        return reg

    def save_target_image(self, path):
        pass


class CriterionWrapper(torch.nn.Module):

    def __init__(self, criterion, key):
        super().__init__()
        self.criterion = criterion
        self.key = key

    def forward(self, d):
        return self.criterion(d[self.key])

    @property
    def name(self):
        return self.criterion.name

