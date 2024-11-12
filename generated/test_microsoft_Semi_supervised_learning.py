
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


import numpy as np


import torch


from torch.utils.data import DataLoader


import pandas as pd


import re


import time


import random


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from collections import Counter


import math


from torch.autograd import Variable


from inspect import signature


from collections import OrderedDict


from sklearn.metrics import accuracy_score


from sklearn.metrics import balanced_accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


from sklearn.metrics import confusion_matrix


from sklearn.metrics import top_k_accuracy_score


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.nn import functional as F


import logging


import torch.distributed as dist


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset


import torchvision.transforms.functional as F


import warnings


import copy


import torchvision


from torchvision import transforms


import torchvision.transforms as transforms


from torch.utils.data.sampler import Sampler


from torch.utils.data import sampler


import torch.utils.data as data


import torch.optim as optim


from torch.utils.data.sampler import WeightedRandomSampler


from scipy import optimize


import queue


from collections import defaultdict


from torch import Tensor


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


import collections


from itertools import islice


from itertools import chain


from typing import Tuple


from typing import Dict


from torch.hub import load_state_dict_from_url


from functools import partial


import torch.utils.checkpoint


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


import torch.nn.parallel


class CoMatch_Net(nn.Module):

    def __init__(self, base, proj_size=128, epass=False):
        super(CoMatch_Net, self).__init__()
        self.backbone = base
        self.epass = epass
        self.num_features = base.num_features
        self.mlp_proj = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])
        if self.epass:
            self.mlp_proj_2 = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])
            self.mlp_proj_3 = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1.0 / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat)) / 3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits': logits, 'feat': feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class CRMatch_Net(nn.Module):

    def __init__(self, base, args, use_rot=True):
        super(CRMatch_Net, self).__init__()
        self.backbone = base
        self.use_rot = use_rot
        self.num_features = base.num_features
        self.args = args
        if self.use_rot:
            self.rot_classifier = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, 4))
        if 'wrn' in args.net or 'resnet' in args.net:
            if args.dataset == 'stl10':
                feat_map_size = 6 * 6 * self.num_features
            elif args.dataset == 'imagenet':
                feat_map_size = 7 * 7 * self.num_features
            else:
                feat_map_size = 8 * 8 * self.num_features
        elif 'vit' in args.net or 'bert' in args.net or 'wave2vec' in args.net:
            feat_map_size = self.backbone.num_features
        else:
            raise NotImplementedError
        self.ds_classifier = nn.Linear(feat_map_size, self.num_features, bias=True)

    def forward(self, x):
        feat_maps = self.backbone.extract(x)
        if 'wrn' in self.args.net or 'resnet' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
            feat_maps = torch.mean(feat_maps, dim=(2, 3))
        elif 'vit' in self.args.net:
            if self.backbone.global_pool:
                feat_maps = feat_maps[:, 1:].mean(dim=1) if self.backbone.global_pool == 'avg' else feat_maps[:, 0]
            feat_maps = self.backbone.fc_norm(feat_maps)
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        elif 'bert' in self.args.net or 'wave2vec' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        else:
            raise NotImplementedError
        logits = self.backbone(feat_maps, only_fc=True)
        results_dict = {'logits': logits, 'logits_ds': logits_ds, 'feat': feat_maps}
        if self.use_rot:
            logits_rot = self.rot_classifier(feat_maps)
            results_dict['logits_rot'] = logits_rot
        else:
            results_dict['logits_rot'] = None
        return results_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class ReMixMatch_Net(nn.Module):

    def __init__(self, base, use_rot=True):
        super(ReMixMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        if use_rot:
            self.rot_classifier = nn.Linear(self.num_features, 4)

    def forward(self, x, use_rot=False, **kwargs):
        if not use_rot:
            return self.backbone(x, **kwargs)
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_rot = self.rot_classifier(feat)
        return {'logits': logits, 'logits_rot': logits_rot, 'feat': feat}

    def init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class SimMatch_Net(nn.Module):

    def __init__(self, base, proj_size=128, epass=False):
        super(SimMatch_Net, self).__init__()
        self.backbone = base
        self.epass = epass
        self.num_features = base.num_features
        self.mlp_proj = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])
        if self.epass:
            self.mlp_proj_2 = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])
            self.mlp_proj_3 = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(inplace=False), nn.Linear(self.num_features, proj_size)])

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1.0 / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat)) / 3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits': logits, 'feat': feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """
    assert name in ['ce', 'mse', 'kl']
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')
    if mask is not None and name != 'kl':
        loss = loss * mask
    return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """

    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """

    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)


class ABCNet(nn.Module):

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class CoSSL_Net(nn.Module):

    def __init__(self, backbone, num_classes):
        super(CoSSL_Net, self).__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        if hasattr(backbone, 'backbone'):
            self.classifier = backbone.backbone.classifier
        else:
            self.classifier = backbone.classifier
        self.teacher_classifier = nn.Linear(self.num_features, num_classes, bias=True)

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        feat = results_dict['feat']
        logits = self.classifier(feat)
        tfe_logits = self.teacher_classifier(feat)
        results_dict['logits'] = logits
        results_dict['logits_tfe'] = tfe_logits
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class DebiasPLConsistencyLoss(ConsistencyLoss):

    def __init__(self, tau=0.4):
        super().__init__()
        self.tau = 0.4

    def set_param(self, p):
        self.p_hat = p

    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits + self.tau * torch.log(self.p_hat), targets, name, mask)


class SAWCELoss(CELoss):

    def __init__(self):
        super().__init__()

    def set_weights(self, weights):
        self.x_lb_weights = weights

    def forward(self, logits, targets, reduction='none'):
        loss = super().forward(logits, targets, reduction='none')
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        loss = loss * self.x_lb_weights[targets]
        return loss.mean()


class SAWConsistencyLoss(ConsistencyLoss):

    def __init__(self):
        super().__init__()

    def set_weights(self, weights):
        self.x_ulb_weights = weights

    def forward(self, logits, targets, name='ce', mask=None):
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        if mask is None:
            mask = self.x_ulb_weights[targets]
        else:
            mask = mask * self.x_ulb_weights[targets]
        return super().forward(logits, targets, name, mask)


class TRASNet(nn.Module):
    """
        Transfer & Share algorithm (https://arxiv.org/abs/2205.13358).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - tras_A
                A parameter in TRAS
            - tras_B
                B parameter  in TRAS
            - tras_tro:
                tro parameter in TRAS
            - tras_warmup_epochs:
                TRAS warmup epochs
    """

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class TRASLogitsAdjCELoss(CELoss):

    def __init__(self, la):
        super().__init__()
        self.la = la

    def forward(self, logits, targets, reduction='mean'):
        return super().forward(logits + self.la, targets, reduction=reduction)


class TRASKLLoss(nn.Module):

    def forward(self, outputs, targets, T, mask):
        _p = F.log_softmax(outputs / T, dim=1)
        _q = F.softmax(targets / (T * 2), dim=1)
        _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1) * mask)
        _soft_loss = _soft_loss * T * T
        return _soft_loss


class ClassificationBert(nn.Module):

    def __init__(self, name, num_classes=2):
        super(ClassificationBert, self).__init__()
        self.bert = BertModel.from_pretrained(name)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[nn.Linear(768, 768), nn.GELU(), nn.Linear(768, num_classes)])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        if only_feat:
            return pooled_output
        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
        return result_dict

    def extract(self, x):
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}bert.embeddings'.format(prefix), blocks='^{}bert.encoder.layer.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


class CustomDINONormModel(nn.Module):

    def __init__(self, name, num_classes=8):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = Dinov2Model.from_pretrained(name)
        self.classifier = nn.Sequential(*[nn.Linear(1024, 256), nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes)])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if return_embed:
            embed = self.dino_model(x)
            return embed
        out_dict = self.dino_model(x, output_hidden_states=True, return_dict=True)
        last_hidden_state = out_dict['last_hidden_state']
        pooled_output = torch.mean(last_hidden_state, 1)
        if only_fc:
            logits = self.classifier(pooled_output)
            return logits
        if only_feat:
            return pooled_output
        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}dino_model.embeddings'.format(prefix), blocks='^{}dino_model.encoder.layer.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


class ClassificationHubert(nn.Module):

    def __init__(self, name, num_classes=2):
        super(ClassificationHubert, self).__init__()
        self.model = HubertModel.from_pretrained(name)
        self.model.feature_extractor._requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[nn.Linear(768, 768), nn.GELU(), nn.Linear(768, num_classes)])

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        pooled_output = self.extract(x)
        if only_feat:
            return pooled_output
        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict

    def extract(self, x):
        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        embed = out_dict['hidden_states'][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}model.feature_projection|^{}model.feature_extractor|^{}model.encoder.pos_conv_embed'.format(prefix, prefix, prefix), blocks='^{}model.encoder.layers.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, groups: 'int'=1, dilation: 'int'=1) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion: 'int' = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(Bottleneck, self).__init__()
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

    def forward(self, x: 'Tensor') ->Tensor:
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


class ResNet50(nn.Module):

    def __init__(self, block: 'Type[Union[BasicBlock, Bottleneck]]'=Bottleneck, layers: 'List[int]'=[3, 4, 6, 3], num_classes: 'int'=1000, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(ResNet50, self).__init__()
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
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

    def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False) ->nn.Sequential:
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

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            return self.fc(x)
        x = self.extract(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if only_feat:
            return x
        out = self.classifier(x)
        result_dict = {'logits': out, 'feat': x}
        return result_dict

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}conv1|^{}bn1|^{}maxpool'.format(prefix, prefix, prefix), blocks='^{}layer(\\d+)'.format(prefix) if coarse else '^{}layer(\\d+)\\.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, init_values=None, embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.grad_checkpointing = False
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.num_features = self.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def extract(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            return self.head(x)
        x = self.extract(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        if only_feat:
            return x
        output = self.head(x)
        result_dict = {'logits': output, 'feat': x}
        return result_dict

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def group_matcher(self, coarse=False, prefix=''):
        return dict(stem='^{}cls_token|{}pos_embed|{}patch_embed'.format(prefix, prefix, prefix), blocks=[('^{}blocks\\.(\\d+)'.format(prefix), None), ('^{}norm'.format(prefix), (99999,))])


class ClassificationWave2Vec(nn.Module):

    def __init__(self, name, num_classes=2):
        super(ClassificationWave2Vec, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained(name)
        self.model.feature_extractor._requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[nn.Linear(768, 768), nn.GELU(), nn.Linear(768, num_classes)])

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        pooled_output = self.extract(x)
        if only_feat:
            return pooled_output
        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict

    def extract(self, x):
        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        embed = out_dict['hidden_states'][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}model.feature_projection|^{}model.feature_extractor'.format(prefix, prefix), blocks='^{}model.encoder.layers.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.classifier = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.num_features = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            return self.classifier(x)
        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if only_feat:
            return out
        output = self.classifier(out)
        result_dict = {'logits': output, 'feat': out}
        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}conv1'.format(prefix), blocks='^{}block(\\d+)'.format(prefix) if coarse else '^{}block(\\d+)\\.layer.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


class WideResNetVar(nn.Module):

    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNetVar, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
        self.block4 = NetworkBlock(n, channels[3], channels[4], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.classifier = nn.Linear(channels[4], num_classes)
        self.channels = channels[4]
        self.num_features = channels[4]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            return self.classifier(x)
        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if only_feat:
            return out
        output = self.classifier(out)
        result_dict = {'logits': output, 'feat': out}
        return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem='^{}conv1'.format(prefix), blocks='^{}block(\\d+)'.format(prefix) if coarse else '^{}block(\\d+)\\.layer.(\\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': torch.nn.ReLU, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PSBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet50,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

