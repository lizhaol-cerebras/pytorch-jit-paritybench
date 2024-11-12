
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


import scipy.io


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import numpy as np


from torch.autograd import Variable


from copy import deepcopy


import torch.nn as nn


from torchvision import transforms


from typing import Tuple


from torch import nn


from torch import Tensor


from torchvision import datasets


import matplotlib


import matplotlib.pyplot as plt


import time


import torch.nn.functional as F


from torch.nn import init


from torchvision import models


import torch.optim as optim


from torch.optim import lr_scheduler


import torch.backends.cudnn as cudnn


import torchvision


import math


import collections


from torch.nn.utils import fuse_conv_bn_eval


class CircleLoss(nn.Module):

    def __init__(self, m: 'float', gamma: 'float') ->None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: 'Tensor', sn: 'Tensor') ->Tensor:
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss


def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-06
    v = v.div(fnorm.expand_as(v))
    return v


class InstanceLoss(nn.Module):

    def __init__(self, gamma=1) ->None:
        super(InstanceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, feature, label=None) ->Tensor:
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature * self.gamma, torch.t(normed_feature))
        if label is None:
            sim_label = torch.arange(sim1.size(0)).detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label)
        return loss


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):

    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.linear_num = linear
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = 1, 1
            model_ft.layer4[0].conv2.stride = 1, 1
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate=0.2)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        None

    def forward(self, x):
        x = self.model.forward_features(x)
        if x.dim() == 3:
            x = self.avgpool1d(x.permute((0, 2, 1)))
        else:
            x = self.avgpool2d(x.permute((0, 3, 1, 2)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


def load_state_dict_mute(self, state_dict: "'OrderedDict[str, Tensor]'", strict: 'bool'=True):
    """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
    missing_keys: 'List[str]' = []
    unexpected_keys: 'List[str]' = []
    error_msgs: 'List[str]' = []
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
    load(self)
    del load
    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))


class ft_net_swinv2(nn.Module):

    def __init__(self, class_num, input_size=(256, 128), droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swinv2, self).__init__()
        model_ft = timm.create_model('swinv2_base_window8_256', pretrained=False, img_size=input_size, drop_path_rate=0.2)
        model_full = timm.create_model('swinv2_base_window8_256', pretrained=True)
        load_state_dict_mute(model_ft, model_full.state_dict(), strict=False)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)
        None

    def forward(self, x):
        x = self.model.forward_features(x)
        if x.dim() == 3:
            x = self.avgpool1d(x.permute((0, 2, 1)))
        else:
            x = self.avgpool2d(x.permute((0, 3, 1, 2)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_convnext(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_convnext, self).__init__()
        model_ft = timm.create_model('convnext_base', pretrained=True, drop_path_rate=0.2)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_hr(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, linear_num=512):
        super().__init__()
        model_name = 'nasnetalarge'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(4032, class_num, droprate, linear=linear_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.part = 6
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):

    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        epsilon (float): weight.
    """

    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        _, num_classes = inputs.shape
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CircleLoss,
     lambda: ([], {'m': 4, 'gamma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ClassBlock,
     lambda: ([], {'input_dim': 4, 'class_num': 4, 'droprate': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (CrossEntropyLabelSmooth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (InstanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (PCB,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ft_net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ft_net_dense,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ft_net_middle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

