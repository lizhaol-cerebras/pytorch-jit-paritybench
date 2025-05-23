
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


import torchvision


from torchvision import transforms


import torch.utils.data as data


import random


import torch.nn.functional as F


import numpy as np


import torchvision.transforms.functional as F


import math


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()
        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle
        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


class Model(nn.Module):

    def __init__(self, part_num, class_num):
        super(Model, self).__init__()
        self.part_num = part_num
        self.class_num = class_num
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = 1, 1
        resnet.layer4[0].conv2.stride = 1, 1
        resnet.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        dropout = nn.Dropout(p=0.5)
        resnet.avgpool_e = nn.AdaptiveAvgPool2d((part_num, 1))
        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.pool_c = nn.Sequential(resnet.avgpool_c, dropout)
        self.pool_e = nn.Sequential(resnet.avgpool_e)
        for i in range(part_num):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(2048, self.class_num, relu=True, dropout=False, bottle_dim=256))
        for i in range(part_num):
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(2048, 256))

    def forward(self, x):
        features = self.resnet_conv(x)
        features_c = torch.squeeze(self.pool_c(features))
        features_e = torch.squeeze(self.pool_e(features))
        logits_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_c
            else:
                features_i = torch.squeeze(features_c[:, :, i])
            classifier_i = getattr(self, 'classifier' + str(i))
            logits_i = classifier_i(features_i)
            logits_list.append(logits_i)
        embeddings_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_e
            else:
                features_i = torch.squeeze(features_e[:, :, i])
            embedder_i = getattr(self, 'embedder' + str(i))
            embedding_i = embedder_i(features_i)
            embeddings_list.append(embedding_i)
        if self.training:
            return logits_list, embeddings_list
        else:
            return F.normalize(features_c, 2, dim=2).reshape([-1, 2048 * self.part_num])


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BottleClassifier,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

