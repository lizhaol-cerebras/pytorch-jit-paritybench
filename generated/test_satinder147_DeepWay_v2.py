
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


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import math


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import torch.utils.model_zoo as model_zoo


import torch.nn.init as init


import logging


from torch.nn import Conv2d


from torch.nn import Sequential


from torch.nn import ModuleList


from torch.nn import ReLU


from typing import List


from typing import Tuple


from torch.nn import BatchNorm2d


from torch import nn


from collections import namedtuple


from torchvision import transforms


import types


from numpy import random


import collections


import itertools


import time


from collections import OrderedDict


class Cnn(nn.Module):

    def __init__(self):
        super(Cnn, self).__init__()
        None
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(2048, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Unet(nn.Module):

    def __init__(self, inp, out):
        super(Unet, self).__init__()
        self.c1 = self.contracting_block(inp, 8)
        self.c2 = self.contracting_block(8, 16)
        self.c3 = self.contracting_block(16, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c4 = self.contracting_block(16 + 32, 16)
        self.c5 = self.contracting_block(8 + 16, 8)
        self.c6 = nn.Conv2d(8, out, 1)

    def contracting_block(self, inp, out, k=3):
        block = nn.Sequential(nn.Conv2d(inp, out, k, padding=1), nn.Dropout(p=0.5, inplace=True), nn.ReLU(inplace=True), nn.BatchNorm2d(out), nn.Conv2d(out, out, k, padding=1), nn.Dropout(p=0.5, inplace=True), nn.ReLU(inplace=True), nn.BatchNorm2d(out), nn.Conv2d(out, out, k, padding=1), nn.Dropout(p=0.5, inplace=True), nn.ReLU(inplace=True), nn.BatchNorm2d(out))
        return block

    def forward(self, x):
        conv1 = self.c1(x)
        x = self.maxpool(conv1)
        conv2 = self.c2(x)
        x = self.maxpool(conv2)
        conv3 = self.c3(x)
        x = self.upsample(conv3)
        x = torch.cat([conv2, x], axis=1)
        x = self.c4(x)
        x = self.upsample(x)
        x = torch.cat([conv1, x], axis=1)
        x = self.c5(x)
        x = self.c6(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
            else:
                self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        elif use_batch_norm:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), ReLU(inplace=True))


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), ReLU(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0, dropout_ratio=0.2, use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(dropout_ratio), nn.Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MultiboxLoss(nn.Module):

    def __init__(self, priors, iou_threshold, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class ScaledL2Norm(nn.Module):

    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1) * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13, stride=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def _xavier_init_(m: 'nn.Module'):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class FPNSSD(nn.Module):

    def __init__(self, num_classes: 'int', base_net: 'nn.ModuleList', source_layer_indexes: 'List[int]', extras: 'nn.ModuleList', classification_headers: 'nn.ModuleList', regression_headers: 'nn.ModuleList', upsample_mode='nearest'):
        """Compose a SSD model using the given components.
        """
        super(FPNSSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.upsample_mode = upsample_mode
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes if isinstance(t, tuple)])
        self.upsamplers = [nn.Upsample(size=(19, 19), mode='bilinear'), nn.Upsample(size=(10, 10), mode='bilinear'), nn.Upsample(size=(5, 5), mode='bilinear'), nn.Upsample(size=(3, 3), mode='bilinear'), nn.Upsample(size=(2, 2), mode='bilinear')]

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        features = []
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
            else:
                added_layer = None
            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            start_layer_index = end_layer_index
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            features.append(y)
            header_index += 1
        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            features.append(x)
            header_index += 1
        upstream_feature = None
        for i in range(len(features) - 1, -1, -1):
            feature = features[i]
            if upstream_feature is not None:
                upstream_feature = self.upsamplers[i](upstream_feature)
                upstream_feature += feature
            else:
                upstream_feature = feature
            confidence, location = self.compute_header(i, upstream_feature)
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=False)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


GraphPath = namedtuple('GraphPath', ['s0', 'name', 's1'])


class SSD(nn.Module):

    def __init__(self, num_classes: 'int', base_net: 'nn.ModuleList', source_layer_indexes: 'List[int]', extras: 'nn.ModuleList', classification_headers: 'nn.ModuleList', regression_headers: 'nn.ModuleList', is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if is_test:
            self.config = config
            self.priors = config.priors

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(locations, self.priors, self.config.center_variance, self.config.size_variance)
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('classification_headers') or k.startswith('regression_headers'))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Cnn,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ScaledL2Norm,
     lambda: ([], {'in_channels': 4, 'initial_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Unet,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

