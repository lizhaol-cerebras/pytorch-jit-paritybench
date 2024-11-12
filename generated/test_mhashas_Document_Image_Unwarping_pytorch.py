
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


import torch.nn.functional as F


import torch.nn as nn


from torch import nn


from torchvision import models


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


import copy


import math


import matplotlib.pyplot as plt


import time


from torch.utils import data


import torchvision.transforms as standard_transforms


from scipy import io


import random


import types


from torchvision.transforms import functional as F


import numpy as np


import torch.optim as optim


from torch.utils.data import ConcatDataset


from torch.utils.data import DataLoader


import functools


from math import exp


from torchvision.utils import make_grid


from torch.utils.tensorboard import SummaryWriter


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):

    def __init__(self, output_stride, norm_layer=nn.BatchNorm2d, inplanes=2048):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(inplanes, 256, 1, stride=1, bias=False), norm_layer(256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, inplanes=256, aspp_outplanes=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU()
        inplanes = 48 + aspp_outplanes
        self.last_conv = nn.Sequential(nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(256), nn.ReLU(), nn.Dropout2d(0.5), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(256), nn.ReLU(), nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x


DEEPLAB = 'deeplab'


DEEPLAB_18 = 'deeplab_18'


DEEPLAB_34 = 'deeplab_34'


DEEPLAB_50 = 'deeplab_50'


DEEPLAB_MOBILENET = 'deeplab_mn'


DEEPLAB_MOBILENET_DILATION = 'deeplab_mnd'


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, first_layer_input_channels=3, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None):
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(first_layer_input_channels, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.low_level_features = nn.Sequential(*features[0:4])
        self.high_level_features = nn.Sequential(*features[4:-1])

    def forward(self, x):
        x = self.low_level_features(x)
        low_level_feat = x
        x = self.high_level_features(x)
        return x, low_level_feat

    def get_train_parameters(self, lr):
        train_params = [{'params': self.parameters(), 'lr': lr}]
        return train_params


def _load_pretrained_model(model, url):
    pretrain_dict = model_zoo.load_url(url)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)


model_urls = {'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'}


def MobileNet_v2(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        _load_pretrained_model(model, model_urls['mobilenet_v2'])
    return model


def MobileNet_v2_dilation(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        _load_pretrained_model(model, model_urls['mobilenet_v2'])
    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, norm_layer=nn.BatchNorm2d, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], norm_layer=norm_layer)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], norm_layer=norm_layer)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat


def ResNet101(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)
    if pretrained:
        _load_pretrained_model(model, RESNET_101)
    return model


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            dilation = 1
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
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


RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def ResNet18(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, norm_layer=norm_layer, input_channels=input_channels)
    if pretrained:
        _load_pretrained_model(model, RESNET_18)
    return model


RESNET_34 = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'


def ResNet34(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(BasicBlock, [3, 4, 23, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)
    if pretrained:
        _load_pretrained_model(model, RESNET_34)
    return model


RESNET_50 = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


def ResNet50(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)
    if pretrained:
        _load_pretrained_model(model, RESNET_50)
    return model


class DeepLabv3_plus(nn.Module):

    def __init__(self, args, num_classes=21, norm_layer=nn.BatchNorm2d, input_channels=3):
        super(DeepLabv3_plus, self).__init__()
        self.args = args
        if args.model == DEEPLAB:
            self.backbone = ResNet101(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256
            if self.args.refine_network:
                self.refine_backbone = ResNet101(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_50:
            self.backbone = ResNet50(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256
            if self.args.refine_network:
                self.refine_backbone = ResNet50(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_34:
            self.backbone = ResNet34(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256
            if self.args.refine_network:
                self.refine_backbone = ResNet34(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_18:
            self.backbone = ResNet18(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256
            if self.args.refine_network:
                self.refine_backbone = ResNet18(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_MOBILENET:
            self.backbone = MobileNet_v2(pretrained=args.pretrained, first_layer_input_channels=input_channels)
            self.aspp_inplanes = 320
            self.decoder_inplanes = 24
            if self.args.refine_network:
                self.refine_backbone = MobileNet_v2(pretrained=args.pretrained, first_layer_input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_MOBILENET_DILATION:
            self.backbone = MobileNet_v2_dilation(pretrained=args.pretrained, first_layer_input_channels=input_channels)
            self.aspp_inplanes = 320
            if self.args.refine_network:
                self.refine_backbone = MobileNet_v2_dilation(pretrained=args.pretrained, first_layer_input_channels=input_channels + num_classes)
            self.decoder_inplanes = 24
        else:
            raise NotImplementedError
        if self.args.use_aspp:
            self.aspp = ASPP(args.output_stride, norm_layer=norm_layer, inplanes=self.aspp_inplanes)
        aspp_outplanes = 256 if self.args.use_aspp else self.aspp_inplanes
        self.decoder = Decoder(num_classes, norm_layer=norm_layer, inplanes=self.decoder_inplanes, aspp_outplanes=aspp_outplanes)
        if self.args.learned_upsampling:
            self.learned_upsampling = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1), nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1))
        if self.args.refine_network:
            if self.args.use_aspp:
                self.refine_aspp = ASPP(args.output_stride, norm_layer=norm_layer, inplanes=self.aspp_inplanes)
            self.refine_decoder = Decoder(num_classes, norm_layer=norm_layer, inplanes=self.decoder_inplanes, aspp_outplanes=aspp_outplanes)
            if self.args.learned_upsampling:
                self.refine_learned_upsampling = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1), nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1))

    def forward(self, input):
        output, low_level_feat = self.backbone(input)
        if self.args.use_aspp:
            output = self.aspp(output)
        output = self.decoder(output, low_level_feat)
        if self.args.learned_upsampling:
            output = self.learned_upsampling(output)
        else:
            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.args.refine_network:
            second_output, low_level_feat = self.refine_backbone(torch.cat((input, output), dim=1))
            if self.args.use_aspp:
                second_output = self.refine_aspp(second_output)
            second_output = self.refine_decoder(second_output, low_level_feat)
            if self.args.learned_upsampling:
                second_output = self.refine_learned_upsampling(second_output)
            else:
                second_output = F.interpolate(second_output, size=input.size()[2:], mode='bilinear', align_corners=True)
            return output, second_output
        return output

    def get_train_parameters(self, lr):
        train_params = [{'params': self.parameters(), 'lr': lr}]
        return train_params


class _PyramidPoolingModule(nn.Module):

    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), nn.BatchNorm2d(reduction_dim, momentum=0.95), nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):

    def __init__(self, num_classes, args=None):
        super(PSPNet, self).__init__()
        resnet = models.resnet101()
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = 1, 1
        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512, momentum=0.95), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(512, num_classes, kernel_size=1))

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


MAXPOOL = 'maxpool'


STRIDECONV = 'strided'


class UNetDownBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer (str)    -- normalization layer
            down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
            bias (boolean)      -- if convolution should use bias
    """

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, down_type=STRIDECONV, outermost=False, innermost=False, dropout=0.2, kernel_size=4, bias=True):
        super(UNetDownBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.use_maxpool = down_type == MAXPOOL
        stride = 1 if self.use_maxpool else 2
        kernel_size = 3 if self.use_maxpool else 4
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.norm = norm_layer(output_nc)
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.conv(x)
            x = self.norm(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout:
                x = self.dropout(x)
            x = self.conv(x)
        else:
            x = self.relu(x)
            if self.dropout:
                x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)
        return x


class UnetConvBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer          -- normalization layer
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
    """

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, padding=0, innermost=False, dropout=0.2):
        super(UnetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=int(padding)))
        block.append(norm_layer(output_nc))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=int(padding)))
        block.append(norm_layer(output_nc))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)
        self.innermost = innermost

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    """
      Constructs a UNet upsampling block

         Parameters:
              input_nc (int)      -- the number of input channels
              output_nc (int)     -- the number of output channels
              norm_layer          -- normalization layer
              outermost (bool)    -- if this module is the outermost module
              innermost (bool)    -- if this module is the innermost module
              user_dropout (bool) -- if use dropout layers.
              kernel_size (int)   -- convolution kernel size
              remove_skip (bool)  -- if skip connections should be disabled or not
      """

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, padding=1, remove_skip=False, outermost=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=2, stride=2)
        self.conv_block = UnetConvBlock(output_nc * 2, output_nc, norm_layer, padding)
        self.outermost = outermost

    def forward(self, x, skip=None):
        out = self.up(x)
        if skip is not None:
            out = torch.cat([out, skip], 1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|

        Parameters:
            num_classes (int)      -- the number of channels in output images
            norm_layer             -- normalization layer
            input_nc               -- number of channels of input image

            Args:
            mode (str)             -- process single frames or sequence of frames
            timesteps (int)        --
            num_downs (int)        -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                      image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet, self).__init__()
        self.refine_network = args.refine_network
        self.num_downs = args.num_downs
        self.ngf = args.ngf
        self.encoder = self.build_encoder(self.num_downs, input_nc, self.ngf, norm_layer, down_type=args.down_type)
        self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)
        if self.refine_network:
            self.refine_encoder = self.build_encoder(self.num_downs, input_nc + num_classes, self.ngf, norm_layer, down_type=args.down_type)
            self.refine_decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)

    def build_encoder(self, num_downs, input_nc, ngf, norm_layer, down_type=STRIDECONV):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks
            
             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer (str)    -- normalization layer
                  down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
        """
        layers = []
        layers.append(UNetDownBlock(input_nc=input_nc, output_nc=ngf, norm_layer=norm_layer, down_type=down_type, outermost=True))
        layers.append(UNetDownBlock(input_nc=ngf, output_nc=ngf * 2, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf * 2, output_nc=ngf * 4, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf * 4, output_nc=ngf * 8, norm_layer=norm_layer, down_type=down_type))
        for i in range(num_downs - 5):
            layers.append(UNetDownBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, down_type=down_type, innermost=True))
        return nn.Sequential(*layers)

    def build_decoder(self, num_downs, num_classes, ngf, norm_layer):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        layers = []
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, innermost=True))
        for i in range(num_downs - 5):
            layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 4, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf * 4, output_nc=ngf * 2, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf * 2, output_nc=ngf, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf, output_nc=num_classes, norm_layer=norm_layer, outermost=True))
        return nn.Sequential(*layers)

    def encoder_forward(self, x, use_refine_network=False):
        skip_connections = []
        model = self.refine_encoder if use_refine_network else self.encoder
        for i, down in enumerate(model):
            x = down(x)
            if down.use_maxpool:
                x = down.maxpool(x)
            if not down.innermost:
                skip_connections.append(x)
        return x, skip_connections

    def decoder_forward(self, x, skip_connections, use_refine_network=False):
        model = self.refine_decoder if use_refine_network else self.decoder
        for i, up in enumerate(model):
            if not up.innermost:
                skip = skip_connections[-i]
                out = torch.cat([skip, out], 1)
                out = up(out)
            else:
                out = up(x)
        return out

    def forward(self, x):
        output, skip_connections = self.encoder_forward(x)
        output = self.decoder_forward(output, skip_connections)
        if self.refine_network:
            second_output, skip_connections = self.encoder_forward(torch.cat((x, output), dim=1), use_refine_network=True)
            second_output = self.decoder_forward(second_output, skip_connections, use_refine_network=True)
            return output, second_output
        return output

    def get_train_parameters(self, lr):
        params = [{'params': self.parameters(), 'lr': lr}]
        return params


class UNet_paper(nn.Module):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|

        Parameters:
            num_classes (int)      -- the number of channels in output images
            norm_layer             -- normalization layer
            input_nc               -- number of channels of input image

            Args:
            mode (str)             -- process single frames or sequence of frames
            timesteps (int)        --
            num_downs (int)        -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                      image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            remove_skip (int [0,1])-- if skip connections should be disabled or not
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet_paper, self).__init__(args)
        self.num_downs = args.num_downs
        self.ngf = args.ngf
        self.encoder = self.build_encoder(self.num_downs, input_nc, self.ngf, norm_layer)
        self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)
        self.decoder_last_conv = nn.Conv2d(self.ngf, num_classes, 1)

    def build_encoder(self, num_downs, input_nc, ngf, norm_layer):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks

             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer          -- normalization layer
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
          """
        layers = []
        layers.append(UnetConvBlock(input_nc=input_nc, output_nc=ngf, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf, output_nc=ngf * 2, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf * 2, output_nc=ngf * 4, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf * 4, output_nc=ngf * 8, norm_layer=norm_layer, padding=1))
        for i in range(num_downs - 5):
            layers.append(UnetConvBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, padding=1, innermost=True))
        return nn.Sequential(*layers)

    def build_decoder(self, num_downs, num_classes, ngf, norm_layer, remove_skip=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer
                remove_skip (int)   -- if skip connections should be disabled or not

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        layers = []
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip))
        for i in range(num_downs - 5):
            layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 4, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf * 4, output_nc=ngf * 2, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf * 2, output_nc=ngf, norm_layer=norm_layer, remove_skip=remove_skip, outermost=True))
        return nn.Sequential(*layers)

    def encoder_forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if not down.innermost:
                skip_connections.append(x)
                x = F.max_pool2d(x, 2)
        return x, skip_connections

    def decoder_forward(self, x, skip_connections):
        out = None
        for i, up in enumerate(self.decoder):
            skip = skip_connections.pop()
            if out is None:
                out = up(x, skip)
            else:
                out = up(out, skip)
        out = self.decoder_last_conv(out)
        return out

    def forward(self, x):
        x, skip_connections = self.encoder_forward(x)
        out = self.decoder_forward(x, skip_connections)
        return out


class UNet_torch(nn.Module):

    def __init__(self, num_classes=1, args=None, in_channels=3):
        super(UNet_torch, self).__init__()
        self.ngf = args.ngf
        self.encoder1 = UNet_torch._block(in_channels, self.ngf, name='enc1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_torch._block(self.ngf, self.ngf * 2, name='enc2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_torch._block(self.ngf * 2, self.ngf * 4, name='enc3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_torch._block(self.ngf * 4, self.ngf * 8, name='enc4')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet_torch._block(self.ngf * 8, self.ngf * 16, name='bottleneck')
        self.upconv4 = nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_torch._block(self.ngf * 8 * 2, self.ngf * 8, name='dec4')
        self.upconv3 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_torch._block(self.ngf * 4 * 2, self.ngf * 4, name='dec3')
        self.upconv2 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_torch._block(self.ngf * 2 * 2, self.ngf * 2, name='dec2')
        self.upconv1 = nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=2, stride=2)
        self.decoder1 = UNet_torch._block(self.ngf * 2, self.ngf, name='dec1')
        self.conv = nn.Conv2d(in_channels=self.ngf, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        final = self.conv(dec1)
        return final

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([(name + 'conv1', nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)), (name + 'norm1', nn.BatchNorm2d(num_features=features)), (name + 'relu1', nn.ReLU(inplace=True)), (name + 'conv1', nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)), (name + 'norm1', nn.BatchNorm2d(num_features=features)), (name + 'relu1', nn.ReLU(inplace=True))]))

    def get_train_parameters(self, lr):
        params = [{'params': self.parameters(), 'lr': lr}]
        return params


class DocunetLoss_v2(nn.Module):

    def __init__(self, r=0.1, reduction='mean'):
        super(DocunetLoss_v2, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.r = r
        self.reduction = reduction

    def forward(self, y, label):
        bs, n, h, w = y.size()
        d = y - label
        loss1 = []
        for d_i in d:
            loss1.append(torch.abs(d_i).mean() - self.r * torch.abs(d_i.mean()))
        loss1 = torch.stack(loss1)
        loss2 = F.mse_loss(y, label, reduction=self.reduction)
        if self.reduction == 'mean':
            loss1 = loss1.mean()
        elif self.reduction == 'sum':
            loss1 = loss1.sum()
        return loss1 + loss2


class DocunetLoss(nn.Module):

    def __init__(self, lamda=0.1, reduction='mean'):
        super(DocunetLoss, self).__init__()
        self.lamda = lamda
        self.reduction = reduction

    def forward(self, output, target):
        x = target[:, 0, :, :]
        y = target[:, 1, :, :]
        back_sign_x, back_sign_y = (x == -1).int(), (y == -1).int()
        back_sign = (back_sign_x + back_sign_y == 2).float()
        fore_sign = 1 - back_sign
        loss_term_1_x = torch.sum(torch.abs(output[:, 0, :, :] - x) * fore_sign) / torch.sum(fore_sign)
        loss_term_1_y = torch.sum(torch.abs(output[:, 1, :, :] - y) * fore_sign) / torch.sum(fore_sign)
        loss_term_1 = loss_term_1_x + loss_term_1_y
        loss_term_2_x = torch.abs(torch.sum((output[:, 0, :, :] - x) * fore_sign)) / torch.sum(fore_sign)
        loss_term_2_y = torch.abs(torch.sum((output[:, 1, :, :] - y) * fore_sign)) / torch.sum(fore_sign)
        loss_term_2 = loss_term_2_x + loss_term_2_y
        zeros_x = torch.zeros(x.size()) if torch.cuda.is_available() else torch.zeros(x.size())
        zeros_y = torch.zeros(y.size()) if torch.cuda.is_available() else torch.zeros(y.size())
        loss_term_3_x = torch.max(zeros_x, output[:, 0, :, :])
        loss_term_3_y = torch.max(zeros_y, output[:, 1, :, :])
        loss_term_3 = torch.sum((loss_term_3_x + loss_term_3_y) * back_sign) / torch.sum(back_sign)
        loss = loss_term_1 - self.lamda * loss_term_2 + loss_term_3
        return loss


def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size)
    coords -= size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, normalize=False):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
    if normalize:
        ssim_map = F.relu(ssim_map, inplace=True)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


class SSIM_v2(torch.nn.Module):

    def __init__(self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3, K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
        """
        super(SSIM_v2, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


def gaussian_filter(input, win):
    """ Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False, K=(0.01, 0.03), nonnegative_ssim=False):
    """ Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if nonnegative_ssim:
        cs_map = F.relu(cs_map, inplace=True)
    ssim_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs_map
    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)
        cs = cs_map.mean(-1).mean(-1).mean(-1)
    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not win_size % 2 == 1:
        raise ValueError('Window size must be odd.')
    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]
    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, full=True, K=K, nonnegative_ssim=nonnegative_ssim)
        mcs.append(cs)
        padding = X.shape[2] % 2, X.shape[3] % 2
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
    mcs = torch.stack(mcs, dim=0)
    msssim_val = torch.prod(mcs[:-1] ** weights[:-1].unsqueeze(1) * ssim_val ** weights[-1], dim=0)
    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


class MS_SSIM_v2(torch.nn.Module):

    def __init__(self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
        """
        super(MS_SSIM_v2, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range, normalize=False)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class MS_SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DocunetLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UNetDownBlock,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnetConvBlock,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (_ASPPModule,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

