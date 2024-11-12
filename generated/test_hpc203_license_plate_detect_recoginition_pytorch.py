
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


import time


import torch.nn as nn


import torchvision.models._utils as _utils


import torchvision.models as models


import torch.nn.functional as F


from torch.autograd import Variable


from itertools import product as product


from math import ceil


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


class SSH(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)
        out = [output1, output2, output3]
        return out


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.LeakyReLU(negative_slope=leaky, inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(conv_bn(3, 8, 2, leaky=0.1), conv_dw(8, 16, 1), conv_dw(16, 32, 2), conv_dw(32, 32, 1), conv_dw(32, 64, 2), conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1))
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 8, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 8)


class Retina(nn.Module):

    def __init__(self, cfg=None):
        super(Retina, self).__init__()
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        output = bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions
        return output


class small_basic_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(ch_in, ch_out // 4, kernel_size=1), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)), nn.ReLU(), nn.Conv2d(ch_out // 4, ch_out, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class my_lprnet(nn.Module):

    def __init__(self, class_num):
        super(my_lprnet, self).__init__()
        self.class_num = class_num
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.stage2 = nn.Sequential(small_basic_block(ch_in=64, ch_out=128), nn.BatchNorm2d(num_features=128), nn.ReLU())
        self.stage3 = nn.Sequential(small_basic_block(ch_in=64, ch_out=256), nn.BatchNorm2d(num_features=256), nn.ReLU(), small_basic_block(ch_in=256, ch_out=256), nn.BatchNorm2d(num_features=256), nn.ReLU())
        self.stage4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1), nn.BatchNorm2d(num_features=256), nn.ReLU(), nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), nn.BatchNorm2d(num_features=class_num), nn.ReLU())
        self.container = nn.Conv2d(in_channels=448 + class_num, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out1 = self.stage1(x)
        out = F.max_pool2d(out1, 3, stride=(1, 1))
        out2 = self.stage2(out)
        out = F.max_pool2d(out2, 3, stride=(1, 2))
        out = F.max_pool2d(out.permute(0, 2, 3, 1).contiguous(), 1, stride=(1, 2))
        out = out.permute(0, 3, 1, 2).contiguous()
        out3 = self.stage3(out)
        out = F.max_pool2d(out3, 3, stride=(1, 2))
        out = F.max_pool2d(out.permute(0, 2, 3, 1).contiguous(), 1, stride=(1, 4))
        out = out.permute(0, 3, 1, 2).contiguous()
        out4 = self.stage4(out)
        out1 = F.avg_pool2d(out1, kernel_size=5, stride=5)
        f = torch.pow(out1, 2)
        f = torch.mean(f)
        out1 = torch.div(out1, f.item())
        out2 = F.avg_pool2d(out2, kernel_size=5, stride=5)
        f = torch.pow(out2, 2)
        f = torch.mean(f)
        out2 = torch.div(out2, f.item())
        out3 = F.avg_pool2d(out3, kernel_size=(4, 10), stride=(4, 2))
        f = torch.pow(out3, 2)
        f = torch.mean(f)
        out3 = torch.div(out3, f.item())
        f = torch.pow(out4, 2)
        f = torch.mean(f)
        out4 = torch.div(out4, f.item())
        logits = torch.cat((out1, out2, out3, out4), 1)
        logits = self.container(logits)
        logits = torch.mean(logits, dim=2)
        logits = logits.view(self.class_num, -1)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BboxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ClassHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (LandmarkHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (MobileNetV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SSH,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (small_basic_block,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

