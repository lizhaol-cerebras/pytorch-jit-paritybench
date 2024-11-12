
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


from itertools import product as product


from math import ceil


import numpy as np


import torch


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models._utils as _utils


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
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class PriorBox(object):

    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = 's'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [(x * self.steps[k] / self.image_size[1]) for x in [j + 0.5]]
                    dense_cy = [(y * self.steps[k] / self.image_size[0]) for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


cfg_mnet = {'name': 'mobilenet0.25', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 32, 'ngpu': 1, 'epoch': 250, 'decay1': 190, 'decay2': 220, 'image_size': 640, 'pretrain': True, 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}, 'in_channel': 32, 'out_channel': 64}


def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), dim=1)
    return landms


def parse_det(det):
    landmarks = det[5:].reshape(5, 2)
    box = det[:4]
    score = det[4]
    return box, landmarks, score


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def post_process(loc, conf, landms, prior_data, cfg, scale, scale1, resize, confidence_threshold, top_k, nms_threshold, keep_top_k):
    boxes = decode(loc, prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.cpu().numpy()[:, 1]
    landms_copy = decode_landm(landms, prior_data, cfg['variance'])
    landms_copy = landms_copy * scale1 / resize
    landms_copy = landms_copy.cpu().numpy()
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms_copy = landms_copy[inds]
    scores = scores[inds]
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms_copy = landms_copy[order]
    scores = scores[order]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms_copy = landms_copy[keep]
    dets = dets[:keep_top_k, :]
    landms_copy = landms_copy[:keep_top_k, :]
    dets = np.concatenate((dets, landms_copy), axis=1)
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    dets = [parse_det(x) for x in dets]
    return dets


def batch_detect(net, images, device, is_tensor=False, normalized=False):
    with torch.no_grad():
        confidence_threshold = 0.02
        cfg = cfg_mnet
        top_k = 5000
        nms_threshold = 0.4
        keep_top_k = 750
        resize = 1
        if not is_tensor:
            try:
                img = np.float32(images)
            except ValueError:
                raise NotImplementedError('Input images must of same size')
            img = torch.from_numpy(img)
        else:
            img = images.float()
        img = img
        mean = torch.as_tensor([104, 117, 123], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        img -= mean
        img = img.permute(0, 3, 1, 2)
        batch_size, _, im_height, im_width = img.shape
        scale = torch.as_tensor([im_width, im_height, im_width, im_height], dtype=img.dtype, device=img.device)
        scale = scale
        loc, conf, landms = net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors
        scale1 = torch.as_tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3], img.shape[2]], dtype=img.dtype, device=img.device)
        scale1 = scale1
        all_dets = [post_process(loc_i, conf_i, landms_i, prior_data, cfg, scale, scale1, resize, confidence_threshold, top_k, nms_threshold, keep_top_k) for loc_i, conf_i, landms_i in zip(loc, conf, landms)]
        return all_dets

