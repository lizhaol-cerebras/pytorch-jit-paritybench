
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


import numpy as np


import math


from copy import deepcopy


from functools import partial


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim as optim


from torch.utils.data import DataLoader


import matplotlib


import scipy.signal


from matplotlib import pyplot as plt


from torch.utils.tensorboard import SummaryWriter


from random import sample


from random import shuffle


from torch.utils.data.dataset import Dataset


import random


from torchvision.ops import nms


import time


class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Multi_Concat_Block(nn.Module):

    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList([Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)])
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


class MP(nn.Module):

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):

    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        self.mp = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        return torch.cat([x_2, x_1], 1)


class Backbone(nn.Module):

    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        ids = {'l': [-1, -3, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        self.stem = nn.Sequential(Conv(3, transition_channels, 3, 1), Conv(transition_channels, transition_channels * 2, 3, 2), Conv(transition_channels * 2, transition_channels * 2, 3, 1))
        self.dark2 = nn.Sequential(Conv(transition_channels * 2, transition_channels * 4, 3, 2), Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids))
        self.dark3 = nn.Sequential(Transition_Block(transition_channels * 8, transition_channels * 4), Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids))
        self.dark4 = nn.Sequential(Transition_Block(transition_channels * 16, transition_channels * 8), Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids))
        self.dark5 = nn.Sequential(Transition_Block(transition_channels * 32, transition_channels * 16), Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids))
        if pretrained:
            url = {'l': 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth', 'x': 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth'}[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', model_dir='./model_data')
            self.load_state_dict(checkpoint, strict=False)
            None

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


class SPPCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False), nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False), nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03))

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t
        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True, padding_mode=conv.padding_mode)
        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        None
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        if isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm):
            identity_conv_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))
        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
        self.rbr_reparam = self.rbr_dense
        self.deploy = True
        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None
        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None
        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv


class YoloBody(nn.Module):

    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]
        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)
        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)
        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)
        self.down_sample2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        None
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone.forward(x)
        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)
        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)
        return [out0, out1, out2]


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smoothing=0):
        super(YOLOLoss, self).__init__()
        self.anchors = [anchors[mask] for mask in anchors_mask]
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.balance = [0.4, 1.0, 4]
        self.stride = [32, 16, 8]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / 640 ** 2
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.threshold = 4
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)
        self.BCEcls, self.BCEobj, self.gr = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07):
        box2 = box2.T
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    v = 4 / math.pi ** 2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
            else:
                c_area = cw * ch + eps
                return iou - (c_area - union) / c_area
        else:
            return iou

    def __call__(self, predictions, targets, imgs):
        for i in range(len(predictions)):
            bs, _, h, w = predictions[i].size()
            predictions[i] = predictions[i].view(bs, len(self.anchors_mask[i]), -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
        device = targets.device
        cls_loss, box_loss, obj_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(predictions, targets, imgs)
        feature_map_sizes = [torch.tensor(prediction.shape, device=device)[[3, 2, 3, 2]].type_as(prediction) for prediction in predictions]
        for i, prediction in enumerate(predictions):
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = torch.zeros_like(prediction[..., 0], device=device)
            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]
                grid = torch.stack([gi, gj], dim=1)
                xy = prediction_pos[:, :2].sigmoid() * 2.0 - 0.5
                wh = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box = torch.cat((xy, wh), 1)
                selected_tbox = targets[i][:, 2:6] * feature_map_sizes[i]
                selected_tbox[:, :2] -= grid.type_as(prediction)
                iou = self.bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = 1.0 - self.gr + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                selected_tcls = targets[i][:, 1].long()
                t = torch.full_like(prediction_pos[:, 5:], self.cn, device=device)
                t[range(n), selected_tcls] = self.cp
                cls_loss += self.BCEcls(prediction_pos[:, 5:], t)
            obj_loss += self.BCEobj(prediction[..., 4], tobj) * self.balance[i]
        box_loss *= self.box_ratio
        obj_loss *= self.obj_ratio
        cls_loss *= self.cls_ratio
        bs = tobj.shape[0]
        loss = box_loss + obj_loss + cls_loss
        return loss

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def box_iou(self, box1, box2):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def build_targets(self, predictions, targets, imgs):
        indices, anch = self.find_3_positive(predictions, targets)
        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]
        num_layer = len(predictions)
        for batch_idx in range(predictions[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = self.xywh2xyxy(txywh)
            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            for i, prediction in enumerate(predictions):
                b, a, gj, gi = indices[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                fg_pred = prediction[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                grid = torch.stack([gi, gj], dim=1).type_as(fg_pred)
                pxy = (fg_pred[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
            pair_wise_iou = self.box_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-08)
            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
            gt_cls_per_image = F.one_hot(this_target[:, 1], self.num_classes).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
            num_gt = this_target.shape[0]
            cls_preds_ = p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image, reduction='none').sum(-1)
            del cls_preds_
            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
            matching_matrix = torch.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
            this_target = this_target[matched_gt_inds]
            for i in range(num_layer):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])
        for i in range(num_layer):
            matching_bs[i] = torch.cat(matching_bs[i], dim=0) if len(matching_bs[i]) != 0 else torch.Tensor(matching_bs[i])
            matching_as[i] = torch.cat(matching_as[i], dim=0) if len(matching_as[i]) != 0 else torch.Tensor(matching_as[i])
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0) if len(matching_gjs[i]) != 0 else torch.Tensor(matching_gjs[i])
            matching_gis[i] = torch.cat(matching_gis[i], dim=0) if len(matching_gis[i]) != 0 else torch.Tensor(matching_gis[i])
            matching_targets[i] = torch.cat(matching_targets[i], dim=0) if len(matching_targets[i]) != 0 else torch.Tensor(matching_targets[i])
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0) if len(matching_anchs[i]) != 0 else torch.Tensor(matching_anchs[i])
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        num_anchor, num_gt = len(self.anchors_mask[0]), targets.shape[0]
        indices, anchors = [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, num_gt)
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g
        for i in range(len(predictions)):
            anchors_i = torch.from_numpy(self.anchors[i] / self.stride[i]).type_as(predictions[i])
            anchors_i, shape = torch.from_numpy(self.anchors[i] / self.stride[i]).type_as(predictions[i]), predictions[i].shape
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if num_gt:
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.threshold
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            anchors.append(anchors_i[a])
        return indices, anchors


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Multi_Concat_Block,
     lambda: ([], {'c1': 4, 'c2': 4, 'c3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SPPCSPC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transition_Block,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

