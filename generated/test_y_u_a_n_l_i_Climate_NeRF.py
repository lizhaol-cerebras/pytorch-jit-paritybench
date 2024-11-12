
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


from torch.utils.data import Dataset


import numpy as np


import tensorflow as tf


import torch


from torchvision import transforms


import copy


from typing import List


from typing import Optional


from typing import Dict


from typing import Tuple


from typing import Union


from typing import Type


from torch.utils.data import DataLoader


import math


from itertools import cycle


import torch.nn.functional as F


import scipy


import torch.nn as nn


from torch import nn


import time


from torch.autograd import Variable


import torchvision.transforms as transforms


import torchvision.utils as utils


from collections import namedtuple


import torchvision.transforms as T


import random


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch import autograd


import warnings


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


import torch.functional as F


from logging.config import valid_ident


class VGGEncoder(nn.Module):

    def __init__(self, level):
        super(VGGEncoder, self).__init__()
        self.level = level
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        if level < 2:
            return
        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        if level < 3:
            return
        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        if level < 4:
            return
        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)
        if self.level < 2:
            return out
        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        out, pool1_idx = self.maxpool1(pool1)
        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        if self.level < 3:
            return out, pool1_idx, pool1.size()
        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)
        out, pool2_idx = self.maxpool2(pool2)
        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        if self.level < 4:
            return out, pool1_idx, pool1.size(), pool2_idx, pool2.size()
        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)
        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        return out, pool1_idx, pool1.size(), pool2_idx, pool2.size(), pool3_idx, pool3.size()

    def forward_multiple(self, x):
        out = self.conv0(x)
        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)
        if self.level < 2:
            return out
        out1 = out
        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        out, pool1_idx = self.maxpool1(pool1)
        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        if self.level < 3:
            return out, out1
        out2 = out
        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)
        out, pool2_idx = self.maxpool2(pool2)
        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        if self.level < 4:
            return out, out2, out1
        out3 = out
        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)
        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        return out, out3, out2, out1


class VGGDecoder(nn.Module):

    def __init__(self, level):
        super(VGGDecoder, self).__init__()
        self.level = level
        if level > 3:
            self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu4_1 = nn.ReLU(inplace=True)
            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_4 = nn.ReLU(inplace=True)
            self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_3 = nn.ReLU(inplace=True)
            self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_2 = nn.ReLU(inplace=True)
        if level > 2:
            self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu3_1 = nn.ReLU(inplace=True)
            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu2_2 = nn.ReLU(inplace=True)
        if level > 1:
            self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu2_1 = nn.ReLU(inplace=True)
            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu1_2 = nn.ReLU(inplace=True)
        if level > 0:
            self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
        out = x
        if self.level > 3:
            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            out = self.unpool3(out, pool3_idx, output_size=pool3_size)
            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)
            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)
            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)
        if self.level > 2:
            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            out = self.unpool2(out, pool2_idx, output_size=pool2_size)
            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)
        if self.level > 1:
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            out = self.unpool1(out, pool1_idx, output_size=pool1_size)
            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)
        if self.level > 0:
            out = self.pad1_1(out)
            out = self.conv1_1(out)
        return out


BaseImage = collections.namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


class GIFSmoothing(nn.Module):

    def forward(self, *input):
        pass

    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)

    def process_opencv(self, initImg, contentImg):
        """
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        """
        if type(initImg) == str:
            init_img = cv2.imread(initImg)
            init_img = init_img[2:-2, 2:-2, :]
        else:
            init_img = np.array(initImg)[:, :, ::-1].copy()
        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
        else:
            cont_img = np.array(contentImg)[:, :, ::-1].copy()
        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img


class PhotoWCT(nn.Module):

    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)

    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)
        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)
        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)
        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)
        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2, cpool_idx, cpool)
        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1)
        return Im1

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            if l in [0, 1, 5, 6]:
                self.label_indicator[l] = False
                continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()
        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__wct_core(cont_feat_view, styl_feat_view)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))
            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 1 or styl_mask[0].size <= 1:
                    continue
                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                if self.is_cuda:
                    cont_indi = cont_indi
                    styl_indi = styl_indi
                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
                tmp_target_feature = self.__wct_core(cFFG, sFFG)
                if torch.__version__ >= '0.4.0':
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, torch.transpose(tmp_target_feature, 1, 0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)
        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF

    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        iden = torch.eye(cFSize[0])
        if self.is_cuda:
            iden = iden
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 1e-05:
                k_c = i + 1
                break
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 1e-05:
                k_s = i + 1
                break
        c_d = c_e[0:k_c].pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, c_v[:, 0:k_c].t())
        whiten_cF = torch.mm(step2, cont_feat)
        s_d = s_e[0:k_s].pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), s_v[:, 0:k_s].t()), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)
    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        loss: (N_rays)
    """

    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class ExponentialAnnealingWeight:

    def __init__(self, max, min, k):
        super().__init__()
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur * self.k))


def compute_scale_and_shift(prediction, target):
    a_00 = torch.sum(prediction * prediction)
    a_01 = torch.sum(prediction)
    ones = torch.ones_like(prediction)
    a_11 = torch.sum(ones)
    b_0 = torch.sum(prediction * target)
    b_1 = torch.sum(target)
    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0 = torch.FloatTensor(0)
        x_1 = torch.FloatTensor(0)
    return x_0, x_1


class NeRFLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.lambda_opa = 0.0002
        self.lambda_distortion = 0.001
        self.lambda_depth_mono = 1
        self.lambda_normal_mono = 0.0001
        self.lambda_sky = 0.1
        self.lambda_semantic = 0.01
        self.lambda_normal_rp = 0.001
        self.Annealing = ExponentialAnnealingWeight(max=1, min=0.06, k=0.001)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=256)

    def forward(self, results, target, **kwargs):
        d = {}
        if kwargs.get('embed_msk', False):
            d['r_ms'], _ = self.mask_regularize(kwargs['mask'], self.Annealing.getWeight(kwargs['step']), 0)
            d['rgb'] = (1 - kwargs['mask']) * (results['rgb'] - target['rgb']) ** 2
        else:
            d['rgb'] = (results['rgb'] - target['rgb']) ** 2
        if not kwargs.get('stylize', False):
            if kwargs.get('normal_p', False):
                d['Rp'] = self.lambda_normal_rp * (results['Rp'] - torch.zeros_like(results['Rp']))
                d['Ro'] = 0.001 * self.lambda_normal_rp * results['Ro']
            if self.lambda_distortion > 0:
                d['distortion'] = self.lambda_distortion * DistortionLoss.apply(results['ws'], results['deltas'], results['ts'], results['rays_a'])
            if kwargs.get('normal_mono', False):
                d['normal_mono'] = self.lambda_normal_mono * torch.exp(-results['depth'].detach() / kwargs.get('scale', 1))[:, None] * (target['normal'] - results['normal_pred']) ** 2
            if kwargs.get('semantic', False):
                d['CELoss'] = self.lambda_semantic * self.CrossEntropyLoss(results['semantic'], target['label'])
                sky_mask = torch.where(target['label'] == kwargs.get('sky_label', 4), 1.0, 0.0)
                d['sky_depth'] = self.lambda_sky * sky_mask * torch.exp(-results['depth'])
            if kwargs.get('depth_mono', False):
                depth_2d = target['depth'] / 25
                mask = depth_2d > 0
                weight = torch.zeros_like(depth_2d)
                weight[mask] = 1.0
                scale, shift = compute_scale_and_shift(results['depth'][mask].detach(), depth_2d[mask])
                d['depth_mono'] = weight * self.lambda_depth_mono * torch.exp(-results['depth'].detach() / kwargs.get('scale', 1)) * (scale * results['depth'] + shift - depth_2d) ** 2
        return d

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta
        loss_focus_digit = 1 / ((mask - 0.5) ** 2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta
        return loss_focus_size, loss_focus_digit


class implicit_mask(nn.Module):

    def __init__(self, latent=32, W=128):
        super().__init__()
        L = 8
        F = 2
        log2_T = 16
        N_min = 16
        b = np.exp(np.log(2048 / N_min) / (L - 1))
        None
        self.mask_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Grid', 'type': 'Hash', 'n_levels': L, 'n_features_per_level': F, 'log2_hashmap_size': log2_T, 'base_resolution': N_min, 'per_level_scale': b, 'interpolation': 'Linear'})
        self.mask_net = nn.Sequential(nn.Linear(self.mask_encoder.n_output_dims, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, uvi):
        uvi_enc = self.mask_encoder(uvi)
        mask = self.mask_net(uvi_enc)
        return mask


def dkernel_function(density_o, R, r):
    """
    calculate derivatives for densities inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    ddensity_dr: derivatives of densities of samples (N_samples, ) 
    """
    r = torch.clamp(r, max=R)
    ddensity_dr = -6 * 315 / (64 * torch.pi * 1.5 ** 7) * (1.5 ** 2 - (r / R) ** 2) ** 2 * (r / R ** 2) * density_o
    ddensity_dr = torch.clamp(ddensity_dr, max=-0.0001)
    return ddensity_dr


def kernel_function(density_o, R, r):
    """
    calculate densities for points inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    density_r: densities of samples (N_samples, ) 
    """
    r = torch.clamp(r, max=R)
    density_r = 315 / (64 * torch.pi * 1.5 ** 7) * (1.5 ** 2 - (r / R) ** 2) ** 3 * density_o
    density_r = torch.clamp(density_r, min=0)
    return density_r


def wrap_light(L, dir, wrap=8):
    """
    Input:
    L: normalized light direction
    dir: direction related to the metaball center
    
    Output:
    diffuse_scale: a grey scale for diffuse color
    """
    dot = torch.sum(L[None, :] * dir, dim=-1)
    diffuse_scale = (dot + wrap) / (1.0 + wrap)
    return diffuse_scale


class NGP_mb(nn.Module):

    def __init__(self, scale, up, ground_height, R, R_inv, interval, b=1.5, rgb_act='Sigmoid'):
        super().__init__()
        self.rgb_act = rgb_act
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)
        self.up = up
        self.ground_height = ground_height
        self.interval = interval
        self.R = R
        self.R_inv = R_inv
        self.mb_cascade = 5
        self.b = b
        L_mb = 8
        F_mb = 2
        log2_T_mb = 19
        N_min_mb = 32
        b_mb = np.exp(np.log(2048 * scale / N_min_mb) / (L_mb - 1))
        None
        self.mb_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Grid', 'type': 'Hash', 'n_levels': L_mb, 'n_features_per_level': F_mb, 'log2_hashmap_size': log2_T_mb, 'base_resolution': N_min_mb, 'per_level_scale': b_mb, 'interpolation': 'Linear'})
        self.mb_net = nn.Sequential(nn.Linear(self.mb_encoder.n_output_dims, 32), nn.ReLU(), nn.Linear(32, 1))
        self.mb_act = nn.Sigmoid()
        self.grey_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Grid', 'type': 'Hash', 'n_levels': L_mb, 'n_features_per_level': F_mb, 'log2_hashmap_size': log2_T_mb, 'base_resolution': N_min_mb, 'per_level_scale': b_mb, 'interpolation': 'Linear'})
        self.rgb_net = nn.Sequential(nn.Linear(self.grey_encoder.n_output_dims, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rgb_act = nn.Sigmoid()

    def alpha(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        h = self.mb_encoder(x)
        h = self.mb_net(h)
        alphas = self.mb_act(h[:, 0])
        h = self.grey_encoder(x)
        h = self.rgb_net(h)
        rgbs = self.rgb_act(h[:, 0:])
        return alphas, rgbs

    def forward(self, x, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3)
        alphas, rgbs = self.alpha(x)
        return alphas, rgbs.squeeze(-1)

    def forward_test(self, x, density_only=False, geometry_model=None, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        with torch.no_grad():
            x_sf = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3)
            N_samples = x_sf.shape[0]
            x_sf_vertices = []
            x_sf_dis = []
            x_sf_radius = []
            x_sf_grad = []
            for i in range(self.mb_cascade):
                radius = self.interval / self.b ** i
                x_sf_coord = torch.floor(x_sf / radius) * radius
                offsets = radius * torch.FloatTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
                x_sf_vertices_ = x_sf_coord[:, None, :] + offsets[None, :, :]
                x_sf_dis_ = torch.norm(x_sf_vertices_ - x_sf[:, None, :], dim=-1)
                x_sf_grad_ = (x_sf[:, None, :] - x_sf_vertices_) / x_sf_dis_[..., None]
                x_sf_vertices.append(x_sf_vertices_.reshape(-1, 3))
                x_sf_dis.append(x_sf_dis_.reshape(-1))
                x_sf_radius.append(radius * torch.ones(x_sf_vertices_.shape[0] * x_sf_vertices_.shape[1]))
                x_sf_grad.append(x_sf_grad_.reshape(-1, 3))
            x_sf_vertices = torch.cat(x_sf_vertices, dim=0)
            x_sf_dis = torch.cat(x_sf_dis, dim=0)
            x_sf_radius = torch.cat(x_sf_radius, dim=0)
            x_sf_grad = torch.cat(x_sf_grad, dim=0)
            alpha, rgbs = self.alpha(x_sf_vertices)
            x_sf_colmap = torch.matmul(self.R, x_sf_vertices.reshape(-1, 3, 1)).reshape(-1, 3)
            if geometry_model is not None:
                valid_mask = vren.test_occ(x_sf_colmap, geometry_model.density_bitfield, geometry_model.cascades, geometry_model.scale, geometry_model.grid_size)[0]
                alpha[valid_mask < 1] *= 0
            if kwargs.get('cal_snow_occ', False):
                weighted_sigmoid = lambda x, weight, bias: 1.0 / (1 + torch.exp(-weight * (x - bias)))
                snow_occ = weighted_sigmoid(kwargs['snow_occ_net'](x_sf_colmap), 10, 0.5)
                alpha *= snow_occ
            rgbs = (rgbs + 4) / (1 + 4)
            if kwargs.get('pred_shadow', False):
                shadow = 0.7 * (kwargs['sun_vis_net'](x_sf_colmap) < 0.5).float() + 0.3
                rgbs *= shadow[:, None]
            center_density = kwargs.get('center_density', 2000.0)
            density_c = alpha * center_density
            density_sample = kernel_function(density_c, x_sf_radius, x_sf_dis)
            ddensity_sample_dxsf = dkernel_function(density_c, x_sf_radius, x_sf_dis)[:, None] * x_sf_grad
            densities = torch.chunk(density_sample, self.mb_cascade, dim=0)
            sigmas = torch.stack(densities, dim=-1)
            rgbs = torch.chunk(rgbs, self.mb_cascade, dim=0)
            rgbs = torch.stack(rgbs, dim=1).reshape(N_samples, 8 * self.mb_cascade) * sigmas.view(N_samples, 8 * self.mb_cascade)
            sigmas = torch.sum(sigmas.view(N_samples, self.mb_cascade * 8), dim=-1)
            rgbs = (torch.sum(rgbs, dim=-1, keepdim=True) / (sigmas.view(N_samples, 1) + 1e-06)).expand(-1, 3)
            weighted_sigmoid = lambda x, weight, bias: 1.0 / (1 + torch.exp(-weight * (x - bias)))
            thres_ratio = kwargs.get('mb_thres', 1 / 8)
            thres = weighted_sigmoid(sigmas, 50, center_density * thres_ratio)
            sigmas = sigmas * thres
            if density_only:
                return sigmas
            ddensities_dxsf = torch.chunk(ddensity_sample_dxsf, self.mb_cascade, dim=0)
            normals = torch.stack(ddensities_dxsf, dim=1)
            normals = torch.sum(normals.reshape(N_samples, 8 * self.mb_cascade, 3), dim=1)
            normals = -F.normalize(normals, dim=-1)
        rgbs = rgbs * wrap_light(torch.FloatTensor([0, -1.0, 0]), normals)[:, None]
        return sigmas, rgbs, normals


class vis_net(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        L = 8
        F = 2
        log2_T = 19
        N_min = 32
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        None
        self.vis_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Grid', 'type': 'Hash', 'n_levels': L, 'n_features_per_level': F, 'log2_hashmap_size': log2_T, 'base_resolution': N_min, 'per_level_scale': b, 'interpolation': 'Linear'})
        self.vis_net = nn.Sequential(nn.Linear(self.vis_encoder.n_output_dims, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        x_enc = self.vis_encoder(x)
        vis = self.vis_net(x_enc)
        return vis[:, 0]


class NGP(nn.Module):

    def __init__(self, scale, rgb_act='Sigmoid', use_skybox=False, embed_a=False, embed_a_len=12, classes=7):
        super().__init__()
        self.rgb_act = rgb_act
        self.scale = scale
        self.use_skybox = use_skybox
        self.embed_a = embed_a
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield', torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))
        G = self.grid_size
        self.register_buffer('density_grid', torch.zeros(self.cascades, G ** 3))
        self.register_buffer('grid_coords', create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        None
        self.xyz_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'Grid', 'type': 'Hash', 'n_levels': L, 'n_features_per_level': F, 'log2_hashmap_size': log2_T, 'base_resolution': N_min, 'per_level_scale': b, 'interpolation': 'Linear'})
        self.xyz_net = nn.Sequential(nn.Linear(self.xyz_encoder.n_output_dims, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sigma_act = nn.Softplus()
        L_ = 32
        F_ = 2
        log2_T_ = 19
        N_min_ = 16
        b_ = np.exp(np.log(2048 * scale / N_min_) / (L_ - 1))
        None
        self.rgb_encoder = tcnn.Encoding(3, {'otype': 'HashGrid', 'n_levels': L_, 'n_features_per_level': F_, 'log2_hashmap_size': log2_T_, 'base_resolution': N_min_, 'per_level_scale': b_, 'interpolation': 'Linear'})
        self.dir_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'SphericalHarmonics', 'degree': 4})
        rgb_input_dim = self.rgb_encoder.n_output_dims + self.dir_encoder.n_output_dims
        None
        self.rgb_net = nn.Sequential(nn.Linear(rgb_input_dim + embed_a_len if embed_a else rgb_input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid())
        self.norm_pred_header = nn.Sequential(nn.Linear(self.rgb_encoder.n_output_dims, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        self.semantic_header = nn.Sequential(nn.Linear(self.rgb_encoder.n_output_dims, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, classes))
        self.semantic_act = nn.Softmax(dim=-1)
        if use_skybox:
            None
            self.skybox_dir_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={'otype': 'SphericalHarmonics', 'degree': 3})
            self.skybox_rgb_net = tcnn.Network(n_input_dims=9, n_output_dims=3, network_config={'otype': 'CutlassMLP', 'activation': 'ReLU', 'output_activation': rgb_act, 'n_neurons': 32, 'n_hidden_layers': 1})

    def density(self, x, return_feat=False, grad=True, grad_feat=True):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        with torch.set_grad_enabled(grad):
            h = self.xyz_encoder(x)
            h = self.xyz_net(h)
            sigmas = self.sigma_act(h[:, 0] - 1)
        if return_feat:
            with torch.set_grad_enabled(grad_feat):
                feat_rgb = self.rgb_encoder(x)
            return sigmas, feat_rgb
        return sigmas

    @torch.enable_grad()
    def grad(self, x):
        x = x.requires_grad_(True)
        sigmas, feat_rgb = self.density(x, return_feat=True)
        grads = torch.autograd.grad(outputs=sigmas, inputs=x, grad_outputs=torch.ones_like(sigmas, requires_grad=False), retain_graph=True, create_graph=True)[0]
        return sigmas, feat_rgb, grads

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb, grads = self.grad(x)
        if torch.any(torch.isnan(sigmas)):
            None
        if torch.any(torch.isinf(sigmas)):
            None
        cnt = torch.sum(torch.isinf(grads))
        if torch.any(torch.isnan(grads)):
            None
        if torch.any(torch.isinf(grads)):
            None
        normals_raw = -F.normalize(grads, p=2, dim=-1, eps=1e-06)
        if torch.any(torch.isnan(normals_raw)):
            None
        if torch.any(torch.isinf(normals_raw)):
            None
        normals_pred = self.norm_pred_header(feat_rgb)
        normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-06)
        if torch.any(torch.isnan(normals_pred)):
            None
        if torch.any(torch.isinf(normals_pred)):
            None
        semantic = self.semantic_header(feat_rgb)
        semantic = self.semantic_act(semantic)
        d = F.normalize(d, p=2, dim=-1, eps=1e-06)
        d = self.dir_encoder((d + 1) / 2)
        if self.embed_a:
            rgbs = self.rgb_net(torch.cat([d, feat_rgb, kwargs['embedding_a']], 1))
        else:
            rgbs = self.rgb_net(torch.cat([d, feat_rgb], 1))
        return sigmas, rgbs, normals_raw, normals_pred, semantic, cnt

    def forward_test(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb = self.density(x, return_feat=True, grad=False, grad_feat=kwargs.get('stylize', False))
        with torch.no_grad():
            normals_pred = self.norm_pred_header(feat_rgb)
            normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-06)
        if torch.any(torch.isnan(normals_pred)):
            None
        if torch.any(torch.isinf(normals_pred)):
            None
        with torch.no_grad():
            semantic = self.semantic_header(feat_rgb)
            semantic = self.semantic_act(semantic)
        d = F.normalize(d, p=2, dim=-1, eps=1e-06)
        d = self.dir_encoder((d + 1) / 2)
        with torch.set_grad_enabled(kwargs.get('stylize', False)):
            if self.embed_a:
                rgbs = self.rgb_net(torch.cat([d, feat_rgb, kwargs['embedding_a']], 1))
            else:
                rgbs = self.rgb_net(torch.cat([d, feat_rgb], 1))
        return sigmas, rgbs, normals_pred, semantic

    def forward_skybox(self, d):
        if not self.use_skybox:
            return None
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.skybox_dir_encoder((d + 1) / 2)
        rgbs = self.skybox_rgb_net(d)
        return rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades
        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32, device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,), device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
        return cells

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False, aux_model=None):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4, density_threshold)
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)
            if aux_model is not None:
                chunk_size = 2 ** 20
                for i in range(0, xyzs_w.shape[0], chunk_size):
                    density_grid_tmp[c, indices[i:i + chunk_size]] += aux_model.forward_test(xyzs_w[i:i + chunk_size], density_only=True, geometry_model=self)
        self.density_grid = torch.where(self.density_grid < 0, self.density_grid, torch.maximum(self.density_grid * decay, density_grid_tmp))
        mean_density = self.density_grid[self.density_grid > 0].mean().item()
        vren.packbits(self.density_grid, min(mean_density, density_threshold), self.density_bitfield)

    def uniform_sample(self, resolution=128):
        half_grid_size = self.scale / resolution
        samples = torch.stack(torch.meshgrid(torch.linspace(0, 1 - half_grid_size, resolution), torch.linspace(0, 1 - half_grid_size, resolution), torch.linspace(0, 1 - half_grid_size, resolution)), -1)
        dense_xyz = self.xyz_min * (1 - samples) + self.xyz_max * samples
        dense_xyz += half_grid_size * torch.rand_like(dense_xyz)
        density = self.density(dense_xyz.view(-1, 3))
        return density


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GIFSmoothing,
     lambda: ([], {'r': 4, 'eps': 4}),
     lambda: ([], {})),
    (PhotoWCT,
     lambda: ([], {}),
     lambda: ([], {})),
    (VGGEncoder,
     lambda: ([], {'level': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

