
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


import copy


import torch.nn as nn


import torch.nn.functional as F


def init_weights(m):
    None
    if type(m) == nn.Linear:
        None
    else:
        None


class MMTM(nn.Module):

    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)
            self.fc_visual.apply(init_weights)
            self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)
        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)
        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)
        return visual * vis_out, skeleton * sk_out


class MMTNet(nn.Module):

    def __init__(self, args):
        super(MMTNet, self).__init__()
        self.visual = None
        self.skeleton = None
        self.final_pred = None
        self.mmtm0 = MMTM(512, 128, 4)
        self.mmtm1 = MMTM(1024, 256, 4)
        self.mmtm2 = MMTM(2048, 512, 4)
        self.return_interm_feas = False
        self.return_both = False
        if hasattr(args, 'fc_final_preds') and args.fc_final_preds:
            self.final_pred = nn.Linear(args.num_classes * 2, args.num_classes)

    def get_mmtm_params(self):
        parameters = [{'params': self.mmtm0.parameters()}, {'params': self.mmtm1.parameters()}, {'params': self.mmtm2.parameters()}]
        return parameters

    def get_visual_params(self):
        parameters = [{'params': self.visual.parameters()}, {'params': self.mmtm0.parameters()}, {'params': self.mmtm1.parameters()}, {'params': self.mmtm2.parameters()}]
        return parameters

    def get_skeleton_params(self):
        parameters = [{'params': self.skeleton.parameters()}, {'params': self.mmtm0.parameters()}, {'params': self.mmtm1.parameters()}, {'params': self.mmtm2.parameters()}]
        return parameters

    def set_visual_skeleton_nets(self, visual, skeleton, return_interm_feas=False):
        self.visual = visual
        self.skeleton = skeleton
        self.return_interm_feas = return_interm_feas

    def set_return_both(self, p):
        self.return_both = p

    def forward(self, tensor_tuple):
        frames, skeleton = tensor_tuple[:2]
        N, C, T, V, M = skeleton.size()
        motion = skeleton[:, :, 1:, :, :] - skeleton[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.interpolate(motion, size=(T, V), mode='bilinear', align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)
        sk_hidden = []
        for i in range(self.skeleton.num_person):
            out1 = self.skeleton.conv1(skeleton[:, :, :, :, i])
            out2 = self.skeleton.conv2(out1)
            out2 = out2.permute(0, 3, 2, 1).contiguous()
            out3 = self.skeleton.conv3(out2)
            out_p = self.skeleton.conv4(out3)
            out1m = self.skeleton.conv1m(motion[:, :, :, :, i])
            out2m = self.skeleton.conv2m(out1m)
            out2m = out2m.permute(0, 3, 2, 1).contiguous()
            out3m = self.skeleton.conv3m(out2m)
            out_m = self.skeleton.conv4m(out3m)
            out4 = torch.cat((out_p, out_m), dim=1)
            sk_hidden.append([out1, out2, out3, out4])
        new_sk_hidden = []
        for h1, h2 in zip(sk_hidden[0], sk_hidden[1]):
            new_sk_hidden.append(torch.max(h1, h2))
        out4_p0 = sk_hidden[0][-1]
        out4_p1 = sk_hidden[1][-1]
        out5_p0 = self.skeleton.conv5(out4_p0)
        sk_hidden[0].append(out5_p0)
        out5_p1 = self.skeleton.conv5(out4_p1)
        sk_hidden[1].append(out5_p1)
        out5_max = torch.max(out5_p0, out5_p1)
        rgb_resnet = self.visual.cnn
        B, T, W, H, C = frames.size()
        frames = frames.view(B, 1, T, W, H, C)
        frames = frames.transpose(1, -1)
        frames = frames.view(B, C, T, W, H)
        frames = frames.contiguous()
        frames = transform_input(frames, rgb_resnet.input_dim, T=T)
        frames = rgb_resnet.conv1(frames)
        frames = rgb_resnet.bn1(frames)
        frames = rgb_resnet.relu(frames)
        frames = rgb_resnet.maxpool(frames)
        frames = transform_input(frames, rgb_resnet.layer1[0].input_dim, T=T)
        frames = rgb_resnet.layer1(frames)
        fm1 = frames
        frames = transform_input(frames, rgb_resnet.layer2[0].input_dim, T=T)
        frames = rgb_resnet.layer2(frames)
        fm2 = frames
        fm2, out5_p0 = self.mmtm0(fm2, out5_p0)
        out6_p0 = self.skeleton.conv6(out5_p0)
        sk_hidden[0].append(out6_p0)
        out6_p1 = self.skeleton.conv6(out5_p1)
        sk_hidden[1].append(out6_p0)
        out6_max = torch.max(out6_p0, out6_p1)
        out7 = out6_max
        frames = transform_input(frames, rgb_resnet.layer3[0].input_dim, T=T)
        frames = rgb_resnet.layer3(frames)
        fm3 = frames
        fm3, out7 = self.mmtm1(fm3, out7)
        out7 = out7.view(out7.size(0), -1)
        out8 = self.skeleton.fc7(out7)
        frames = transform_input(frames, rgb_resnet.layer4[0].input_dim, T=T)
        frames = rgb_resnet.layer4(frames)
        final_fm = transform_input(frames, rgb_resnet.out_dim, T=T)
        final_fm, out8 = self.mmtm2(final_fm, out8)
        outf = self.skeleton.fc8(out8)
        new_sk_hidden.append(out5_max)
        new_sk_hidden.append(out6_max)
        new_sk_hidden.append(out7)
        new_sk_hidden.append(out8)
        t = outf
        assert not torch.isnan(t).any()
        skeleton_features = [new_sk_hidden, outf]
        vis_out5 = self.visual.temporal_pooling(final_fm)
        vis_out6 = self.visual.classifier(vis_out5)
        visual_features = [fm1, fm2, fm3, final_fm, vis_out5, vis_out6]
        if self.return_interm_feas:
            return visual_features, skeleton_features
        vis_pred = vis_out6
        skeleton_pred = outf
        if self.final_pred is None:
            pred = (skeleton_pred + vis_pred) / 2
        else:
            pred = self.final_pred(torch.cat([skeleton_pred, vis_pred], dim=-1))
        if self.return_both:
            return vis_pred, skeleton_pred
        return pred


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MMTM,
     lambda: ([], {'dim_visual': 4, 'dim_skeleton': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

