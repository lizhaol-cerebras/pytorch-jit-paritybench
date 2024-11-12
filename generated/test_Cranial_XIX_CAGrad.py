
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


from torch.utils.data.dataset import Dataset


import torch


import torch.nn.functional as F


import numpy as np


import random


import torch.nn as nn


import torch.optim as optim


import torch.utils.data.sampler as sampler


import matplotlib.pyplot as plt


from scipy import stats as st


import time


from copy import deepcopy


from scipy.optimize import minimize


from scipy.optimize import Bounds


from scipy.optimize import minimize_scalar


from typing import Iterable


from typing import List


from typing import Optional


from typing import Tuple


import matplotlib


from matplotlib import cm


from matplotlib import ticker


from matplotlib.colors import LogNorm


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]), self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]), self.conv_layer([filter[i], filter[i]])))
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for j in range(1):
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))
        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
        if opt.task == 'semantic':
            self.pred_task = self.conv_layer([filter[0], self.class_nb], pred=True)
        if opt.task == 'depth':
            self.pred_task = self.conv_layer([filter[0], 1], pred=True)
        if opt.task == 'normal':
            self.pred_task = self.conv_layer([filter[0], 3], pred=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1), nn.BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1), nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0))
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0), nn.BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0), nn.BatchNorm2d(channel[2]), nn.Sigmoid())
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        for i in range(1):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = atten_encoder[i][j][0] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = atten_encoder[i][j][0] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1] * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1] * g_decoder[j][-1]
        if opt.task == 'semantic':
            pred = F.log_softmax(self.pred_task(atten_decoder[0][-1][-1]), dim=1)
        if opt.task == 'depth':
            pred = self.pred_task(atten_decoder[0][-1][-1])
        if opt.task == 'normal':
            pred = self.pred_task(atten_decoder[0][-1][-1])
            pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        return pred


class SegNetSplit(nn.Module):

    def __init__(self):
        super(SegNetSplit, self).__init__()
        if opt.type == 'wide':
            filter = [64, 128, 256, 512, 1024]
        else:
            filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]), self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]), self.conv_layer([filter[i], filter[i]])))
        self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))
        self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel):
        if opt.type == 'deep':
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1), nn.BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True), nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1), nn.BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1), nn.BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        t1_pred = F.log_softmax(self.pred_task1(g_decoder[i][1]), dim=1)
        t2_pred = self.pred_task2(g_decoder[i][1])
        t3_pred = self.pred_task3(g_decoder[i][1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        return [t1_pred, t2_pred, t3_pred], self.logsigma


LOWER = 5e-06


class Toy(nn.Module):

    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([[-3.0, 0], [3.0, 0]])

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]
        f1 = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)
        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)
        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2
        f = torch.tensor([f1, f2])
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)
        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)
        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2
        f = torch.cat([f1.view(-1, 1), f2.view(-1, 1)], -1)
        return f


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Toy,
     lambda: ([], {}),
     lambda: ([torch.rand([4])], {})),
]

