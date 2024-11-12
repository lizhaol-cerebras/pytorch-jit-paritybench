
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


import time


import torch


import numpy as np


from torch import optim


import random


import torch.nn.functional as F


from torch.utils.data import Dataset


import warnings


import torchvision


import pandas as pd


import numbers


from functools import partial


from itertools import repeat


from typing import Callable


import torch.utils.data


import torch.nn as nn


import math


from torch import nn


import torch.fft


from numpy import *


from numpy.linalg import *


from scipy.special import factorial


from functools import reduce


from itertools import accumulate


import logging


from collections import defaultdict


from collections import OrderedDict


from typing import Tuple


from torch import distributed as dist


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import DataLoader


from torch.nn.modules import GroupNorm


from torch.nn.modules.batchnorm import _BatchNorm


import functools


from copy import deepcopy


from typing import Iterable


from torch.autograd import Variable


from torch.optim.optimizer import Optimizer


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn

    def forward(self, img1, img2):
        img1 = lpips.im2tensor(img1 * 255)
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1, img2
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
        else:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM_Model(nn.Module):
    """ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


class tf_Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super(tf_Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode='nearest')


class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, window_length, height, width, filter_size, stride, layer_norm):
        super(Eidetic3DLSTMCell, self).__init__()
        self._norm_c_t = nn.LayerNorm([num_hidden, window_length, height, width])
        self.num_hidden = num_hidden
        self.padding = 0, filter_size[1] // 2, filter_size[2] // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 7, window_length, height, width]))
            self.conv_h = nn.Sequential(tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, window_length, height, width]))
            self.conv_gm = nn.Sequential(tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, window_length, height, width]))
            self.conv_new_cell = nn.Sequential(tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, window_length, height, width]))
            self.conv_new_gm = nn.Sequential(tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, window_length, height, width]))
        else:
            self.conv_x = nn.Sequential(tf_Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h = nn.Sequential(tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_gm = nn.Sequential(tf_Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_new_cell = nn.Sequential(tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_new_gm = nn.Sequential(tf_Conv3d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = tf_Conv3d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def _attn(self, in_query, in_keys, in_values):
        batch, num_channels, _, width, height = in_query.shape
        query = in_query.reshape(batch, -1, num_channels)
        keys = in_keys.reshape(batch, -1, num_channels)
        values = in_values.reshape(batch, -1, num_channels)
        attn = torch.einsum('bxc,byc->bxy', query, keys)
        attn = torch.softmax(attn, dim=2)
        attn = torch.einsum('bxy,byc->bxc', attn, values)
        return attn.reshape(batch, num_channels, -1, width, height)

    def forward(self, x_t, h_t, c_t, global_memory, eidetic_cell):
        h_concat = self.conv_h(h_t)
        i_h, g_h, r_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        x_concat = self.conv_x(x_t)
        i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)
        new_cell = c_t + self._attn(r_t, eidetic_cell, eidetic_cell)
        new_cell = self._norm_c_t(new_cell) + i_t * g_t
        new_global_memory = self.conv_gm(global_memory)
        i_m, f_m, g_m, m_m = torch.split(new_global_memory, self.num_hidden, dim=1)
        temp_i_t = torch.sigmoid(temp_i_x + i_m)
        temp_f_t = torch.sigmoid(temp_f_x + f_m + self._forget_bias)
        temp_g_t = torch.tanh(temp_g_x + g_m)
        new_global_memory = temp_f_t * torch.tanh(m_m) + temp_i_t * temp_g_t
        o_c = self.conv_new_cell(new_cell)
        o_m = self.conv_new_gm(new_global_memory)
        output_gate = torch.tanh(o_x + o_h + o_c + o_m)
        memory = torch.cat((new_cell, new_global_memory), 1)
        memory = self.conv_last(memory)
        output = torch.tanh(memory) * torch.sigmoid(output_gate)
        return output, new_cell, global_memory


class E3DLSTM_Model(nn.Module):
    """E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(E3DLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.window_length = 2
        self.window_stride = 1
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(Eidetic3DLSTMCell(in_channel, num_hidden[i], self.window_length, height, width, (2, 5, 5), configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=(self.window_length, 1, 1), stride=(self.window_length, 1, 1), padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []
        for t in range(self.window_length - 1):
            input_list.append(torch.zeros_like(frames[:, 0]))
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.window_length, height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], self.window_length, height, width], device=device)
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            input_list.append(net)
            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0)
                net = net.permute(1, 2, 0, 3, 4).contiguous()
            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                input = net if i == 0 else h_t[i - 1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])
            x_gen = self.conv_last(h_t[self.num_layers - 1]).squeeze(2)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.L1_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


class MAUCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding), nn.LayerNorm([3 * num_hidden, height, width]))
        self.conv_t_next = nn.Sequential(nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding), nn.LayerNorm([num_hidden, height, width]))
        self.conv_s = nn.Sequential(nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding), nn.LayerNorm([3 * num_hidden, height, width]))
        self.conv_s_next = nn.Sequential(nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding), nn.LayerNorm([num_hidden, height, width]))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend
        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        return T_new, S_new


class MAU_Model(nn.Module):
    """MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MAU_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []
        width = W // configs.patch_size // configs.sr_size
        height = H // configs.patch_size // configs.sr_size
        self.MSE_criterion = nn.MSELoss()
        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, self.tau, self.cell_mode))
        self.cell_list = nn.ModuleList(cell_list)
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1), module=nn.Conv2d(in_channels=self.frame_channel, out_channels=self.num_hidden[0], stride=1, padding=0, kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1), module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i), module=nn.Conv2d(in_channels=self.num_hidden[0], out_channels=self.num_hidden[0], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3)))
            encoder.add_module(name='encoder_t_relu{0}'.format(i), module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        decoders = []
        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i), module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1)))
            decoder.add_module(name='c_decoder_relu{0}'.format(i), module=nn.LeakyReLU(0.2))
            decoders.append(decoder)
        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1), module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1)))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        self.srcnn = nn.Sequential(nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0))
        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_channel, height, width]))
                tmp_s.append(torch.zeros([batch_size, in_channel, height, width]))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)
        for t in range(self.configs.total_length - 1):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width])
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t
            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]
            x_gen = self.srcnn(out)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames[:, 1:])
        else:
            loss = None
        return next_frames, loss


class MIMBlock(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMBlock, self).__init__()
        self.convlstm_c = None
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))
        if layer_norm:
            self.conv_t_cc = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_s_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_x_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_h_concat = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_x_concat = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
        else:
            self.conv_t_cc = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_s_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_x_cc = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h_concat = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_x_concat = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def MIMS(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)
        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h
        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)
        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x, diff_h, h, c, m):
        h = self._init_state(x) if h is None else h
        c = self._init_state(x) if c is None else c
        m = self._init_state(x) if m is None else m
        diff_h = self._init_state(x) if diff_h is None else diff_h
        t_cc = self.conv_t_cc(h)
        s_cc = self.conv_s_cc(m)
        x_cc = self.conv_x_cc(x)
        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)
        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        c, self.convlstm_c = self.MIMS(diff_h, c, self.convlstm_c if self.convlstm_c is None else self.convlstm_c.detach())
        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        new_h = o * torch.tanh(self.conv_last(cell))
        return new_h, new_c, new_m


class MIMN(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(MIMN, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.ct_weight = nn.Parameter(torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))
        if layer_norm:
            self.conv_h_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_x_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
        else:
            self.conv_h_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_x_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)
        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        ct_activation = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, dim=1)
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h
        if x != None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)
        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
        return h_new, c_new


class SpatioTemporalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 7, height, width]))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
        else:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class MIM_Model(nn.Module):
    """MIM Model

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MIM_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        stlstm_layer, stlstm_layer_diff = [], []
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            if i < 1:
                stlstm_layer.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
            else:
                stlstm_layer.append(MIMBlock(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        for i in range(num_layers - 1):
            stlstm_layer_diff.append(MIMN(num_hidden[i], num_hidden[i + 1], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.stlstm_layer = nn.ModuleList(stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(stlstm_layer_diff)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        hidden_state_diff = []
        cell_state_diff = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)
        st_memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            preh = h_t[0]
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](net, h_t[0], c_t[0], st_memory)
            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](h_t[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)
                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](h_t[i - 1], hidden_state_diff[i - 1], h_t[i], c_t[i], st_memory)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


def sim_matrix_postprocess(similar_matrix):
    B, T, hw1, hw2 = similar_matrix.shape
    similar_matrix = similar_matrix.reshape(similar_matrix.shape[0], similar_matrix.shape[1], -1)
    similar_matrix = F.softmax(similar_matrix, dim=-1)
    return similar_matrix.reshape(B, T, hw1, hw2)


def cum_multiply(value_seq, cum_softmax=False, reverse=True):
    """

    :param value_seq: (B,S,***), B - batch num; S- sequence len
    :return: output value_seq: (B,S,***)
    """
    if not reverse:
        value_seq = torch.flip(value_seq, dims=[1])
    B, T, hw, hw = value_seq.shape
    new_output = value_seq.clone()
    for i in range(value_seq.shape[1] - 2, -1, -1):
        cur_sim = new_output[:, i].reshape(B, hw, hw).clone()
        next_sim = new_output[:, i + 1].reshape(B, hw, hw).clone()
        new_output[:, i] = torch.bmm(cur_sim, next_sim).reshape(B, hw, hw)
    if not reverse:
        new_output = torch.flip(new_output, dims=[1])
    if cum_softmax:
        new_output = sim_matrix_postprocess(new_output)
    return new_output


class Compose(nn.Module):

    def __init__(self, downsample_scale, mat_size, prev_len, aft_seq_length):
        super(Compose, self).__init__()
        self.downsample_scale = downsample_scale
        self.mat_size = mat_size
        self.prev_len = prev_len
        self.aft_seq_length = aft_seq_length
        self.feat_shuffle = []
        self.feat_unshuffle = []
        self.feat_scale_list = []
        for i in range(len(self.downsample_scale) - 1):
            feat_shuffle_scale = 1
            for s in range(len(self.downsample_scale) - 2, i - 1, -1):
                feat_shuffle_scale *= self.downsample_scale[s]
            self.feat_scale_list.append(feat_shuffle_scale)
            self.feat_shuffle.append(nn.PixelShuffle(feat_shuffle_scale))
            self.feat_unshuffle.append(nn.PixelUnshuffle(feat_shuffle_scale))
        self.feat_shuffle = nn.ModuleList(self.feat_shuffle)
        self.feat_unshuffle = nn.ModuleList(self.feat_unshuffle)

    def feat_generator(self, feats, sim_matrix, feat_idx, img_compose=False, scale=1):
        """

        :param feats: [B,T,c,h,w]
        :param sim_matrix: [B,T,h*w,h*w]
        :return: new_feats: [B,c,h,w]
        """
        B, T, c, h, w = feats.shape
        if scale > 1:
            feats = feats[:, -1:]
            sim_matrix = sim_matrix[:, -1:]
            T = 1
        feats = feats.permute(0, 2, 1, 3, 4)
        feats = feats.reshape(B, c, T * h * w).permute(0, 2, 1)
        B, T, hw_cur, hw_target = sim_matrix.shape
        sim_matrix = sim_matrix.reshape(B, T * hw_cur, hw_target).permute(0, 2, 1)
        weight = torch.sum(sim_matrix, dim=-1).reshape(-1, 1, hw_target) + 1e-06
        new_feats = torch.bmm(sim_matrix, feats).permute(0, 2, 1) / weight
        new_feats = new_feats.reshape(B, c, h * scale, w * scale)
        return new_feats

    def feat_compose(self, emb_feat_list, sim_matrix, img_compose=False, scale=1, use_gt=False):
        """

        :param emb_feat_list: (scale_num, (B,T,c,h,w))
        :param sim_matrix:  (B,T-1,h,w,h,w)
        :param use_gt_sim_matrix: bool
        :return: fut_emb_feat_list (scale_num, (B,t,c,h,w))
        """
        fut_emb_feat_list = []
        ori_emb_feat_list = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                fut_emb_feat_list.append(None)
                ori_emb_feat_list.append(None)
                continue
            fut_emb_feat_list.append([])
            cur_emb_feat = emb_feat_list[i]
            ori_emb_feat_list.append(torch.mean(emb_feat_list[i], dim=1))
            sim_matrix_seq = sim_matrix[i]
            B = sim_matrix_seq.shape[0]
            N, c, h, w = cur_emb_feat.shape
            cur_emb_feat = cur_emb_feat.reshape(B, -1, c, h, w)
            cur_emb_feat = cur_emb_feat[:, :self.prev_len] if not use_gt else cur_emb_feat.clone()
            for t in range(self.aft_seq_length):
                active_matrix_seq = sim_matrix_seq[:, :self.prev_len - 1]
                if t > 0:
                    fut_t_matrix = sim_matrix_seq[:, self.prev_len + t - 1:self.prev_len + t]
                else:
                    fut_t_matrix = sim_matrix_seq[:, self.prev_len - 1:self.prev_len + t]
                active_matrix_seq = torch.cat([active_matrix_seq, fut_t_matrix], dim=1)
                cur_sim_matrix = cum_multiply(active_matrix_seq.clone())
                composed_t_feats = self.feat_generator(cur_emb_feat[:, :self.prev_len].clone(), cur_sim_matrix, feat_idx=i, img_compose=img_compose, scale=scale)
                fut_emb_feat_list[i].append(composed_t_feats.clone())
                if not use_gt:
                    if scale == 1:
                        if cur_emb_feat.shape[1] > self.prev_len + t:
                            cur_emb_feat[:, t + self.prev_len] = composed_t_feats.clone()
                        else:
                            cur_emb_feat = torch.cat([cur_emb_feat, composed_t_feats.clone().unsqueeze(1)], dim=1)
            temp = torch.stack(fut_emb_feat_list[i], dim=1)
            fut_emb_feat_list[i] = temp.reshape(-1, c, h * scale, w * scale)
        return fut_emb_feat_list, ori_emb_feat_list

    def forward(self, x, similar_matrix, feat_shape):
        compose_feat_list = []
        similar_matrix_for_compose = []
        for i in range(len(x)):
            if x[i] is None:
                compose_feat_list.append(None)
                similar_matrix_for_compose.append(None)
                continue
            if i < len(x) - 2:
                h, w = x[i].shape[-2:]
                target_size = h // self.feat_scale_list[i] * self.feat_scale_list[i], w // self.feat_scale_list[i] * self.feat_scale_list[i]
                cur_feat = self.feat_unshuffle[i](F.interpolate(x[i].clone(), size=target_size, mode='bilinear'))
                if cur_feat.shape[-2] != self.mat_size[0][-2] or cur_feat.shape[-1] != self.mat_size[0][-1]:
                    compose_feat_list.append(F.interpolate(cur_feat, size=tuple(self.mat_size[0]), mode='bilinear'))
                else:
                    compose_feat_list.append(cur_feat.clone())
                similar_matrix_for_compose.append(similar_matrix[0])
            elif x[i].shape[-2] != self.mat_size[i - len(x) + 2][-2] or x[i].shape[-1] != self.mat_size[i - len(x) + 2][-1]:
                compose_feat_list.append(F.interpolate(x[i], size=tuple(self.mat_size[i - len(x) + 2]), mode='bilinear'))
            else:
                compose_feat_list.append(x[i])
        similar_matrix_for_compose.append(similar_matrix[0])
        similar_matrix_for_compose.append(similar_matrix[1])
        compose_fut_feat_list, _ = self.feat_compose(compose_feat_list, similar_matrix_for_compose)
        for i in range(len(compose_fut_feat_list)):
            if compose_fut_feat_list[i] is None:
                continue
            if i < len(x) - 2:
                compose_fut_feat_list[i] = self.feat_shuffle[i](compose_fut_feat_list[i])
            if compose_fut_feat_list[i].shape[-2] != feat_shape[i][-2] or compose_fut_feat_list[i].shape[-1] != feat_shape[i][-1]:
                compose_fut_feat_list[i] = F.interpolate(compose_fut_feat_list[i], size=tuple(feat_shape[i]), mode='bilinear')
        return compose_fut_feat_list


class ResidualDenseBlock_4C(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_4C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf):
        super(RRDB, self).__init__()
        gc = nf // 2
        self.RDB1 = ResidualDenseBlock_4C(nf, gc)
        self.RDB2 = ResidualDenseBlock_4C(nf, gc)
        self.RDB3 = ResidualDenseBlock_4C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ImageEnhancer(nn.Module):

    def __init__(self, C_in=1, hid_S=32, downsample_scale=[2, 2, 2], rrdb_enhance_num=2):
        super(ImageEnhancer, self).__init__()
        self.C_in = C_in
        layers = [nn.Conv2d(C_in * 2, hid_S, 3, 1, 1)]
        for i in range(rrdb_enhance_num):
            layers.append(RRDB(hid_S))
        self.model = nn.Sequential(*layers)
        self.outconv = nn.Conv2d(hid_S, C_in, kernel_size=1)

    def forward(self, x):
        feat = self.model(x)
        out = self.outconv(feat)
        return out


class Conv3D(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.leaky_relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class MatrixPredictor3DConv(nn.Module):

    def __init__(self, hidden_len=64):
        super(MatrixPredictor3DConv, self).__init__()
        self.unet_base = hidden_len
        self.hidden_len = hidden_len
        self.conv_pre_1 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv3d_1 = Conv3D(self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = Conv3D(self.unet_base * 2, self.unet_base * 2, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
        self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
        self.bn1_1 = nn.BatchNorm2d(self.unet_base)
        self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
        self.bn3_1 = nn.BatchNorm2d(self.unet_base)
        self.bn4_1 = nn.BatchNorm2d(self.hidden_len)

    def forward(self, x):
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x)))
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()
        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, scale_factor=(2, 2)))))
        return x


class SimpleMatrixPredictor3DConv_direct(nn.Module):

    def __init__(self, T, hidden_len=64, image_pred=False, aft_seq_length=10):
        super(SimpleMatrixPredictor3DConv_direct, self).__init__()
        self.unet_base = hidden_len
        self.hidden_len = hidden_len
        self.conv_pre_1 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.fut_len = aft_seq_length
        self.conv3d_1 = Conv3D(self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        if self.fut_len > 1:
            self.temporal_layer = Conv3D(self.unet_base * 2, self.unet_base * 2, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        else:
            self.temporal_layer = nn.Sequential(nn.Conv2d(self.unet_base * 2, self.unet_base * 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        input_len = T if image_pred else T - 1
        self.conv_translate = nn.Sequential(nn.Conv2d(self.unet_base * input_len, self.unet_base * self.fut_len, kernel_size=1, stride=1, padding=0), nn.LeakyReLU())
        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
        self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
        self.bn1_1 = nn.BatchNorm2d(self.unet_base)
        self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
        self.bn3_1 = nn.BatchNorm2d(self.unet_base)
        self.bn4_1 = nn.BatchNorm2d(self.hidden_len)
        self.bn_translate = nn.BatchNorm2d(self.unet_base * self.fut_len)

    def forward(self, x):
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x)))
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()
        x_1 = self.conv3d_1(x_1)
        batch, seq, c, h, w = x_1.shape
        x_tmp = x_1.reshape(batch, -1, h, w)
        x_tmp = self.bn_translate(self.conv_translate(x_tmp))
        x_1 = x_tmp.reshape(batch, self.fut_len, c, h, w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1)))
        if self.fut_len > 1:
            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()
            x_2 = self.temporal_layer(x_2)
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()
        else:
            x_2 = self.temporal_layer(x_2)
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.reshape(-1, x_1.size(2), x_1.size(3), x_1.size(4))
        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, size=x_1.shape[2:]), x_1), dim=1))))
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, size=x.shape[3:]))))
        return x


class PredictModel(nn.Module):

    def __init__(self, T, hidden_len=32, aft_seq_length=10, mx_h=32, mx_w=32, use_direct_predictor=True):
        super(PredictModel, self).__init__()
        self.mx_h = mx_h
        self.mx_w = mx_w
        self.hidden_len = hidden_len
        self.fut_len = aft_seq_length
        self.conv1 = nn.Conv2d(1, hidden_len, kernel_size=3, padding=1, bias=False)
        self.fuse_conv = nn.Conv2d(hidden_len * 2, hidden_len, kernel_size=3, padding=1, bias=False)
        if use_direct_predictor:
            self.predictor = SimpleMatrixPredictor3DConv_direct(T=T, hidden_len=hidden_len, aft_seq_length=aft_seq_length)
        else:
            self.predictor = MatrixPredictor3DConv(hidden_len)
        self.out_conv = nn.Conv2d(hidden_len, 1, kernel_size=3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def res_interpolate(self, in_tensor, template_tensor):
        """
        in_tensor: batch,c,h'w',H'W'
        tempolate_tensor: batch,c,hw,HW
        out_tensor: batch,c,hw,HW
        """
        out_tensor = F.interpolate(in_tensor, template_tensor.shape[-2:])
        return out_tensor

    def forward(self, matrix_seq, softmax=False, res=None):
        B, T, hw, window_size = matrix_seq.size()
        matrix_seq = matrix_seq.reshape(-1, hw, self.mx_h, self.mx_w)
        matrix_seq = matrix_seq.reshape(B * T * hw, self.mx_h, self.mx_w).unsqueeze(1)
        x = self.conv1(matrix_seq)
        x = x.reshape(B, T, hw, -1, self.mx_h, self.mx_w)
        x = x.permute(0, 2, 1, 3, 4, 5).reshape(B * hw, T, -1, self.mx_h, self.mx_w)
        emb = self.predictor(x)
        emb = emb.reshape(B * hw * self.fut_len, -1, self.mx_h, self.mx_w)
        res_emb = emb.clone()
        if res is not None:
            template = emb.clone().reshape(B, hw, emb.shape[1], -1).permute(0, 2, 1, 3)
            in_tensor = res.clone().reshape(B, hw // 4, emb.shape[1], -1).permute(0, 2, 1, 3)
            res_tensor = self.res_interpolate(in_tensor, template).permute(0, 2, 1, 3).reshape(emb.shape)
            emb = self.fuse_conv(torch.cat([emb, res_tensor], dim=1))
        out = self.out_conv(emb)
        out = out.reshape(B, hw, -1, self.mx_h, self.mx_w)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(B, -1, hw, window_size)
        if softmax:
            out = out.view(B, out.shape[1], -1)
            out = self.softmax(out)
            out = out.reshape(B, -1, hw, window_size)
        return out, res_emb


class ConvLayer(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=True, motion=False, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) if motion else nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, skip=True, scale=2, bn=True, motion=False):
        super().__init__()
        factor = scale
        if bilinear:
            if skip:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(in_channels, out_channels, bn=bn)
            else:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(in_channels, out_channels)
        elif skip:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.conv = ConvLayer(out_channels * 2, out_channels, bn=bn, motion=motion)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.conv = ConvLayer(out_channels, out_channels, bn=bn, motion=motion)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is None:
            return self.conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def cut_off_process(similarity_matrix, thre, sigmoid=False, k=-1):
    B = similarity_matrix.shape[0]
    T_prime = similarity_matrix.shape[1]
    hw = similarity_matrix.shape[2]
    new_similarity_matrix = similarity_matrix.clone()
    """
    diagonal_mask = torch.zeros_like(new_similarity_matrix[0,0]).to(similarity_matrix.device).bool() #(h*w,h*w)
    diagonal_mask.fill_diagonal_(True)
    diagonal_mask = diagonal_mask.reshape(1,1,hw,hw).repeat(B,T_prime,1,1)
    new_similarity_matrix[diagonal_mask] = 0.
    """
    if sigmoid:
        new_similarity_matrix[new_similarity_matrix < 0] = 0.0
        new_similarity_matrix = F.sigmoid(new_similarity_matrix)
    elif k > -1:
        new_similarity_matrix[new_similarity_matrix < 0.0] = 0.0
        select_num = int(new_similarity_matrix.shape[-1] * k)
        top_k, _ = torch.topk(new_similarity_matrix, select_num, dim=-1)
        thre_value = top_k[:, :, :, -1:]
        new_similarity_matrix[new_similarity_matrix < thre_value] = 0.0
    else:
        new_similarity_matrix[new_similarity_matrix < thre] = 0.0
    return new_similarity_matrix


def build_similarity_matrix(emb_feats, thre=-1, sigmoid=False, k=-1, cut_off=False):
    """

    :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)
    :return: similarity matrix (N, T-1, h*w, h*w) current frame --> next frame
    """
    B, T, c, h, w = emb_feats.shape
    emb_feats = emb_feats.permute(0, 1, 3, 4, 2)
    normalize_feats = emb_feats / (torch.norm(emb_feats, dim=-1, keepdim=True) + 1e-06)
    prev_frame = normalize_feats[:, :T - 1].reshape(-1, h * w, c)
    next_frame = normalize_feats[:, 1:].reshape(-1, h * w, c)
    similarity_matrix = torch.bmm(prev_frame, next_frame.permute(0, 2, 1)).reshape(B, T - 1, h * w, h * w)
    if cut_off:
        similarity_matrix = cut_off_process(similarity_matrix, thre, sigmoid, k)
    return similarity_matrix


def sim_matrix_interpolate(in_matrix, ori_hw, target_hw):
    ori_h, ori_w = ori_hw[0], ori_hw[1]
    target_h, target_w = target_hw[0], target_hw[1]
    B, T, hw, hw = in_matrix.shape
    ori_matrix = in_matrix.clone().reshape(B, T, ori_h, ori_w, ori_h, ori_w)
    ori_matrix_half = F.interpolate(ori_matrix.reshape(-1, ori_h, ori_w).unsqueeze(1), (int(target_h), int(target_w)), mode='bilinear').squeeze(1)
    new_matrix = F.interpolate(ori_matrix_half.reshape(B, T, ori_h, ori_w, target_h, target_w).permute(0, 1, 4, 5, 2, 3).reshape(-1, ori_h, ori_w).unsqueeze(1), (int(target_h), int(target_w)), mode='bicubic').squeeze(1)
    new_matrix = new_matrix.reshape(B, T, target_h, target_w, target_h, target_w).permute(0, 1, 4, 5, 2, 3).reshape(B, T, target_h * target_w, target_h * target_w)
    return new_matrix


class MidMotionMatrix(nn.Module):

    def __init__(self, T, hid_S=32, hid_T=192, mat_size=[[8, 8], [4, 4]], aft_seq_length=10, use_direct_predictor=True):
        super(MidMotionMatrix, self).__init__()
        self.pre_seq_len = T
        self.mat_size = mat_size
        self.mx_h = mat_size[0][0]
        self.mx_w = mat_size[0][1]
        self.scale_fuser_1 = Up(hid_S * 2, hid_S, bilinear=False, scale=2)
        self.scale_fuser_2 = nn.Sequential(nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S), nn.LeakyReLU(), nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S), nn.LeakyReLU())
        self.predictor = PredictModel(T=T, hidden_len=hid_T, aft_seq_length=aft_seq_length, mx_h=self.mx_h, mx_w=self.mx_w, use_direct_predictor=use_direct_predictor)

    def forward(self, x, B, T):
        similar_matrix = []
        prev_sim_matrix = []
        pred_sim_matrix = [None, None]
        for i in [-2, -1]:
            N = x[i].shape[0]
            h, w = x[i].shape[2:]
            cur_sim_matrix = build_similarity_matrix(x[i].reshape(B, T, -1, h, w))
            prev_sim_matrix.append(cur_sim_matrix[:, :self.pre_seq_len - 1].clone())
        pred_fut_matrix, _ = self.predictor(prev_sim_matrix[0], softmax=False, res=None)
        pred_sim_matrix[0] = pred_fut_matrix.clone()
        pred_sim_matrix[1] = sim_matrix_interpolate(pred_fut_matrix.clone(), self.mat_size[0], self.mat_size[1])
        pred_sim_matrix[0] = sim_matrix_postprocess(pred_sim_matrix[0])
        pred_sim_matrix[1] = sim_matrix_postprocess(pred_sim_matrix[1])
        for i in range(len(prev_sim_matrix)):
            new_cur_sim_matrix = torch.cat([sim_matrix_postprocess(prev_sim_matrix[i]), pred_sim_matrix[i]], dim=1)
            similar_matrix.append(new_cur_sim_matrix)
        return similar_matrix


class RRDBDecoder(nn.Module):

    def __init__(self, C=1, hid_S=32, downsample_scale=[2, 2, 2], rrdb_decoder_num=2, scale_in_use=3):
        super(RRDBDecoder, self).__init__()
        self.scale_num = len(downsample_scale)
        out_channel = C * 4
        self.upsample_block_low2high = Up(in_channels=hid_S * 2 ** self.scale_num, out_channels=hid_S * 2 ** (self.scale_num - 1), bilinear=False, scale=downsample_scale[-1])
        upsample_block_list = []
        for i in range(self.scale_num - 2, -1, -1):
            skip = False if i < self.scale_num - 1 and scale_in_use == 2 or i < self.scale_num - 2 and scale_in_use == 3 else True
            upsample_block_list.append(Up(in_channels=hid_S * 2 ** (i + 1), out_channels=hid_S * 2 ** i, bilinear=False, scale=downsample_scale[i], skip=skip))
        self.upsample_block = nn.ModuleList(upsample_block_list)
        self.rrdb_block = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_decoder_num)])
        self.outc = nn.Conv2d(hid_S, out_channel, kernel_size=1)

    def forward(self, in_feat):
        x = self.upsample_block_low2high(in_feat[-1], in_feat[-2])
        for i in range(self.scale_num - 1):
            x = self.upsample_block[i](x, in_feat[-i - 3])
        x = self.rrdb_block(x)
        logits = self.outc(x)
        return logits


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, upsample=False, skip=False, factor=2, motion=False):
        super().__init__()
        self.upsample = upsample
        self.maxpool = None
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if factor == 4:
                self.maxpool = nn.MaxPool2d(2)
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            if motion:
                self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        input = input + shortcut
        if self.maxpool is not None:
            input = self.maxpool(input)
        return nn.LeakyReLU()(input)


class RRDBEncoder(nn.Module):

    def __init__(self, C=1, hid_S=32, downsample_ratio=[2, 2, 2], rrdb_encoder_num=2, scale_in_use=3):
        super(RRDBEncoder, self).__init__()
        self.C_in = C * 4
        self.hid_S = hid_S
        self.scale_num = len(downsample_ratio)
        self.downsample_ratio = downsample_ratio
        self.scale_in_use = scale_in_use
        self.inconv = nn.Conv2d(self.C_in, self.hid_S, 3, 1, 1)
        self.block_rrdb = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_encoder_num)])
        pre_downsample_block_list = []
        for i in range(self.scale_num - 2):
            pre_downsample_block_list.append(ResBlock(hid_S * 2 ** i, hid_S * 2 ** (i + 1), downsample=True, factor=downsample_ratio[i]))
        self.pre_downsample_block = nn.ModuleList(pre_downsample_block_list)
        self.downsample_high = ResBlock(hid_S * 2 ** (self.scale_num - 2), hid_S * 2 ** (self.scale_num - 1), downsample=True, factor=downsample_ratio[-2])
        self.downsample_low = ResBlock(hid_S * 2 ** (self.scale_num - 1), hid_S * 2 ** self.scale_num, downsample=True, factor=downsample_ratio[-1])

    def forward(self, x, save_all=False):
        in_feat = []
        x = self.inconv(x)
        x = self.block_rrdb(x)
        in_feat.append(x)
        for i in range(self.scale_num - 2):
            x = self.pre_downsample_block[i](x)
            in_feat.append(x)
        x = self.downsample_high(x)
        in_feat.append(x)
        x = self.downsample_low(x)
        in_feat.append(x)
        if self.scale_in_use == 3:
            for i in range(len(in_feat) - 3):
                in_feat[i] = None
        elif self.scale_in_use == 2:
            for i in range(len(in_feat) - 2):
                in_feat[i] = None
        return in_feat


class filter_block(nn.Module):

    def __init__(self, downsample_scale, hid_S, mat_size):
        super(filter_block, self).__init__()
        self.filter_block = []
        high_scale = len(downsample_scale) - 1
        feat_len = hid_S * 2 ** high_scale
        self.mat_size = mat_size
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S), nn.LeakyReLU(), nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S), nn.LeakyReLU(), nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S), nn.LeakyReLU()))
        low_scale = high_scale + 1
        feat_len = hid_S * 2 ** low_scale
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S * 2, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S * 2), nn.LeakyReLU(), nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S * 2), nn.LeakyReLU(), nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1), nn.BatchNorm2d(hid_S * 2), nn.LeakyReLU()))
        self.filter_block = nn.ModuleList(self.filter_block)

    def forward(self, x):
        gi = []
        for s in [-2, -1]:
            feat_area = x[s].shape[-1] * x[s].shape[-2]
            mat_area = self.mat_size[s][-1] * self.mat_size[s][-2]
            if mat_area != feat_area:
                out = F.interpolate(x[s].clone(), size=tuple(self.mat_size[s]), mode='bilinear')
            else:
                out = x[s].clone()
            out = self.filter_block[s](out)
            gi.append(out)
        return gi


class MMVP_Model(nn.Module):
    """MMVP

    Implementation of `MMVP: Motion-Matrix-based Video Prediction
    <https://arxiv.org/abs/2308.16154>`_.

    """

    def __init__(self, in_shape, aft_seq_length=10, hid_S=32, hid_T=192, rrdb_encoder_num=2, rrdb_decoder_num=2, rrdb_enhance_num=2, downsample_setting='2,2,2', shuffle_setting=True, use_direct_predictor=True, **kwargs):
        super(MMVP_Model, self).__init__()
        T, C, H, W = in_shape
        downsample_ratio = [int(x) for x in downsample_setting.split(',')]
        highres_scale = np.prod(downsample_ratio[:-1]) * 2
        lowres_scale = np.prod(downsample_ratio) * 2
        self.pre_seq_length = T
        self.mat_size = [[H // highres_scale, W // highres_scale], [H // lowres_scale, W // lowres_scale]]
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)
        self.enc = RRDBEncoder(C=C, hid_S=hid_S, rrdb_encoder_num=rrdb_encoder_num, downsample_ratio=downsample_ratio)
        self.filter = filter_block(downsample_scale=downsample_ratio, hid_S=hid_S, mat_size=self.mat_size)
        self.dec = RRDBDecoder(C=C, hid_S=hid_S, rrdb_decoder_num=rrdb_decoder_num, downsample_scale=downsample_ratio)
        self.fuse = Compose(downsample_scale=downsample_ratio, mat_size=self.mat_size, prev_len=T, aft_seq_length=aft_seq_length)
        self.hid = MidMotionMatrix(T=T, hid_S=hid_S, hid_T=hid_T, mat_size=self.mat_size, aft_seq_length=aft_seq_length, use_direct_predictor=use_direct_predictor)
        res_shuffle_scale = 1
        for s in range(len(downsample_ratio) - 1):
            res_shuffle_scale *= downsample_ratio[s]
        self.res_shuffle = nn.PixelShuffle(res_shuffle_scale)
        self.res_unshuffle = nn.PixelUnshuffle(res_shuffle_scale)
        self.res_shuffle_scale = res_shuffle_scale
        self.enhance = ImageEnhancer(C_in=C, hid_S=hid_S, downsample_scale=downsample_ratio, rrdb_enhance_num=rrdb_enhance_num)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x_raw = x_raw.reshape(B * T, C, H, W)
        x_raw = self.unshuffle(x_raw)
        x = x_raw.clone()
        x_wh = x.shape[-2:]
        fi = self.enc(x)
        feat_shape = []
        for i in range(len(fi)):
            if fi[i] is None:
                feat_shape.append(None)
            else:
                feat_shape.append(fi[i].shape[2:])
        gi = self.filter(fi)
        similarity_matrix = self.hid(gi, B, T)
        composed_fut_feat = self.fuse(fi, similarity_matrix, feat_shape)
        recon_img = self.dec(composed_fut_feat)
        final_recon_img = recon_img.clone()
        if x_wh != recon_img.shape[2:]:
            std_w = int(self.mat_size[0][0] * self.res_shuffle_scale)
            std_h = int(self.mat_size[0][1] * self.res_shuffle_scale)
            x_raw = F.interpolate(x_raw, (std_w, std_h))
        image_list = [self.res_unshuffle(x_raw)]
        compose_image, avg_image = self.fuse.feat_compose(image_list, [similarity_matrix[0]])
        compose_image = compose_image[0]
        compose_image = self.res_shuffle(compose_image)
        fut_img_seq = self.shuffle(compose_image)
        recon_img = self.shuffle(recon_img)
        final_recon_img = self.shuffle(final_recon_img)
        if fut_img_seq.shape[2:] != final_recon_img.shape[2:]:
            fut_img_seq = F.interpolate(fut_img_seq, final_recon_img.shape[2:])
        final_recon_img = self.enhance(torch.cat([final_recon_img, fut_img_seq], dim=1))
        if recon_img.shape[-2] != H or recon_img.shape[-1] != W:
            recon_img = F.interpolate(recon_img, (H, W))
            final_recon_img = F.interpolate(final_recon_img, (H, W))
        recon_img = recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)
        final_recon_img = final_recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)
        return final_recon_img


class _MK(nn.Module):

    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(zeros((l, l)))
            for i in range(l):
                M[-1][i] = (arange(l) - (l - 1) // 2) ** i / factorial(i)
            invM.append(inv(M[-1]))
            self.register_buffer('_M' + str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM' + str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M' + str(j)] for j in range(self.dim()))

    @property
    def invM(self):
        return list(self._buffers['_invM' + str(j)] for j in range(self.dim()))

    def size(self):
        return self._size

    def dim(self):
        return self._dim

    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[newaxis, :]
        x = x.contiguous()
        x = x.view([-1] + list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats) + 1
    sizex = x.size()
    k = x.dim() - 1
    for i in range(k):
        x = tensordot(mats[k - i - 1], x, dim=[1, k])
    x = x.permute([k] + list(range(k))).contiguous()
    x = x.view(sizex)
    return x


class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """

    def __init__(self, shape):
        super(K2M, self).__init__(shape)

    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


class PhyCell_Cell(nn.Module):

    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim, out_channels=self.input_dim, kernel_size=(3, 3), padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)
        next_hidden = hidden_tilde + K * (x - hidden_tilde)
        return next_hidden


class PhyCell(nn.Module):

    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim, F_hidden_dim=self.F_hidden_dims[i], kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):
        batch_size = input_.data.size()[0]
        if first_timestep:
            self.initHidden(batch_size)
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j]
            if j == 0:
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])
        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]))

    def setHidden(self, H):
        self.H = H


class PhyD_ConvLSTM_Cell(nn.Module):

    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(PhyD_ConvLSTM_Cell, self).__init__()
        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PhyD_ConvLSTM(nn.Module):

    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(PhyD_ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            None
            cell_list.append(PhyD_ConvLSTM_Cell(input_shape=self.input_shape, input_dim=cur_input_dim, hidden_dim=self.hidden_dims[i], kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):
        batch_size = input_.data.size()[0]
        if first_timestep:
            self.initHidden(batch_size)
        for j, cell in enumerate(self.cell_list):
            self.H[j] = self.H[j]
            self.C[j] = self.C[j]
            if j == 0:
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))
        return (self.H, self.C), self.H

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]))
            self.C.append(torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class dcgan_upconv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if stride == 2:
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1, output_padding=output_padding), nn.GroupNorm(16, nout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.main(input)


class decoder_D(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(decoder_D, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        output_padding = 1 if stride_2 == 2 else 0
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2)
        self.upc2 = dcgan_upconv(nf, nf, stride=1)
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=stride_2, padding=1, output_padding=output_padding)

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class decoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)
        self.upc2 = dcgan_upconv(nf, nc, stride=1)

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class dcgan_conv(nn.Module):

    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1), nn.GroupNorm(16, nout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):

    def __init__(self, nc=1, nf=32, patch_size=4):
        super(encoder_E, self).__init__()
        assert patch_size in [2, 4]
        stride_2 = patch_size // 2
        self.c1 = dcgan_conv(nc, nf, stride=2)
        self.c2 = dcgan_conv(nf, nf, stride=1)
        self.c3 = dcgan_conv(nf, 2 * nf, stride=stride_2)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class encoder_specific(nn.Module):

    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)
        self.c2 = dcgan_conv(nf, nf, stride=1)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class PhyD_EncoderRNN(torch.nn.Module):

    def __init__(self, phycell, convcell, in_channel=1, patch_size=4):
        super(PhyD_EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(nc=in_channel, patch_size=patch_size)
        self.encoder_Ep = encoder_specific()
        self.encoder_Er = encoder_specific()
        self.decoder_Dp = decoder_specific()
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D(nc=in_channel, patch_size=patch_size)
        self.phycell = phycell
        self.convcell = convcell

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input)
        if decoding:
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)
        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)
        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))
        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        return out_phys, hidden1, output_image, out_phys, out_conv


class PhyDNet_Model(nn.Module):
    """PhyDNet Model

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, configs, **kwargs):
        super(PhyDNet_Model, self).__init__()
        self.pre_seq_length = configs.pre_seq_length
        self.aft_seq_length = configs.aft_seq_length
        _, C, H, W = configs.in_shape
        patch_size = configs.patch_size if configs.patch_size in [2, 4] else 4
        input_shape = H // patch_size, W // patch_size
        self.phycell = PhyCell(input_shape=input_shape, input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7, 7), device=configs.device)
        self.convcell = PhyD_ConvLSTM(input_shape=input_shape, input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3, 3), device=configs.device)
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell, in_channel=C, patch_size=patch_size)
        self.k2m = K2M([7, 7])
        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, target_tensor, constraints, teacher_forcing_ratio=0.0):
        loss = 0
        for ei in range(self.pre_seq_length - 1):
            _, _, output_image, _, _ = self.encoder(input_tensor[:, ei, :, :, :], ei == 0)
            loss += self.criterion(output_image, input_tensor[:, ei + 1, :, :, :])
        decoder_input = input_tensor[:, -1, :, :, :]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(self.aft_seq_length):
            _, _, output_image, _, _ = self.encoder(decoder_input)
            target = target_tensor[:, di, :, :, :]
            loss += self.criterion(output_image, target)
            if use_teacher_forcing:
                decoder_input = target
            else:
                decoder_input = output_image
        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
            m = self.k2m(filters.double()).float()
            loss += self.criterion(m, constraints)
        return loss

    def inference(self, input_tensor, target_tensor, constraints, **kwargs):
        with torch.no_grad():
            loss = 0
            for ei in range(self.pre_seq_length - 1):
                encoder_output, encoder_hidden, output_image, _, _ = self.encoder(input_tensor[:, ei, :, :, :], ei == 0)
                if kwargs.get('return_loss', True):
                    loss += self.criterion(output_image, input_tensor[:, ei + 1, :, :, :])
            decoder_input = input_tensor[:, -1, :, :, :]
            predictions = []
            for di in range(self.aft_seq_length):
                _, _, output_image, _, _ = self.encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image)
                if kwargs.get('return_loss', True):
                    loss += self.criterion(output_image, target_tensor[:, di, :, :, :])
            for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
                filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
                m = self.k2m(filters.double()).float()
                if kwargs.get('return_loss', True):
                    loss += self.criterion(m, constraints)
            return torch.stack(predictions, dim=1), loss


class PredRNN_Model(nn.Module):
    """PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNN_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width])
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


class CausalLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(CausalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 7, height, width]))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_c = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
            self.conv_c2m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_om = nn.Sequential(nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
        else:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_c = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_c2m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_om = nn.Sequential(nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(c_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g_t = torch.tanh(g_x + g_h + g_c)
        c_new = f_t * c_t + i_t * g_t
        c2m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)
        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)
        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m = self.conv_om(m_new)
        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class GHU(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm, initializer=0.001):
        super(GHU, self).__init__()
        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        if layer_norm:
            self.z_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
            self.x_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
        else:
            self.z_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.x_concat = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        if initializer != -1:
            self.initializer = initializer
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, -self.initializer, self.initializer)

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def forward(self, x, z):
        if z is None:
            z = self._init_state(x)
        z_concat = self.z_concat(z)
        x_concat = self.x_concat(x)
        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


class PredRNNpp_Model(nn.Module):
    """PredRNN++ Model

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNpp_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        self.gradient_highway = GHU(num_hidden[0], num_hidden[0], height, width, configs.filter_size, configs.stride, configs.layer_norm)
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(CausalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)
        z_t = None
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1], c_t[1], memory)
            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


class SpatioTemporalLSTMCellv2(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellv2, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 7, height, width]))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), nn.LayerNorm([num_hidden, height, width]))
        else:
            self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
            self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False))
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m


class PredRNNv2_Model(nn.Module):
    """PredRNNv2 Model

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2_Model, self).__init__()
        T, C, H, W = configs.in_shape
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], height, width, configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        return_loss = kwargs.get('return_loss', True)
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)
        for t in range(self.configs.total_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            elif t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            if return_loss:
                for i in range(0, self.num_layers):
                    decouple_loss.append(torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
        if return_loss:
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if return_loss:
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        else:
            loss = None
        return next_frames, loss


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, upsampling=False, act_norm=False, act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation), nn.PixelShuffle(2)])
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True, act_inplace=True):
        super(ConvSC, self).__init__()
        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, upsampling=upsampling, padding=padding, act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(*[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace) for s in samplings[:-1]], ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace))
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace), *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace) for s in samplings[1:]])

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class GroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, act_norm=False, act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker // 2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3, 5, 7, 11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            enc_layers.append(gInception_ST(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(gInception_ST(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers = [gInception_ST(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            dec_layers.append(gInception_ST(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(gInception_ST(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        y = z.reshape(B, T, C, H, W)
        return y


class ConvMixerSubBlock(nn.Module):
    """A block of ConvMixer."""

    def __init__(self, dim, kernel_size=9, activation=nn.GELU):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same')
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        x = x + self.norm_1(self.act_1(self.conv_dw(x)))
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MixMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + (kernel_size // dilation % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.1, init_value=0.01, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation in MogaNet."""

    def __init__(self, embed_dims, mlp_hidden_dims, kernel_size=3, act_layer=nn.GELU, ffn_drop=0.0):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_hidden_dims = mlp_hidden_dims
        self.fc1 = nn.Conv2d(in_channels=embed_dims, out_channels=self.mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(in_channels=self.mlp_hidden_dims, out_channels=self.mlp_hidden_dims, kernel_size=kernel_size, padding=kernel_size // 2, bias=True, groups=self.mlp_hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels=mlp_hidden_dims, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv2d(in_channels=self.mlp_hidden_dims, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(1e-05 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel in MogaNet."""

    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [(i / sum(channel_split)) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0
        self.DW_conv0 = nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=5, padding=(1 + 4 * dw_dilation[0]) // 2, groups=self.embed_dims, stride=1, dilation=dw_dilation[0])
        self.DW_conv1 = nn.Conv2d(in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=5, padding=(1 + 4 * dw_dilation[1]) // 2, groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1])
        self.DW_conv2 = nn.Conv2d(in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=7, padding=(1 + 6 * dw_dilation[2]) // 2, groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2])
        self.PW_conv = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0:self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation in MogaNet."""

    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_shortcut=True):
        super(MultiOrderGatedAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.attn_shortcut = attn_shortcut
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        self.sigma = nn.Parameter(1e-05 * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma * (x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.feat_decompose(x)
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        if self.attn_shortcut:
            x = x + shortcut
        return x


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, init_value=1e-05, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(embed_dims, attn_dw_dilation=attn_dw_dilation, attn_channel_split=attn_channel_split)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=1e-05, act_layer=nn.GELU, norm_layer=GroupNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path, drop=drop, init_value=1e-05)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + (kernel_size // dilation % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(dim, dim // self.reduction, bias=False), nn.ReLU(True), nn.Linear(dim // self.reduction, dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_x = self.conv1(attn)
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u


class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.1, init_value=0.01, act_layer=nn.GELU):
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        self.attn = TemporalAttention(dim, kernel_size)


class CMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LKA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Attention(nn.Module):

    def __init__(self, d_model, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class SABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_value=1e-06, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_1', 'gamma_2'}

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


def UniformerSubBlock(embed_dims, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=1e-06, block_type='Conv'):
    """Build a block of Uniformer."""
    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        return SABlock(dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True, drop=drop, drop_path=drop_path, init_value=init_value)


class VANBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=0.01, act_layer=nn.GELU, attn_shortcut=True):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, attn_shortcut=attn_shortcut)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=0.01, act_layer=nn.GELU):
        super().__init__(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None, mlp_ratio=8.0, drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'
        if model_type == 'gsta':
            self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and 'Invalid model_type in SimVP'
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2, input_resolution=None, model_type=None, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(0.01, drop_path, self.N2)]
        enc_layers = [MetaBlock(channel_in, channel_hid, input_resolution, model_type, mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock(channel_hid, channel_hid, input_resolution, model_type, mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        enc_layers.append(MetaBlock(channel_hid, channel_in, input_resolution, model_type, mlp_ratio, drop, drop_path=drop_path, layer_i=N2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z.reshape(B, T, C, H, W)
        return y


class SimVP_Model(nn.Module):
    """SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA', mlp_ratio=8.0, drop=0.0, drop_path=0.0, spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T * hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T * hid_S, hid_T, N_T, input_resolution=(H, W), model_type=model_type, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


class SwinLSTMCell(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size, depth, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, flag=None):
        """
        Args:
        flag:  0 UpSample   1 DownSample  2 STconvert
        """
        super(SwinLSTMCell, self).__init__()
        self.STBs = nn.ModuleList(STB(i, dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, flag=flag) for i in range(depth))

    def forward(self, xt, hidden_states):
        """
        Args:
        xt: input for t period 
        hidden_states: [hx, cx] hidden_states for t-1 period
        """
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C)
            cx = torch.zeros(B, L, C)
        else:
            hx, cx = hidden_states
        outputs = []
        for index, layer in enumerate(self.STBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)
                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)
        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)
        return Ht, (Ht, Ct)


class DownSample(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, flag=1):
        super(DownSample, self).__init__()
        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), dim=int(embed_dim * 2 ** i_layer))
            layer = SwinLSTMCell(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths_downsample[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])], norm_layer=norm_layer, flag=flag)
            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):
        x = self.patch_embed(x)
        hidden_states_down = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)
        return hidden_states_down, x


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    """ Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution
        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3), stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)
        return x


class UpSample(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, flag=0):
        super(UpSample, self).__init__()
        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]
        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            resolution1 = patches_resolution[0] // 2 ** (self.num_layers - i_layer)
            resolution2 = patches_resolution[1] // 2 ** (self.num_layers - i_layer)
            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)
            layer = SwinLSTMCell(dim=dimension, input_resolution=(resolution1, resolution2), depth=depths_upsample[self.num_layers - 1 - i_layer], num_heads=num_heads[self.num_layers - 1 - i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_upsample[:self.num_layers - 1 - i_layer]):sum(depths_upsample[:self.num_layers - 1 - i_layer + 1])], norm_layer=norm_layer, flag=flag)
            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        hidden_states_up = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)
        x = torch.sigmoid(self.Unembed(x))
        return hidden_states_up, x


class SwinLSTM_D_Model(nn.Module):
    """SwinLSTM 
    Implementation of `SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin
    Transformer and LSTM <http://arxiv.org/abs/2308.09891>`_.

    """

    def __init__(self, depths_downsample, depths_upsample, num_heads, configs, **kwargs):
        super(SwinLSTM_D_Model, self).__init__()
        T, C, H, W = configs.in_shape
        assert H == W, 'Only support H = W for image input'
        self.configs = configs
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample
        self.Downsample = DownSample(img_size=H, patch_size=configs.patch_size, in_chans=C, embed_dim=configs.embed_dim, depths_downsample=depths_downsample, num_heads=num_heads, window_size=configs.window_size)
        self.Upsample = UpSample(img_size=H, patch_size=configs.patch_size, in_chans=C, embed_dim=configs.embed_dim, depths_upsample=depths_upsample, num_heads=num_heads, window_size=configs.window_size)
        self.MSE_criterion = nn.MSELoss()

    def forward(self, frames_tensor, **kwargs):
        T, C, H, W = self.configs.in_shape
        total_T = frames_tensor.shape[1]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = frames[:, :T]
        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)
        next_frames = []
        last_frame = input_frames[:, -1]
        for i in range(T - 1):
            states_down, x = self.Downsample(input_frames[:, i], states_down)
            states_up, output = self.Upsample(x, states_up)
            next_frames.append(output)
        for i in range(total_T - T):
            states_down, x = self.Downsample(last_frame, states_down)
            states_up, output = self.Upsample(x, states_up)
            next_frames.append(output)
            last_frame = output
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


class STconvert(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, flag=2):
        super(STconvert, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size
        self.patch_inflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        self.layer = SwinLSTMCell(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]), depth=depths, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, flag=flag)

    def forward(self, x, h=None):
        x = self.patch_embed(x)
        x, hidden_state = self.layer(x, h)
        x = torch.sigmoid(self.patch_inflated(x))
        return x, hidden_state


class SwinLSTM_B_Model(nn.Module):

    def __init__(self, configs, **kwargs):
        super(SwinLSTM_B_Model, self).__init__()
        T, C, H, W = configs.in_shape
        assert H == W, 'Only support H = W for image input'
        self.configs = configs
        self.ST = STconvert(img_size=H, patch_size=configs.patch_size, in_chans=C, embed_dim=configs.embed_dim, depths=configs.depths, num_heads=configs.num_heads, window_size=configs.window_size)
        self.MSE_criterion = nn.MSELoss()

    def forward(self, frames_tensor, **kwargs):
        T, C, H, W = self.configs.in_shape
        total_T = frames_tensor.shape[1]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = frames[:, :T]
        states = None
        next_frames = []
        last_frame = input_frames[:, -1]
        for i in range(T - 1):
            output, states = self.ST(input_frames[:, i], states)
            next_frames.append(output)
        for i in range(total_T - T):
            output, states = self.ST(last_frame, states)
            next_frames.append(output)
            last_frame = output
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class gnconv(nn.Module):

    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [(dim // 2 ** i) for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])
        self.scale = s
        None

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x


class HorBlock(nn.Module):
    """ HorNet block """

    def __init__(self, dim, order=4, mlp_ratio=4, drop_path=0.0, init_value=1e-06, gnconv=gnconv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.gnconv = gnconv(dim, order)
        self.norm2 = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        gamma1 = self.gamma1.view(C, 1, 1)
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """

    def __init__(self, shape):
        super(M2K, self).__init__(shape)

    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.1, init_value=1e-06):
        super().__init__(dim, mlp_ratio=mlp_ratio, drop_path=drop_path, init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)


class WaveletTransform2D(nn.Module):
    """Compute a two-dimensional wavelet transform.
        loss = nn.MSELoss()
        data = torch.rand(1, 3, 128, 256)
        DWT = WaveletTransform2D()
        IDWT = WaveletTransform2D(inverse=True)

        LL, LH, HL, HH = DWT(data)
        recdata = IDWT([LL, LH, HL, HH])
        print(loss(data, recdata))
    """

    def __init__(self, inverse=False, wavelet='haar', mode='constant'):
        super(WaveletTransform2D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)
        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)

    def outer(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: 'int', filter_len: 'int'):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)
        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv2d(data, filter.repeat(c, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = torch.nn.functional.conv_transpose2d(data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c)
            return rec_res


class HighFocalFrequencyLoss(nn.Module):
    """ Example:
        fake = torch.randn(4, 3, 128, 64)
        real = torch.randn(4, 3, 128, 64)
        hffl = HighFocalFrequencyLoss()

        loss = hffl(fake, real)
        print(loss)
    """

    def __init__(self, loss_weight=0.001, level=1, tau=0.1, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=True, batch_matrix=False):
        super(HighFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.level = level
        self.tau = tau
        self.DWT = WaveletTransform2D()

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, 'Patch factor should be divisible by image height and width'
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])
        y = torch.stack(patch_list, 1)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def build_freq_mask(self, shape):
        H, W = shape[-2:]
        radius = self.tau * max(H, W)
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        mask = torch.ones_like(X, dtype=torch.float32)
        centers = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
        for center in centers:
            distance = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
            mask[distance <= radius] = 0
        return mask

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, 'The values of spectrum weight matrix should be in the range [0, 1], but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item())
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        mask = self.build_freq_mask(weight_matrix.shape)
        loss = weight_matrix * freq_distance * mask
        return torch.mean(loss)

    def frequency_loss(self, pred, target, matrix=None):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred = rearrange(pred, 'b t c h w -> (b t) c h w') if kwargs['reshape'] is True else pred
        target = rearrange(target, 'b t c h w -> (b t) c h w') if kwargs['reshape'] is True else target
        loss = 0
        for level in range(self.level):
            pred, _, _, _ = self.DWT(pred)
            target, _, _, _ = self.DWT(target)
            loss += self.frequency_loss(pred, target, matrix)
        return loss * self.loss_weight


class WaveletTransform3D(nn.Module):
    """Compute a three-dimensional wavelet transform.
        Example:
            loss = nn.MSELoss()
            data = torch.rand(1, 3, 10, 128, 256)
            DWT = WaveletTransform3D()
            IDWT = WaveletTransform3D(inverse=True)

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT(data)
            recdata = IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
            print(loss(data, recdata))

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT_3D(data)
            recdata = IDWT_3D(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            print(loss(data, recdata))
        """

    def __init__(self, inverse=False, wavelet='haar', mode='constant'):
        super(WaveletTransform3D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)
        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        self.dim_size = lo.shape[-1]
        size = [self.dim_size] * 3
        lll = self.outer(lo, self.outer(lo, lo)).reshape(size)
        llh = self.outer(lo, self.outer(lo, hi)).reshape(size)
        lhl = self.outer(lo, self.outer(hi, lo)).reshape(size)
        lhh = self.outer(lo, self.outer(hi, hi)).reshape(size)
        hll = self.outer(hi, self.outer(lo, lo)).reshape(size)
        hlh = self.outer(hi, self.outer(lo, hi)).reshape(size)
        hhl = self.outer(hi, self.outer(hi, lo)).reshape(size)
        hhh = self.outer(hi, self.outer(hi, hi)).reshape(size)
        filters = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)

    def outer(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: 'int', filter_len: 'int'):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        pad_back, pad_front = self.get_pad(data.shape[-3], self.dim_size)
        pad_bottom, pad_top = self.get_pad(data.shape[-2], self.dim_size)
        pad_right, pad_left = self.get_pad(data.shape[-1], self.dim_size)
        data_pad = torch.nn.functional.pad(data, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, t, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv3d(data, filter.repeat(c, 1, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, t, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, t, h, w)
            rec_res = torch.nn.functional.conv_transpose3d(data, self.filters.repeat(c, 1, 1, 1, 1), stride=2, groups=c)
            return rec_res


class FrequencyAttention(nn.Module):

    def __init__(self, in_dim, out_dim, reduction=32):
        super(FrequencyAttention, self).__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = activations.HardSwish(inplace=True)
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.avgpool_h(x)
        x_w = self.avgpool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class TF_AwareBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, ls_init_value=0.01, drop_path=0.1, large_kernel=51, small_kernel=5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.lk1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(large_kernel, 5), groups=dim, padding='same'), nn.BatchNorm2d(dim))
        self.lk2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(5, large_kernel), groups=dim, padding='same'), nn.BatchNorm2d(dim))
        self.sk = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(small_kernel, small_kernel), groups=dim, padding='same'), nn.BatchNorm2d(dim))
        self.low_frequency_attn = FrequencyAttention(in_dim=dim, out_dim=dim, reduction=4)
        self.high_frequency_attn = FrequencyAttention(in_dim=dim, out_dim=dim, reduction=4)
        self.temporal_mixer = InvertedResidual(in_chs=dim, out_chs=dim, dw_kernel_size=7, exp_ratio=mlp_ratio, se_layer=partial(SqueezeExcite, rd_ratio=0.25), noskip=True)
        self.layer_scale_1 = nn.Parameter(ls_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(ls_init_value * torch.ones(dim), requires_grad=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        attn = self.norm1(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * (self.low_frequency_attn(self.lk1(attn) + self.lk2(attn)) + self.high_frequency_attn(self.sk(attn))))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.temporal_mixer(self.norm2(x)))
        return x


class TF_AwareBlocks(nn.Module):

    def __init__(self, dim, num_blocks, drop_path, use_bottleneck=None, use_hid=False, mlp_ratio=4.0, drop=0.0, ls_init_value=0.01, large_kernel=51, small_kernel=5):
        super().__init__()
        assert len(drop_path) == num_blocks, "drop_path list doesn't match num_blocks"
        self.use_hid = use_hid
        self.use_bottleneck = use_bottleneck
        blocks = []
        for i in range(num_blocks):
            block = TF_AwareBlock(dim, mlp_ratio, drop, ls_init_value, drop_path[i], large_kernel, small_kernel)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.concat_block = nn.Conv2d(dim * 2, dim, 3, 1, 1) if use_hid == True else None
        self.DWT = WaveletTransform3D(inverse=False) if use_bottleneck == 'decompose' else None
        self.IDWT = WaveletTransform3D(inverse=True) if use_bottleneck == 'decompose' else None

    def forward(self, x, skip=None):
        if self.concat_block is not None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = self.concat_block(torch.cat([x, skip], dim=1))
            x = self.blocks(x)
            x = rearrange(x, 'b (c t) h w -> b c t h w', t=t)
            return x
        elif self.concat_block is None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = skip = self.blocks(x)
            x = rearrange(x, 'b (c t) h w -> b c t h w', t=t)
            return x, skip
        elif self.use_bottleneck is not None:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x) if self.use_bottleneck == 'decompose' else [x, None, None, None, None, None, None, None]
            b, c, t, h, w = LLL.shape
            LLL = rearrange(LLL, 'b c t h w -> b (c t) h w')
            LLL = self.blocks(LLL)
            LLL = rearrange(LLL, 'b (c t) h w -> b c t h w', t=t)
            x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]) if self.use_bottleneck == 'decompose' else LLL
            return x


class Wavelet_3D_Embedding(nn.Module):

    def __init__(self, in_dim, out_dim, emb_dim=None):
        super().__init__()
        emb_dim = in_dim if emb_dim == None else emb_dim
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, in_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(in_dim), nn.GELU())
        self.conv_1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(out_dim), nn.GELU())
        self.conv_emb = nn.Conv3d(emb_dim * 4, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.DWT = WaveletTransform3D(inverse=False)

    def forward(self, x, x_emb=None):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x_emb)
        lo_temp = torch.cat([LLL, LHL, HLL, HHL], dim=1)
        hi_temp = torch.cat([LLH, LHH, HLH, HHH], dim=1)
        x_emb = torch.cat([lo_temp, hi_temp], dim=2)
        x_emb = self.conv_emb(x_emb)
        x = self.conv_0(x)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x)
        spatio_lo_coeffs = torch.cat([LLL, LLH], dim=2)
        spatio_hi_coeffs = torch.cat([LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x = self.conv_1(spatio_lo_coeffs)
        return x + x_emb, spatio_hi_coeffs


class Wavelet_3D_Reconstruction(nn.Module):

    def __init__(self, in_dim, out_dim, hi_dim):
        super().__init__()
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(out_dim), nn.GELU())
        self.conv_hi = nn.Sequential(nn.Conv3d(int(hi_dim * 6), int(out_dim * 6), kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=6), nn.BatchNorm3d(out_dim * 6), nn.GELU())
        self.IDWT = WaveletTransform3D(inverse=True)

    def forward(self, x, skip_hi=None):
        LLL, LLH = torch.chunk(self.conv_0(x), chunks=2, dim=2)
        LHL, LHH, HLL, HLH, HHL, HHH = torch.chunk(self.conv_hi(skip_hi), chunks=6, dim=1)
        x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
        return x


class WaST_level1(nn.Module):

    def __init__(self, in_shape, encoder_dim, block_list=[2, 2, 2], drop_path_rate=0.1, mlp_ratio=4.0, **kwargs):
        super().__init__()
        frame, in_dim, H, W = in_shape
        self.block_list = block_list
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.block_list))]
        indexes = list(accumulate(block_list))
        dp_list = [dp_list[start:end] for start, end in zip([0] + indexes, indexes)]
        self.conv_in = nn.Sequential(nn.Conv3d(in_dim, encoder_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), nn.BatchNorm3d(encoder_dim), nn.GELU())
        self.translator1 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[0], drop_path=dp_list[0], mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5)
        self.wavelet_embed1 = Wavelet_3D_Embedding(in_dim=encoder_dim, out_dim=encoder_dim * 2, emb_dim=in_dim)
        self.bottleneck_translator = TF_AwareBlocks(dim=encoder_dim * 2 * frame, num_blocks=block_list[1], drop_path=dp_list[1], use_bottleneck=True, mlp_ratio=mlp_ratio, large_kernel=21, small_kernel=5)
        self.wavelet_recon1 = Wavelet_3D_Reconstruction(in_dim=encoder_dim * 2, out_dim=encoder_dim, hi_dim=encoder_dim)
        self.translator2 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[2], drop_path=dp_list[2], use_hid=True, mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5)
        self.conv_out = nn.Sequential(nn.BatchNorm3d(encoder_dim), nn.GELU(), nn.Conv3d(encoder_dim, in_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)))

    def update_drop_path(self, drop_path_rate):
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.block_list))]
        indexes = list(accumulate(self.block_list))
        dp_lists = [dp_list[start:end] for start, end in zip([0] + indexes, indexes)]
        dp_apply_blocks = [self.translator1.blocks, self.bottleneck_translator.blocks, self.translator2.blocks]
        for translators, dp_list_translators in zip(dp_apply_blocks, dp_lists):
            for translator, dp_list_translator in zip(translators, dp_list_translators):
                translator.drop_path.drop_prob = dp_list_translator

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        ori_img = x
        x = self.conv_in(x)
        x, tskip1 = self.translator1(x)
        x, skip1 = self.wavelet_embed1(x, x_emb=ori_img)
        x = self.bottleneck_translator(x)
        x = self.wavelet_recon1(x, skip1)
        x = self.translator2(x, tskip1)
        x = self.conv_out(x)
        x = rearrange(x, 'b c t h w -> b t c h w')
        return x


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.fc = nn.Linear(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.bn(self.conv(imgs))

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {'loss': 0.5, 'log_vars': {'accuracy': 0.98}, 'num_samples': 1}
        return outputs


class SingleBNModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.bn(imgs)


class GNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.GroupNorm(1, 1)
        self.test_cfg = None


class NoBNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.conv(imgs)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttentionModule,
     lambda: ([], {'dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelAggregationFFN,
     lambda: ([], {'embed_dims': 4, 'mlp_hidden_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvMixerSubBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DWConv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 768, 64, 64])], {})),
    (GroupConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupNorm,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImageEnhancer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64])], {})),
    (LKA,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MetaBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MidIncepNet,
     lambda: ([], {'channel_in': 4, 'channel_hid': 4, 'N2': 4}),
     lambda: ([torch.rand([4, 1, 4, 4, 4])], {})),
    (MixMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RRDBEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualDenseBlock_4C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (SimpleMatrixPredictor3DConv_direct,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 3, 64, 4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SubModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Up,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VANBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VANSubBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_MK,
     lambda: ([], {'shape': [4, 4]}),
     lambda: ([], {})),
    (decoder_D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {})),
    (decoder_specific,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {})),
    (encoder_E,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (encoder_specific,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (gInception_ST,
     lambda: ([], {'C_in': 4, 'C_hid': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

