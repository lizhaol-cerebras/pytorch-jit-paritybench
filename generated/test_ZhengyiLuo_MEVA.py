
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


import random


import logging


import numpy as np


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import torch.utils.data as data


import time


import copy


import scipy.misc


import scipy.io as scio


from collections import defaultdict


from torchvision.transforms.functional import to_tensor


from torch.utils.data import ConcatDataset


from torch.utils.data import DataLoader


import torch.nn as nn


from torch import nn


from torchvision import models


from torch.nn.utils import weight_norm


import math


from torch.optim import lr_scheduler


from torch.nn import functional as F


import torch.nn.functional as F


from scipy.spatial.transform import Rotation as sRot


import torchvision.models.resnet as resnet


from collections import OrderedDict


from functools import reduce


from typing import List


from typing import Union


from torch import optim


from torch.utils.tensorboard import SummaryWriter


import torch.backends.cudnn as cudnn


class CMLP(nn.Module):

    def __init__(self, input_dim, cond_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.cond_dim = cond_dim
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim + cond_dim, nh))
            last_dim = nh

    def forward(self, c, x):
        for affine in self.affine_layers:
            x = torch.cat((c, x), dim=1)
            x = self.activation(affine(x))
        return x


class Discriminator(nn.Module):

    def __init__(self, net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.logic = nn.Linear(net_out_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        prob = torch.sigmoid(self.logic(x))
        return prob


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


def batch_to(dst, *args):
    return [(x if x is not None else None) for x in args]


zeros = torch.zeros


class RNN(nn.Module):

    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out


class ERDNet(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.encoder_mlp = MLP(state_dim, (500,), 'relu')
        self.encoder_linear = nn.Linear(500, 500)
        self.lstm1 = RNN(500, 1000, 'lstm')
        self.lstm2 = RNN(1000, 1000, 'lstm')
        self.decoder_mlp = MLP(1000, (500, 100), 'relu')
        self.decoder_linear = nn.Linear(100, state_dim)
        self.mode = 'batch'

    def initialize(self, mode):
        self.mode = mode
        self.lstm1.set_mode(mode)
        self.lstm2.set_mode(mode)
        self.lstm1.initialize()
        self.lstm2.initialize()

    def forward(self, x):
        if self.mode == 'batch':
            batch_size = x.shape[1]
            x = x.view(-1, x.shape[-1])
        x = self.encoder_mlp(x)
        x = self.encoder_linear(x)
        if self.mode == 'batch':
            x = x.view(-1, batch_size, x.shape[-1])
        x = self.lstm1(x)
        x = self.lstm2(x)
        if self.mode == 'batch':
            x = x.view(-1, x.shape[-1])
        x = self.decoder_mlp(x)
        x = self.decoder_linear(x)
        return x


class MobileNet(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class ResNet(nn.Module):

    def __init__(self, out_dim, fix_params=False, running_stats=False):
        super().__init__()
        self.out_dim = out_dim
        self.resnet = models.resnet18(pretrained=True)
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)
        self.bn_stats(running_stats)

    def forward(self, x):
        return self.resnet(x)

    def bn_stats(self, track_running_stats):
        for m in self.modules():
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = track_running_stats


class SimpleCNN(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=4)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=4, stride=4)
        self.fc = nn.Linear(144, out_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0), -1))
        return x


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout, causal):
        super().__init__()
        padding = (kernel_size - 1) * dilation // (1 if causal else 2)
        modules = []
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        modules.append(self.conv1)
        if causal:
            modules.append(Chomp1d(padding))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        modules.append(self.conv2)
        if causal:
            modules.append(Chomp1d(padding))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, causal=False):
        super().__init__()
        assert kernel_size % 2 == 1
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout, causal=causal)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


LongTensor = torch.LongTensor


tensor = torch.tensor


class VideoForecastNet(nn.Module):

    def __init__(self, cnn_feat_dim, state_dim, v_hdim=128, v_margin=10, v_net_type='lstm', v_net_param=None, s_hdim=None, s_net_type='id', dynamic_v=False):
        super().__init__()
        s_hdim = state_dim if s_hdim is None else s_hdim
        self.mode = 'test'
        self.cnn_feat_dim = cnn_feat_dim
        self.state_dim = state_dim
        self.v_net_type = v_net_type
        self.v_hdim = v_hdim
        self.v_margin = v_margin
        self.s_net_type = s_net_type
        self.s_hdim = s_hdim
        self.dynamic_v = dynamic_v
        self.out_dim = v_hdim + s_hdim
        if v_net_type == 'lstm':
            self.v_net = RNN(cnn_feat_dim, v_hdim, v_net_type, bi_dir=False)
        elif v_net_type == 'tcn':
            if v_net_param is None:
                v_net_param = {}
            tcn_size = v_net_param.get('size', [64, 128])
            dropout = v_net_param.get('dropout', 0.2)
            kernel_size = v_net_param.get('kernel_size', 3)
            assert tcn_size[-1] == v_hdim
            self.v_net = TemporalConvNet(cnn_feat_dim, tcn_size, kernel_size=kernel_size, dropout=dropout, causal=True)
        if s_net_type == 'lstm':
            self.s_net = RNN(state_dim, s_hdim, s_net_type, bi_dir=False)
        self.v_out = None
        self.t = 0
        self.indices = None
        self.s_scatter_indices = None
        self.s_gather_indices = None
        self.v_gather_indices = None
        self.cnn_feat_ctx = None
        self.num_episode = None
        self.max_episode_len = None
        self.set_mode('test')

    def set_mode(self, mode):
        self.mode = mode
        if self.s_net_type == 'lstm':
            if mode == 'train':
                self.s_net.set_mode('batch')
            else:
                self.s_net.set_mode('step')

    def initialize(self, x):
        if self.mode == 'test':
            self.v_out = self.forward_v_net(x.unsqueeze(1)[:self.v_margin])[-1]
            if self.s_net_type == 'lstm':
                self.s_net.initialize()
            self.t = 0
        elif self.mode == 'train':
            masks, cnn_feat, v_metas = x
            device, dtype = masks.device, masks.dtype
            end_indice = np.where(masks.cpu().numpy() == 0)[0]
            v_metas = v_metas[end_indice, :]
            num_episode = len(end_indice)
            end_indice = np.insert(end_indice, 0, -1)
            max_episode_len = int(np.diff(end_indice).max())
            self.num_episode = num_episode
            self.max_episode_len = max_episode_len
            self.indices = np.arange(masks.shape[0])
            for i in range(1, num_episode):
                start_index = end_indice[i] + 1
                end_index = end_indice[i + 1] + 1
                self.indices[start_index:end_index] += i * max_episode_len - start_index
            self.cnn_feat_ctx = np.zeros((self.v_margin + max_episode_len if self.dynamic_v else self.v_margin, num_episode, self.cnn_feat_dim))
            for i in range(num_episode):
                exp_ind, start_ind = v_metas[i, :]
                self.cnn_feat_ctx[:self.v_margin, i, :] = cnn_feat[exp_ind][start_ind - self.v_margin:start_ind]
            self.cnn_feat_ctx = tensor(self.cnn_feat_ctx, dtype=dtype, device=device)
            self.s_scatter_indices = LongTensor(np.tile(self.indices[:, None], (1, self.state_dim)))
            self.s_gather_indices = LongTensor(np.tile(self.indices[:, None], (1, self.s_hdim)))
            self.v_gather_indices = LongTensor(np.tile(self.indices[:, None], (1, self.v_hdim)))

    def forward(self, x):
        if self.mode == 'test':
            if self.s_net_type == 'lstm':
                x = self.s_net(x)
            x = torch.cat((self.v_out, x), dim=1)
            self.t += 1
        elif self.mode == 'train':
            if self.dynamic_v:
                v_ctx = self.forward_v_net(self.cnn_feat_ctx)[self.v_margin:]
            else:
                v_ctx = self.forward_v_net(self.cnn_feat_ctx)[[-1]]
                v_ctx = v_ctx.repeat(self.max_episode_len, 1, 1)
            v_ctx = v_ctx.transpose(0, 1).contiguous().view(-1, self.v_hdim)
            v_out = torch.gather(v_ctx, 0, self.v_gather_indices)
            if self.s_net_type == 'lstm':
                s_ctx = zeros((self.num_episode * self.max_episode_len, self.state_dim), device=x.device)
                s_ctx.scatter_(0, self.s_scatter_indices, x)
                s_ctx = s_ctx.view(-1, self.max_episode_len, self.state_dim).transpose(0, 1).contiguous()
                s_ctx = self.s_net(s_ctx).transpose(0, 1).contiguous().view(-1, self.s_hdim)
                s_out = torch.gather(s_ctx, 0, self.s_gather_indices)
            else:
                s_out = x
            x = torch.cat((v_out, s_out), dim=1)
        return x

    def forward_v_net(self, x):
        if self.v_net_type == 'tcn':
            x = x.permute(1, 2, 0).contiguous()
        x = self.v_net(x)
        if self.v_net_type == 'tcn':
            x = x.permute(2, 0, 1).contiguous()
        return x


class VideoRegNet(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, no_cnn=False, frame_shape=(3, 224, 224), mlp_dim=(300, 200), cnn_type='resnet', v_net_type='lstm', v_net_param=None, cnn_rs=True, causal=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.no_cnn = no_cnn
        self.frame_shape = frame_shape
        if no_cnn:
            self.cnn = None
        elif cnn_type == 'resnet':
            self.cnn = ResNet(cnn_fdim, running_stats=cnn_rs)
        elif cnn_type == 'mobile':
            self.cnn = MobileNet(cnn_fdim)
        self.v_net_type = v_net_type
        if v_net_type == 'lstm':
            self.v_net = RNN(cnn_fdim, v_hdim, v_net_type, bi_dir=not causal)
        elif v_net_type == 'tcn':
            if v_net_param is None:
                v_net_param = {}
            tcn_size = v_net_param.get('size', [64, 128])
            dropout = v_net_param.get('dropout', 0.2)
            kernel_size = v_net_param.get('kernel_size', 3)
            assert tcn_size[-1] == v_hdim
            self.v_net = TemporalConvNet(cnn_fdim, tcn_size, kernel_size=kernel_size, dropout=dropout, causal=causal)
        self.mlp = MLP(v_hdim, mlp_dim, 'relu')
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)

    def forward_v_net(self, x):
        if self.v_net_type == 'tcn':
            x = x.permute(1, 2, 0).contiguous()
        x = self.v_net(x)
        if self.v_net_type == 'tcn':
            x = x.permute(2, 0, 1).contiguous()
        return x

    def forward(self, x):
        if self.cnn is not None:
            x = self.cnn(x.view((-1,) + self.frame_shape)).view(-1, x.size(1), self.cnn_fdim)
        x = self.forward_v_net(x).view(-1, self.v_hdim)
        x = self.mlp(x)
        x = self.linear(x)
        return x

    def get_cnn_feature(self, x):
        return self.cnn(x.view((-1,) + self.frame_shape))


class VideoStateNet(nn.Module):

    def __init__(self, cnn_feat_dim, v_hdim=128, v_margin=10, v_net_type='lstm', v_net_param=None, causal=False):
        super().__init__()
        self.mode = 'test'
        self.cnn_feat_dim = cnn_feat_dim
        self.v_net_type = v_net_type
        self.v_hdim = v_hdim
        self.v_margin = v_margin
        if v_net_type == 'lstm':
            self.v_net = RNN(cnn_feat_dim, v_hdim, v_net_type, bi_dir=not causal)
        elif v_net_type == 'tcn':
            if v_net_param is None:
                v_net_param = {}
            tcn_size = v_net_param.get('size', [64, 128])
            dropout = v_net_param.get('dropout', 0.2)
            kernel_size = v_net_param.get('kernel_size', 3)
            assert tcn_size[-1] == v_hdim
            self.v_net = TemporalConvNet(cnn_feat_dim, tcn_size, kernel_size=kernel_size, dropout=dropout, causal=causal)
        self.v_out = None
        self.t = 0
        self.indices = None
        self.scatter_indices = None
        self.gather_indices = None
        self.cnn_feat_ctx = None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, x):
        if self.mode == 'test':
            self.v_out = self.forward_v_net(x.unsqueeze(1)).squeeze(1)[self.v_margin:-self.v_margin]
            self.t = 0
        elif self.mode == 'train':
            masks, cnn_feat, v_metas = x
            device, dtype = masks.device, masks.dtype
            end_indice = np.where(masks.cpu().numpy() == 0)[0]
            v_metas = v_metas[end_indice, :]
            num_episode = len(end_indice)
            end_indice = np.insert(end_indice, 0, -1)
            max_episode_len = int(np.diff(end_indice).max())
            self.indices = np.arange(masks.shape[0])
            for i in range(1, num_episode):
                start_index = end_indice[i] + 1
                end_index = end_indice[i + 1] + 1
                self.indices[start_index:end_index] += i * max_episode_len - start_index
            self.cnn_feat_ctx = np.zeros((max_episode_len + 2 * self.v_margin, num_episode, self.cnn_feat_dim))
            for i in range(num_episode):
                exp_ind, start_ind = v_metas[i, :]
                self.cnn_feat_ctx[:, i, :] = cnn_feat[exp_ind][start_ind - self.v_margin:start_ind + max_episode_len + self.v_margin]
            self.cnn_feat_ctx = tensor(self.cnn_feat_ctx, dtype=dtype, device=device)
            self.scatter_indices = LongTensor(np.tile(self.indices[:, None], (1, self.cnn_feat_dim)))
            self.gather_indices = LongTensor(np.tile(self.indices[:, None], (1, self.v_hdim)))

    def forward(self, x):
        if self.mode == 'test':
            x = torch.cat((self.v_out[[self.t], :], x), dim=1)
            self.t += 1
        elif self.mode == 'train':
            v_ctx = self.forward_v_net(self.cnn_feat_ctx)[self.v_margin:-self.v_margin]
            v_ctx = v_ctx.transpose(0, 1).contiguous().view(-1, self.v_hdim)
            v_out = torch.gather(v_ctx, 0, self.gather_indices)
            x = torch.cat((v_out, x), dim=1)
        return x

    def forward_v_net(self, x):
        if self.v_net_type == 'tcn':
            x = x.permute(1, 2, 0).contiguous()
        x = self.v_net(x)
        if self.v_net_type == 'tcn':
            x = x.permute(2, 0, 1).contiguous()
        return x


class TemporalEncoder(nn.Module):

    def __init__(self, n_layers=1, hidden_size=2048, add_linear=False, bidirectional=False, use_residual=True, output_size=2048):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=n_layers)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, output_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, output_size)
        self.use_residual = use_residual
        self.output_size = output_size

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)
        y, _ = self.gru(x)
        if self.linear:
            y = torch.tanh(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, self.output_size)
        if self.use_residual and y.shape[-1] == self.output_size:
            y = y + x
        y = y.permute(1, 0, 2)
        return y


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


H36M_TO_J14 = H36M_TO_J17[:14]


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


MEVA_DATA_DIR = 'data/meva_data'


SMPL_MODEL_DIR = MEVA_DATA_DIR


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]


def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-09)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints, rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1), translation=pred_cam_t, focal_length=5000.0, camera_center=camera_center)
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-06)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-06)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    return rot_mats


def quaternion_to_angle_axis(quaternion: 'torch.Tensor') ->torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: 'torch.Tensor' = quaternion[..., 1]
    q2: 'torch.Tensor' = quaternion[..., 2]
    q3: 'torch.Tensor' = quaternion[..., 3]
    sin_squared_theta: 'torch.Tensor' = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: 'torch.Tensor' = torch.sqrt(sin_squared_theta)
    cos_theta: 'torch.Tensor' = quaternion[..., 0]
    two_theta: 'torch.Tensor' = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: 'torch.Tensor' = two_theta / sin_theta
    k_neg: 'torch.Tensor' = 2.0 * torch.ones_like(sin_theta)
    k: 'torch.Tensor' = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: 'torch.Tensor' = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


class Config:

    def __init__(self, cfg_id, work_dir=''):
        self.id = cfg_id
        cfg_name = os.path.join(work_dir, 'meva/cfg/%s.yml' % cfg_id)
        if not os.path.exists(cfg_name):
            None
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))
        self.base_dir = os.path.join(work_dir, 'results')
        self.cfg_dir = '%s/meva/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.batch_size = cfg.get('batch_size', 8)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.norm_data = cfg.get('norm_data', True)
        self.lr = cfg['lr']
        self.model_specs = cfg.get('model_specs', dict())
        self.num_epoch = cfg['num_epoch']
        self.num_epoch_fix = cfg.get('num_epoch_fix', self.num_epoch)
        self.model_path = os.path.join(self.model_dir, 'model_%04d.p')
        self.data_specs = cfg.get('data_specs', dict())
        self.loss_specs = cfg.get('loss_specs', dict())
        self.num_samples = cfg.get('num_samples', 5000)


class VAErec(nn.Module):

    def __init__(self, nx, t_total, specs):
        super(VAErec, self).__init__()
        self.nx = nx
        self.nz = nz = specs['nz']
        self.t_total = t_total
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', True)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.additive = specs.get('additive', False)
        self.e_rnn = RNN(nx, nx_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(nx_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nx_rnn, nh_mlp + [nx_rnn], activation='tanh')
        self.d_rnn = RNN(nx + nx_rnn, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_rnn(x).mean(dim=0)
        else:
            h_x = self.e_rnn(x)[-1]
        return h_x

    def encode_x_all(self, x):
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return h_x, self.e_mu(h), self.e_logvar(h)

    def encode(self, x):
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        x_p = x[0, :]
        for i in range(self.t_total):
            rnn_in = torch.cat([x_p, z], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        return x_rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)

    def step(self, model):
        pass


class VAErecV2(nn.Module):

    def __init__(self, nx, t_total, specs):
        super(VAErecV2, self).__init__()
        self.nx = nx
        self.nz = nz = specs['nz']
        self.t_total = t_total
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.e_birnn = e_birnn = specs.get('e_birnn', False)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', True)
        self.nx_rnn = nx_rnn = specs.get('nx_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.additive = specs.get('additive', False)
        self.e_rnn = RNN(nx, nx_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(nx_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nx_rnn, nh_mlp + [nx_rnn], activation='relu')
        self.d_rnn = RNN(nx + nx_rnn, nx_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nx_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
        self.d_rnn.set_mode('step')
        self.init_pose_mlp = MLP(nx_rnn, nh_mlp, activation='relu')
        self.init_pose_out = nn.Linear(self.init_pose_mlp.out_dim, nx)

    def encode_x(self, x):
        if self.e_birnn:
            h_x = self.e_rnn(x).mean(dim=0)
        else:
            h_x = self.e_rnn(x)[-1]
        return h_x

    def encode(self, x):
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_p=None):
        if x_p == None:
            h_init_pose = self.init_pose_mlp(z)
            x = self.init_pose_out(h_init_pose)
            x_p = x
        self.d_rnn.initialize(batch_size=z.shape[0])
        x_rec = []
        for i in range(self.t_total):
            rnn_in = torch.cat([x_p, z], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            x_i = self.d_out(h)
            x_rec.append(x_i)
            x_p = x_i
        x_rec = torch.stack(x_rec)
        return x_rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(z)

    def step(self, model):
        pass


def run_batch_vae_rec():
    pass


def run_epoch_vae_rec(cfg, dataset, model, loss_func, optimizer, device, dtype=torch.float, scheduler=None, mode='train', options=dict()):
    t_s = time.time()
    train_losses = 0
    total_num_batch = 0
    if mode == 'train':
        generator = dataset.sampling_generator(batch_size=cfg.batch_size, num_samples=cfg.num_samples)
    elif mode == 'mode':
        generator = dataset.iter_generator(batch_size=cfg.batch_size)
    pbar = tqdm(generator)
    for data in pbar:
        traj_np = data['traj']
        if torch.is_tensor(traj_np):
            traj_x = traj_np.type(dtype).permute(1, 0, 2).contiguous()
        else:
            traj_x = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        X_r, mu, logvar = model(traj_x)
        loss, losses = loss_func(cfg, X_r=X_r, X=traj_x, mu=mu, logvar=logvar)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        train_losses += losses
        total_num_batch += 1
        pbar.set_description('Runing Loss {:.3f}'.format(np.mean(losses)))
    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_batch
    return train_losses, dt


def get_models(cfg, iter=-1):
    model_specs = cfg.model_specs
    data_specs = cfg.data_specs
    model_name = model_specs['model_name']
    traj_dim = data_specs['traj_dim']
    t_total = data_specs['t_total']
    run_epoch = None
    run_batch = None
    model = None
    if model_name == 'VAErec':
        model = VAErec(traj_dim, t_total, model_specs)
        run_epoch = run_epoch_vae_rec
        run_batch = run_batch_vae_rec
    if model_name == 'VAErecV2':
        model = VAErecV2(traj_dim, t_total, model_specs)
        run_epoch = run_epoch_vae_rec
        run_batch = run_batch_vae_rec
    if iter > 0:
        cp_path = cfg.model_path % iter
        None
        model_cp = pickle.load(open(cp_path, 'rb'))
        model.load_state_dict(model_cp['model_dict'])
    elif iter == -1:
        pass
    elif iter == -2:
        cp_path = sorted(glob.glob(osp.join(cfg.model_dir, '*')), reverse=True)[0]
        None
        model_cp = pickle.load(open(cp_path, 'rb'))
        model.load_state_dict(model_cp['model_dict'])
    return model, run_epoch, run_batch


class MEVA(nn.Module):

    def __init__(self, seqlen, batch_size=64, n_layers=1, hidden_size=2048, add_linear=False, bidirectional=False, use_residual=True, cfg='vae_rec_1'):
        super(MEVA, self).__init__()
        self.vae_cfg = vae_cfg = Config(cfg)
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.vae_model, _, _ = get_models(vae_cfg, iter=-2)
        for param in self.vae_model.parameters():
            param.requires_grad = False
        self.feat_encoder = TemporalEncoder(n_layers=n_layers, hidden_size=hidden_size, bidirectional=bidirectional, add_linear=add_linear, use_residual=use_residual)
        vae_hidden_size = 512
        self.motion_encoder = TemporalEncoder(n_layers=n_layers, hidden_size=512, bidirectional=bidirectional, add_linear=True, output_size=vae_hidden_size, use_residual=False)
        fc1 = nn.Linear(vae_hidden_size, 256)
        act = nn.Tanh()
        fc2 = nn.Linear(256, 144)
        self.vae_init_mlp = nn.Sequential(fc1, act, fc2)
        self.regressor = Regressor()
        mean_params = np.load(SMPL_MEAN_PARAMS)
        self.first_in_flag = True
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
        self.set_gender()

    def set_gender(self, gender='neutral', use_smplx=False):
        self.regressor.set_gender(gender, use_smplx)

    def forward(self, input, J_regressor=None):
        batch_size, seqlen = input.shape[:2]
        feature = self.feat_encoder(input)
        motion_z = self.motion_encoder(feature).mean(dim=1)
        if self.vae_cfg.model_specs['model_name'] == 'VAErec':
            vae_init_pose = self.vae_init_mlp(motion_z)
            X_r = self.vae_model.decode(vae_init_pose[None, :, :], motion_z)
        elif self.vae_cfg.model_specs['model_name'] == 'VAErecV2':
            X_r = self.vae_model.decode(motion_z)
        X_r = X_r.permute(1, 0, 2)[:, :seqlen, :]
        feature = feature.reshape(-1, feature.size(-1))
        init_pose = X_r.reshape(-1, X_r.shape[-1])
        smpl_output = self.regressor(feature, J_regressor=J_regressor, init_pose=init_pose)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
        return smpl_output


class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class HMR(nn.Module):
    """
    SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        return xf

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_features=False):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{'theta': torch.cat([pred_cam, pose, pred_shape], dim=1), 'verts': pred_vertices, 'kp_2d': pred_keypoints_2d, 'kp_3d': pred_joints}]
        if return_features:
            return xf, output
        else:
            return output


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    """
        Inputs:
            disc_value: N x 25
    """
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class VIBELoss(nn.Module):

    def __init__(self, e_loss_weight=60.0, e_3d_loss_weight=30.0, e_pose_loss_weight=1.0, e_shape_loss_weight=0.001, device='cuda'):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.device = device
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(self, generator_outputs, data_2d, data_3d):
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        flatten = lambda x: x.reshape(-1)
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x], 0)
        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
        real_2d = reduce(real_2d)
        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])
        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)
        total_predict_thetas = accumulate_thetas(generator_outputs)
        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)
        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)
        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1.0, gt_weight=1.0) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]
        loss_dict = {'loss_kp_2d': loss_kp_2d, 'loss_kp_3d': loss_kp_3d}
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose
        gen_loss = torch.stack(list(loss_dict.values())).sum()
        return gen_loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.0)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.0)
        return loss_regr_pose, loss_regr_betas


pi = torch.Tensor([3.141592653589793])


def rad2deg(tensor):
    """Function that converts angles from radians to degrees.

    See :class:`~torchgeometry.RadToDeg` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return 180.0 * tensor / pi.type(tensor.dtype)


class RadToDeg(nn.Module):
    """Creates an object that converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.RadToDeg()(input)
    """

    def __init__(self):
        super(RadToDeg, self).__init__()

    def forward(self, input):
        return rad2deg(input)


def deg2rad(tensor):
    """Function that converts angles from degrees to radians.

    See :class:`~torchgeometry.DegToRad` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(tensor)))
    return tensor * pi.type(tensor.dtype) / 180.0


class DegToRad(nn.Module):
    """Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.DegToRad()(input)
    """

    def __init__(self):
        super(DegToRad, self).__init__()

    def forward(self, input):
        return deg2rad(input)


def convert_points_from_homogeneous(points):
    """Function that converts points from homogeneous to Euclidean space.

    See :class:`~torchgeometry.ConvertPointsFromHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not torch.is_tensor(points):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    return points[..., :-1] / points[..., -1:]


class ConvertPointsFromHomogeneous(nn.Module):
    """Creates a transformation that converts points from homogeneous to
    Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N-1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsFromHomogeneous()
        >>> output = transform(input)  # BxNx2
    """

    def __init__(self):
        super(ConvertPointsFromHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_from_homogeneous(input)


def convert_points_to_homogeneous(points):
    """Function that converts points from Euclidean to homogeneous space.

    See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not torch.is_tensor(points):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(points)))
    if len(points.shape) < 2:
        raise ValueError('Input must be at least a 2D tensor. Got {}'.format(points.shape))
    return nn.functional.pad(points, (0, 1), 'constant', 1.0)


class ConvertPointsToHomogeneous(nn.Module):
    """Creates a transformation to convert points from Euclidean to
    homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N+1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsToHomogeneous()
        >>> output = transform(input)  # BxNx4
    """

    def __init__(self):
        super(ConvertPointsToHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_to_homogeneous(input)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CMLP,
     lambda: ([], {'input_dim': 4, 'cond_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvertPointsFromHomogeneous,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvertPointsToHomogeneous,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DegToRad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ERDNet,
     lambda: ([], {'state_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNet,
     lambda: ([], {'out_dim': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (RNN,
     lambda: ([], {'input_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (RadToDeg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet,
     lambda: ([], {'out_dim': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (TemporalBlock,
     lambda: ([], {'n_inputs': 4, 'n_outputs': 4, 'kernel_size': 4, 'stride': 1, 'dilation': 1, 'dropout': 0.5, 'causal': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TemporalConvNet,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {})),
    (TemporalEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 2048])], {})),
]

