
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


import random


import torch


import numpy as np


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch import nn


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


import copy


import logging


class AdaIN(nn.Module):

    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.to_latent = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(num_features, latent_dim, 1, 1, 0), nn.LeakyReLU(0.2))
        self.inject = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2), nn.Linear(latent_dim, num_features * 2))
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        s = self.to_latent(s).squeeze(-1).squeeze(-1)
        h = self.inject(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class SpatialConv(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), padding_mode='reflect', stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class adaptive_style(nn.Module):

    def __init__(self, in_ch):
        super(adaptive_style, self).__init__()
        self.in_ch = in_ch
        self.f = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.g = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.h = nn.Conv2d(in_ch, in_ch, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.k = nn.Conv2d(in_ch, in_ch, (1, 1))

    def forward(self, x, s_sty, return_nl_map=False):
        """
            x: (n, c, t1, v)
            s_sty: (n, c, t2, v)
        """
        b = s_sty.shape[0]
        F = self.f(nn.functional.instance_norm(x))
        G = self.g(nn.functional.instance_norm(s_sty))
        H = self.h(s_sty)
        F = F.view(b, self.in_ch, -1).permute(0, 2, 1)
        G = G.view(b, self.in_ch, -1)
        S = torch.bmm(F, G)
        S = self.sm(S)
        H = H.view(b, self.in_ch, -1)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(x.size())
        O = self.k(O)
        O += x
        if return_nl_map:
            return O, S
        return O


class BPStyleNet(nn.Module):

    def __init__(self, style_dim, in_channels, out_channels, kernel_size, stride=1, activation='lrelu'):
        super().__init__()
        assert len(kernel_size) == 2
        self.adain_leftleg = AdaIN(style_dim, in_channels)
        self.adain_rightleg = AdaIN(style_dim, in_channels)
        self.adain_spine = AdaIN(style_dim, in_channels)
        self.adain_leftarm = AdaIN(style_dim, in_channels)
        self.adain_rightarm = AdaIN(style_dim, in_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        padding = (kernel_size[0] - 1) // 2, 0
        self.gcn1 = SpatialConv(in_channels, in_channels, kernel_size[1], t_kernel_size=1)
        self.tcn1 = nn.Conv2d(in_channels, in_channels, (kernel_size[0], 1), (stride, 1), padding, padding_mode='reflect')
        self.astyle_leftleg = adaptive_style(in_channels)
        self.astyle_rightleg = adaptive_style(in_channels)
        self.astyle_spine = adaptive_style(in_channels)
        self.astyle_leftarm = adaptive_style(in_channels)
        self.astyle_rightarm = adaptive_style(in_channels)
        self.gcn2 = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size=1)
        self.tcn2 = nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding, padding_mode='reflect')

    def forward(self, x, s_leftleg, s_rightleg, s_spine, s_leftarm, s_rightarm, A):
        if A.shape[-1] == 21:
            idx_leftleg = [1, 2, 3, 4]
            idx_rightleg = [5, 6, 7, 8]
            idx_spine = [0, 9, 10, 11, 12]
            idx_leftarm = [13, 14, 15, 16]
            idx_rightarm = [17, 18, 19, 20]
        elif A.shape[-1] == 10:
            idx_leftleg = [0, 1]
            idx_rightleg = [2, 3]
            idx_spine = [4, 5]
            idx_leftarm = [6, 7]
            idx_rightarm = [8, 9]
        elif A.shape[-1] == 5:
            idx_leftleg = [0]
            idx_rightleg = [1]
            idx_spine = [2]
            idx_leftarm = [3]
            idx_rightarm = [4]
        else:
            assert A.shape[-1] == 21 or 10 or 5, 'Graph is wrong!!'
        x_leftleg = x[..., idx_leftleg]
        s_leftleg = s_leftleg[..., idx_leftleg]
        x_rightleg = x[..., idx_rightleg]
        s_rightleg = s_rightleg[..., idx_rightleg]
        x_spine = x[..., idx_spine]
        s_spine = s_spine[..., idx_spine]
        x_leftarm = x[..., idx_leftarm]
        s_leftarm = s_leftarm[..., idx_leftarm]
        x_rightarm = x[..., idx_rightarm]
        s_rightarm = s_rightarm[..., idx_rightarm]
        x_leftleg = self.adain_leftleg(x_leftleg, s_leftleg)
        x_rightleg = self.adain_rightleg(x_rightleg, s_rightleg)
        x_spine = self.adain_spine(x_spine, s_spine)
        x_leftarm = self.adain_leftarm(x_leftarm, s_leftarm)
        x_rightarm = self.adain_rightarm(x_rightarm, s_rightarm)
        x = torch.cat((x_leftleg, x_rightleg, x_spine, x_leftarm, x_rightarm), -1)
        if A.shape[-1] == 21:
            x = torch.cat((x[..., 8:9], x[..., 0:8], x[..., 9:]), -1)
        if self.activation:
            x = self.activation(x)
        x, _ = self.gcn1(x, A)
        x = self.tcn1(x)
        x_leftleg = x[..., idx_leftleg]
        x_rightleg = x[..., idx_rightleg]
        x_spine = x[..., idx_spine]
        x_leftarm = x[..., idx_leftarm]
        x_rightarm = x[..., idx_rightarm]
        x_leftleg = self.astyle_leftleg(x_leftleg, s_leftleg)
        x_rightleg = self.astyle_rightleg(x_rightleg, s_rightleg)
        x_spine = self.astyle_spine(x_spine, s_spine)
        x_leftarm = self.astyle_leftarm(x_leftarm, s_leftarm)
        x_rightarm = self.astyle_rightarm(x_rightarm, s_rightarm)
        x = torch.cat((x_leftleg, x_rightleg, x_spine, x_leftarm, x_rightarm), -1)
        if x.shape[-1] == 21:
            x = torch.cat((x[..., 8:9], x[..., 0:8], x[..., 9:]), -1)
        x, _ = self.gcn2(x, A)
        x = self.tcn2(x)
        return x


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    AD = np.dot(A, Dn)
    return AD


class Graph_Bodypart:

    def __init__(self, layout='cmu', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'cmu':
            self.num_node = 5
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 2), (1, 2), (3, 2), (4, 2)]
            self.edge = self_link + neighbor_link
            self.center = 2

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class Graph_Joint:

    def __init__(self, layout='cmu', strategy='uniform', max_hop=2, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'cmu':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19]
            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]
            self.edge = self_link + neighbor_link
            self.center = 0

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class Graph_Mid:

    def __init__(self, layout='cmu', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'cmu':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 0), (4, 2), (4, 5), (4, 6), (4, 8), (0, 1), (2, 3), (6, 7), (8, 9)]
            self.edge = self_link + neighbor_link
            self.center = 4

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class ResBPStyleNet(nn.Module):

    def __init__(self, style_dim, dim_in, dim_out, kernel_size, stride, activation='relu'):
        super(ResBPStyleNet, self).__init__()
        self.res = nn.ModuleList()
        self.res += [BPStyleNet(style_dim, dim_in, dim_in, kernel_size=kernel_size, stride=stride, activation=activation)]
        self.res += [BPStyleNet(style_dim, dim_in, dim_out, kernel_size=kernel_size, stride=stride, activation='none')]
        if dim_in == dim_out and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=(stride, 1))

    def forward(self, x, s_leftleg, s_rightleg, s_spine, s_leftarm, s_rightarm, A):
        x_org = self.shortcut(x)
        for i, layer in enumerate(self.res):
            x = layer(x, s_leftleg, s_rightleg, s_spine, s_leftarm, s_rightarm, A)
        out = x_org + 0.1 * x
        return out


class UnpoolBodypartToMid(nn.Module):

    def __init__(self):
        super().__init__()
        self.LeftLeg = [0, 1, 4]
        self.RightLeg = [2, 3, 4]
        self.Spine = [4, 5]
        self.LeftArm = [6, 7, 4]
        self.RightArm = [8, 9, 4]
        nbody = 5
        nmid = 10
        weight = torch.zeros(nbody, nmid, dtype=torch.float32, requires_grad=False)
        weight[0, self.LeftLeg] = 1.0
        weight[1, self.RightLeg] = 1.0
        weight[2, self.Spine] = 1.0
        weight[3, self.LeftArm] = 1.0
        weight[4, self.RightArm] = 1.0
        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))
        return x


class UnpoolMidToJoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.LeftHip = [0, 1, 2]
        self.LeftLeg = [2, 3, 4]
        self.RightHip = [0, 5, 6]
        self.RightLeg = [6, 7, 8]
        self.Back = [0, 9, 10]
        self.Neck = [10, 11, 12]
        self.LeftShoulder = [10, 13, 14]
        self.LeftArm = [14, 15, 16]
        self.RightShoulder = [10, 17, 18]
        self.RightArm = [18, 19, 20]
        nmid = 10
        njoints = 21
        weight = torch.zeros(nmid, njoints, dtype=torch.float32, requires_grad=False)
        weight[0, self.LeftHip] = 1.0
        weight[1, self.LeftLeg] = 1.0
        weight[2, self.RightHip] = 1.0
        weight[3, self.RightLeg] = 1.0
        weight[4, self.Back] = 1.0
        weight[5, self.Neck] = 1.0
        weight[6, self.LeftShoulder] = 1.0
        weight[7, self.LeftArm] = 1.0
        weight[8, self.RightShoulder] = 1.0
        weight[9, self.RightArm] = 1.0
        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))
        return x


class Decoder(nn.Module):

    def __init__(self, channels, out_channels, latent_dim, graph_cfg, edge_importance_weighting=True):
        super().__init__()
        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])
        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)
        spatial_kernel_size_b = self.A_b.size(0)
        spatial_kernel_size_m = self.A_m.size(0)
        spatial_kernel_size_j = self.A_j.size(0)
        ks_bottleneck = 3, spatial_kernel_size_b
        ks_bodypart = 5, spatial_kernel_size_b
        ks_mid = 5, spatial_kernel_size_m
        ks_joint = 7, spatial_kernel_size_j
        self.bottleneck = ResBPStyleNet(latent_dim, channels, channels, kernel_size=ks_bottleneck, stride=1, activation='lrelu')
        self.bodypart = BPStyleNet(latent_dim, channels, channels // 2, kernel_size=ks_bodypart, stride=1, activation='lrelu')
        channels //= 2
        self.up_BodypartToMid = UnpoolBodypartToMid()
        self.up_temp1 = F.interpolate
        self.mid = BPStyleNet(latent_dim, channels, channels // 2, kernel_size=ks_mid, stride=1, activation='lrelu')
        channels //= 2
        self.up_MidToJoint = UnpoolMidToJoint()
        self.up_temp2 = F.interpolate
        self.joint = BPStyleNet(latent_dim, channels, channels // 2, kernel_size=ks_joint, stride=1, activation='lrelu')
        channels //= 2
        self.to_mot = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(channels, out_channels, (1, 1)))
        if edge_importance_weighting:
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
        else:
            self.edge_importance_bt = 1
            self.edge_importance_b = 1
            self.edge_importance_m = 1
            self.edge_importance_j = 1

    def forward(self, x, sty_leftleg, sty_rightleg, sty_spine, sty_leftarm, sty_rightarm):
        """
            x: (n, c, t, v)
            sty_features: [(n, 4c, 5, t), ..., (n, c, 21, 4t)] 
        """
        x = self.bottleneck(x, sty_leftleg[0], sty_rightleg[0], sty_spine[0], sty_leftarm[0], sty_rightarm[0], self.A_b * self.edge_importance_bt)
        x = self.bodypart(x, sty_leftleg[1], sty_rightleg[1], sty_spine[1], sty_leftarm[1], sty_rightarm[1], self.A_b * self.edge_importance_b)
        x = self.up_BodypartToMid(x)
        x = self.up_temp1(x, scale_factor=(2, 1), mode='nearest')
        x = self.mid(x, sty_leftleg[2], sty_rightleg[2], sty_spine[2], sty_leftarm[2], sty_rightarm[2], self.A_m * self.edge_importance_m)
        x = self.up_MidToJoint(x)
        x = self.up_temp2(x, scale_factor=(2, 1), mode='nearest')
        x = self.joint(x, sty_leftleg[3], sty_rightleg[3], sty_spine[3], sty_leftarm[3], sty_rightarm[3], self.A_j * self.edge_importance_j)
        x = self.to_mot(x)
        return x.permute(0, 1, 3, 2).contiguous()


class PoolJointToMid(nn.Module):

    def __init__(self):
        super().__init__()
        self.LeftHip = [0, 1, 2]
        self.LeftLeg = [2, 3, 4]
        self.RightHip = [0, 5, 6]
        self.RightLeg = [6, 7, 8]
        self.Back = [0, 9, 10]
        self.Neck = [10, 11, 12]
        self.LeftShoulder = [10, 13, 14]
        self.LeftArm = [14, 15, 16]
        self.RightShoulder = [10, 17, 18]
        self.RightArm = [18, 19, 20]
        njoints = 21
        nmid = 10
        weight = torch.zeros(njoints, nmid, dtype=torch.float32, requires_grad=False)
        weight[self.LeftHip, 0] = 1.0
        weight[self.LeftLeg, 1] = 1.0
        weight[self.RightHip, 2] = 1.0
        weight[self.RightLeg, 3] = 1.0
        weight[self.Back, 4] = 1.0
        weight[self.Neck, 5] = 1.0
        weight[self.LeftShoulder, 6] = 1.0
        weight[self.LeftArm, 7] = 1.0
        weight[self.RightShoulder, 8] = 1.0
        weight[self.RightArm, 9] = 1.0
        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))
        return x


class PoolMidToBodypart(nn.Module):

    def __init__(self):
        super().__init__()
        self.LeftLeg = [0, 1, 4]
        self.RightLeg = [2, 3, 4]
        self.Spine = [4, 5]
        self.LeftArm = [6, 7, 4]
        self.RightArm = [8, 9, 4]
        nmid = 10
        nbody = 5
        weight = torch.zeros(nmid, nbody, dtype=torch.float32, requires_grad=False)
        weight[self.LeftLeg, 0] = 1.0
        weight[self.RightLeg, 1] = 1.0
        weight[self.Spine, 2] = 1.0
        weight[self.LeftArm, 3] = 1.0
        weight[self.RightArm, 4] = 1.0
        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))
        return x


class StgcnBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm='none', activation='relu'):
        super().__init__()
        assert len(kernel_size) == 2
        padding = (kernel_size[0] - 1) // 2, 0
        norm_dim = in_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding, padding_mode='reflect')

    def forward(self, x, A):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        return x


class ResStgcnBlock(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, norm='in', activation='lrelu'):
        super(ResStgcnBlock, self).__init__()
        self.res = nn.ModuleList()
        self.res += [StgcnBlock(dim_in, dim_in, kernel_size=kernel_size, stride=stride, norm=norm, activation=activation)]
        self.res += [StgcnBlock(dim_in, dim_out, kernel_size=kernel_size, stride=stride, norm=norm, activation='none')]
        if dim_in == dim_out and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=(stride, 1))

    def forward(self, x, A):
        x_org = self.shortcut(x)
        for i, layer in enumerate(self.res):
            x = layer(x, A)
        out = x_org + 0.1 * x
        return out


class Encoder_con(nn.Module):

    def __init__(self, in_channels, channels, graph_cfg, edge_importance_weighting=True):
        super().__init__()
        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])
        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)
        spatial_kernel_size_j = self.A_j.size(0)
        spatial_kernel_size_m = self.A_m.size(0)
        spatial_kernel_size_b = self.A_b.size(0)
        ks_joint = 7, spatial_kernel_size_j
        ks_mid = 5, spatial_kernel_size_m
        ks_bodypart = 5, spatial_kernel_size_b
        ks_bottleneck = 3, spatial_kernel_size_b
        self.from_mot = nn.Conv2d(in_channels, channels, (1, 1))
        self.joint = StgcnBlock(channels, 2 * channels, kernel_size=ks_joint, stride=1, norm='in', activation='lrelu')
        channels *= 2
        self.down_JointToMid = PoolJointToMid()
        self.down_temp1 = F.avg_pool2d
        self.mid = StgcnBlock(channels, 2 * channels, kernel_size=ks_mid, stride=1, norm='in', activation='lrelu')
        channels *= 2
        self.down_MidToBodypart = PoolMidToBodypart()
        self.down_temp2 = F.avg_pool2d
        self.bodypart = StgcnBlock(channels, 2 * channels, kernel_size=ks_bodypart, stride=1, norm='in', activation='lrelu')
        channels *= 2
        self.bottleneck = ResStgcnBlock(channels, channels, kernel_size=ks_bottleneck, stride=1, norm='in', activation='lrelu')
        self.output_channels = channels
        if edge_importance_weighting:
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
        else:
            self.edge_importance_j = 1
            self.edge_importance_m = 1
            self.edge_importance_b = 1
            self.edge_importance_bt = 1

    def forward(self, x):
        latents_features = []
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.from_mot(x)
        x = self.joint(x, self.A_j * self.edge_importance_j)
        latents_features.append(x)
        x = self.down_JointToMid(x)
        x = self.down_temp1(x, kernel_size=(2, 1))
        x = self.mid(x, self.A_m * self.edge_importance_m)
        latents_features.append(x)
        x = self.down_MidToBodypart(x)
        x = self.down_temp2(x, kernel_size=(2, 1))
        x = self.bodypart(x, self.A_b * self.edge_importance_b)
        latents_features.append(x)
        x = self.bottleneck(x, self.A_b * self.edge_importance_bt)
        latents_features.append(x)
        return latents_features


class Encoder_sty(nn.Module):

    def __init__(self, in_channels, channels, graph_cfg, edge_importance_weighting=True):
        super().__init__()
        self.graph_j = Graph_Joint(**graph_cfg['joint'])
        self.graph_m = Graph_Mid(**graph_cfg['mid'])
        self.graph_b = Graph_Bodypart(**graph_cfg['bodypart'])
        A_j = torch.tensor(self.graph_j.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        A_m = torch.tensor(self.graph_m.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_m', A_m)
        A_b = torch.tensor(self.graph_b.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)
        spatial_kernel_size_j = self.A_j.size(0)
        spatial_kernel_size_m = self.A_m.size(0)
        spatial_kernel_size_b = self.A_b.size(0)
        ks_joint = 7, spatial_kernel_size_j
        ks_mid = 5, spatial_kernel_size_m
        ks_bodypart = 5, spatial_kernel_size_b
        ks_bottleneck = 3, spatial_kernel_size_b
        self.from_mot = nn.Conv2d(in_channels, channels, (1, 1))
        self.joint = StgcnBlock(channels, 2 * channels, kernel_size=ks_joint, stride=1, norm='none', activation='lrelu')
        channels *= 2
        self.down_JointToMid = PoolJointToMid()
        self.down_temp1 = F.avg_pool2d
        self.mid = StgcnBlock(channels, 2 * channels, kernel_size=ks_mid, stride=1, norm='none', activation='lrelu')
        channels *= 2
        self.down_MidToBodypart = PoolMidToBodypart()
        self.down_temp2 = F.avg_pool2d
        self.bodypart = StgcnBlock(channels, 2 * channels, kernel_size=ks_bodypart, stride=1, norm='none', activation='lrelu')
        channels *= 2
        self.bottleneck = ResStgcnBlock(channels, channels, kernel_size=ks_bottleneck, stride=1, norm='none', activation='lrelu')
        self.output_channels = channels
        if edge_importance_weighting:
            self.edge_importance_j = nn.Parameter(torch.ones(self.A_j.size()))
            self.edge_importance_m = nn.Parameter(torch.ones(self.A_m.size()))
            self.edge_importance_b = nn.Parameter(torch.ones(self.A_b.size()))
            self.edge_importance_bt = nn.Parameter(torch.ones(self.A_b.size()))
        else:
            self.edge_importance_j = 1
            self.edge_importance_m = 1
            self.edge_importance_b = 1
            self.edge_importance_bt = 1

    def forward(self, x):
        latents_features = []
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.from_mot(x)
        x = self.joint(x, self.A_j * self.edge_importance_j)
        latents_features.append(x)
        x = self.down_JointToMid(x)
        x = self.down_temp1(x, kernel_size=(2, 1))
        x = self.mid(x, self.A_m * self.edge_importance_m)
        latents_features.append(x)
        x = self.down_MidToBodypart(x)
        x = self.down_temp2(x, kernel_size=(2, 1))
        x = self.bodypart(x, self.A_b * self.edge_importance_b)
        latents_features.append(x)
        x = self.bottleneck(x, self.A_b * self.edge_importance_bt)
        latents_features.append(x)
        return latents_features


def mixing_styles(style_cnt, style_cls, prob):
    bdy_idx = [0, 1, 2, 3, 4]
    bdy_part_idx = [[0, 1], [2], [3, 4]]
    bdy_part_select, bdy_part_not_select = [], []
    if prob > 0 and random.random() < prob:
        n_choice = random.randint(1, 3)
        bdy_part_select = random.sample(bdy_part_idx, n_choice)
        bdy_part_select = sum(bdy_part_select, [])
        bdy_part_select.sort()
        bdy_part_not_select = [x for x in bdy_idx if x not in bdy_part_select]
        if len(bdy_part_select) == 5:
            istyles = [style_cnt] * 5
        else:
            istyles = [None] * 5
            for i in range(len(bdy_part_select)):
                istyles[bdy_part_select[i]] = style_cnt
            for j in range(len(bdy_part_not_select)):
                istyles[bdy_part_not_select[j]] = style_cls
    else:
        istyles = [style_cls] * 5
    return istyles, bdy_part_select


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        enc_in_dim = config['enc_in_dim']
        enc_nf = config['enc_nf']
        latent_dim = config['latent_dim']
        graph_cfg = config['graph']
        self.enc_content = Encoder_con(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.enc_style = Encoder_sty(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.dec = Decoder(self.enc_content.output_channels, enc_in_dim, latent_dim=latent_dim, graph_cfg=graph_cfg)
        self.apply(self._init_weights)

    def forward(self, xa, xb, phase='train'):
        c_xa = self.enc_content(xa)
        c_xb = self.enc_content(xb)
        s_xa = self.enc_style(xa)
        s_xb = self.enc_style(xb)
        mixing_prob = 0.4 if phase == 'train' else 0.0
        s_mix, bdy_part_select = mixing_styles(s_xa, s_xb, mixing_prob)
        xab = self.dec(c_xa[-1], s_mix[0][::-1], s_mix[1][::-1], s_mix[2][::-1], s_mix[3][::-1], s_mix[4][::-1])
        xaa = self.dec(c_xa[-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1])
        xbb = self.dec(c_xb[-1], s_xb[::-1], s_xb[::-1], s_xb[::-1], s_xb[::-1], s_xb[::-1])
        c_xab = self.enc_content(xab)
        xaba = self.dec(c_xab[-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1], s_xa[::-1])
        if len(bdy_part_select) == 0:
            s_xab = self.enc_style(xab)
            xabb = self.dec(c_xb[-1], s_xab[::-1], s_xab[::-1], s_xab[::-1], s_xab[::-1], s_xab[::-1])
        else:
            xabb = xb
        return xaa, xbb, xab, xaba, xabb

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class RAdam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)
        return loss


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and key in f and '.pt' in f]
    if gen_models is None or len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


logger = logging.getLogger(__name__)


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)


class Trainer(nn.Module):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.gen = Generator(config['model']['gen'])
        self.gen_ema = copy.deepcopy(self.gen)
        self.model_dir = config['model_dir']
        self.config = config
        lr_gen = config['lr_gen']
        gen_params = list(self.gen.parameters())
        self.gen_opt = RAdam([p for p in gen_params if p.requires_grad], lr=lr_gen, weight_decay=config['weight_decay'])
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.gen = nn.DataParallel(self.gen)
            self.gen_ema = nn.DataParallel(self.gen_ema)

    def train(self, loader, wirter):
        config = self.config

        def run_epoch(epoch):
            self.gen.train()
            pbar = tqdm(enumerate(zip(loader['train_src'], loader['train_tar'])), total=len(loader['train_src']))
            for it, (con_data, sty_data) in pbar:
                gen_loss_total, gen_loss_dict = self.compute_gen_loss(con_data, sty_data)
                self.gen_opt.zero_grad()
                gen_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.gen_opt.step()
                update_average(self.gen_ema, self.gen)
                log = 'Epoch [%i/%i], ' % (epoch + 1, config['max_epochs'])
                all_losses = dict()
                for loss in [gen_loss_dict]:
                    for key, value in loss.items():
                        if key.find('total') > -1:
                            all_losses[key] = value
                log += ' '.join([('%s: [%.2f]' % (key, value)) for key, value in all_losses.items()])
                pbar.set_description(log)
                if (it + 1) % config['log_every'] == 0:
                    for k, v in gen_loss_dict.items():
                        wirter.add_scalar(k, v, epoch * len(loader['train_src']) + it)
        for epoch in range(config['max_epochs']):
            run_epoch(epoch)
            if (epoch + 1) % config['save_every'] == 0:
                self.save_checkpoint(epoch + 1)

    def compute_gen_loss(self, xa_data, xb_data):
        config = self.config
        xa = xa_data['motion']
        xb = xb_data['motion']
        xaa, xbb, xab, xaba, xabb = self.gen(xa, xb)
        loss_recon = F.l1_loss(xaa, xa) + F.l1_loss(xbb, xb)
        loss_cyc_con = F.l1_loss(xaba, xa)
        loss_cyc_sty = F.l1_loss(xabb, xb)
        loss_sm_rec = F.l1_loss(xaa[..., :-1] - xaa[..., 1:], xa[..., :-1] - xa[..., 1:]) + F.l1_loss(xbb[..., :-1] - xbb[..., 1:], xb[..., :-1] - xb[..., 1:])
        loss_sm_cyc = F.l1_loss(xaba[..., :-1] - xaba[..., 1:], xa[..., :-1] - xa[..., 1:]) + F.l1_loss(xabb[..., :-1] - xabb[..., 1:], xb[..., :-1] - xb[..., 1:])
        l_total = config['rec_w'] * loss_recon + config['cyc_con_w'] * loss_cyc_con + config['cyc_sty_w'] * loss_cyc_sty + config['sm_rec_w'] * loss_sm_rec + config['sm_cyc_w'] * loss_sm_cyc
        l_dict = {'loss_total': l_total, 'loss_recon': loss_recon, 'loss_cyc_con': loss_cyc_con, 'loss_cyc_sty': loss_cyc_sty, 'loss_sm_rec': loss_sm_rec, 'loss_sm_cyc': loss_sm_cyc}
        return l_total, l_dict

    @torch.no_grad()
    def test(self, xa, xb):
        config = self.config
        self.gen_ema.eval()
        xaa, xbb, xab, xaba, xabb = self.gen_ema(xa, xb, phase='test')
        loss_recon = F.l1_loss(xaa, xa) + F.l1_loss(xbb, xb)
        loss_cyc_con = F.l1_loss(xaba, xa)
        loss_cyc_sty = F.l1_loss(xabb, xb)
        loss_sm_rec = F.l1_loss(xaa[..., :-1] - xaa[..., 1:], xa[..., :-1] - xa[..., 1:]) + F.l1_loss(xbb[..., :-1] - xbb[..., 1:], xb[..., :-1] - xb[..., 1:])
        loss_sm_cyc = F.l1_loss(xaba[..., :-1] - xaba[..., 1:], xa[..., :-1] - xa[..., 1:]) + F.l1_loss(xabb[..., :-1] - xabb[..., 1:], xb[..., :-1] - xb[..., 1:])
        l_total = config['rec_w'] * loss_recon + config['cyc_con_w'] * loss_cyc_con + config['cyc_sty_w'] * loss_cyc_sty + config['sm_rec_w'] * loss_sm_rec + config['sm_cyc_w'] * loss_sm_cyc
        l_dict = {'loss_total': l_total, 'loss_recon': loss_recon, 'loss_cyc_con': loss_cyc_con, 'loss_cyc_sty': loss_cyc_sty, 'loss_sm_rec': loss_sm_rec, 'loss_sm_cyc': loss_sm_cyc}
        out_dict = {'recon_con': xaa, 'stylized': xab, 'con_gt': xa, 'sty_gt': xb}
        return out_dict, l_dict

    def save_checkpoint(self, epoch):
        gen_path = os.path.join(self.model_dir, 'gen_%03d.pt' % epoch)
        raw_gen = self.gen.module if hasattr(self.gen, 'module') else self.gen
        raw_gen_ema = self.gen_ema.module if hasattr(self.gen_ema, 'module') else self.gen_ema
        logger.info('saving %s', gen_path)
        torch.save({'gen': raw_gen.state_dict(), 'gen_ema': raw_gen_ema.state_dict()}, gen_path)
        None

    def load_checkpoint(self, model_path=None):
        if not model_path:
            model_dir = self.model_dir
            model_path = get_model_list(model_dir, 'gen')
        state_dict = torch.load(model_path, map_location=self.device)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_ema.load_state_dict(state_dict['gen_ema'])
        epochs = int(model_path[-6:-3])
        None
        return epochs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaIN,
     lambda: ([], {'latent_dim': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (StgcnBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (adaptive_style,
     lambda: ([], {'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

