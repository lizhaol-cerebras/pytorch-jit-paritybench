
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


import logging


import numpy as np


import torch.nn.functional as F


import torch.nn as nn


import math


from torch.autograd import Variable


import time


from collections import OrderedDict


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data.sampler import Sampler


from numpy import dot


from numpy.linalg import norm


class GatingContext(nn.Module):

    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()
        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)
        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases
        gates = self.sigmoid(gates)
        activation = x * gates
        return activation


class NetVLADLoupe(nn.Module):

    def __init__(self, feature_size, max_samples, cluster_size, output_dim, gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))
        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None
        self.bn2 = nn.BatchNorm1d(output_dim)
        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = torch.matmul(vlad, self.hidden1_weights)
        vlad = self.bn2(vlad)
        if self.gating:
            vlad = self.context_gating(vlad)
        return vlad


class SOP(nn.Module):

    def __init__(self, thresh=1e-08, is_vec=True, signed_sqrt=False, do_pe=True, do_fc=False, input_dim=16, is_tuple=False):
        super(SOP, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.do_pe = do_pe
        self.sop_dim = input_dim * input_dim
        self.signed_sqrt = signed_sqrt
        self.do_fc = do_fc
        self.is_tuple = is_tuple
        cs = [4096, 2048, 1024]
        cr = self.sop_dim / cs[0]
        cs = [int(cr * x) for x in cs]
        self.fc1 = nn.Linear(cs[0], cs[1])
        self.fc2 = nn.Linear(cs[1], cs[2])

    def _so_maxpool(self, x):
        while len(x.data.shape) < 4:
            x = torch.unsqueeze(x, 0)
        batchSize, tupleLen, nFeat, dimFeat = x.data.shape
        x = torch.reshape(x, (-1, dimFeat))
        x = torch.unsqueeze(x, -1)
        x = x.matmul(x.transpose(1, 2))
        x = torch.reshape(x, (batchSize, tupleLen, nFeat, dimFeat, dimFeat))
        x = torch.max(x, 2).values
        x = torch.reshape(x, (-1, dimFeat, dimFeat))
        if self.do_pe:
            x = x.double()
            u_, s_, v_ = torch.svd(x)
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)
        x = torch.reshape(x, (batchSize, tupleLen, dimFeat, dimFeat))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self._so_maxpool(x)
        if self.is_vec:
            x = torch.reshape(x, (x.size(0), x.size(1), -1))
        x = self._l2norm(x)
        return torch.squeeze(x)


class STN3d(nn.Module):

    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()
        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):

    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride), spnn.BatchNorm(outc), spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True), spnn.BatchNorm(outc), spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride), spnn.BatchNorm(outc), spnn.ReLU(True), spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1), spnn.BatchNorm(outc))
        self.downsample = nn.Sequential() if inc == outc and stride == 1 else nn.Sequential(spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride), spnn.BatchNorm(outc))
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat([z.C[:, :3] * init_res / after_res, z.C[:, -1].view(-1, 1)], 1)
    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord
    return new_tensor


def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None or z.additional_features['idx_query'].get(x.s) is None:
        pc_hash = F.sphash(torch.cat([torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0], z.C[:, -1].int().view(-1, 1)], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps
    return new_tensor


def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(torch.cat([torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0], z.C[:, -1].int().view(-1, 1)], 1), off)
        pc_hash = F.sphash(x.C)
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat, z.C, idx_query=z.idx_query, weights=z.weights)
        new_tensor.additional_features = z.additional_features
    return new_tensor


class SPVCNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']
        self.stem = nn.Sequential(spnn.Conv3d(4, cs[0], kernel_size=3, stride=1), spnn.BatchNorm(cs[0]), spnn.ReLU(True), spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1), spnn.BatchNorm(cs[0]), spnn.ReLU(True))
        self.stage1 = nn.Sequential(BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1))
        self.stage2 = nn.Sequential(BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1), ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))
        self.stage3 = nn.Sequential(BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1), ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1), ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1))
        self.stage4 = nn.Sequential(BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1), ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1), ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1))
        self.up1 = nn.ModuleList([BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2), nn.Sequential(ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1), ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1))])
        self.up2 = nn.ModuleList([BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2), nn.Sequential(ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1), ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1))])
        self.up3 = nn.ModuleList([BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2), nn.Sequential(ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1), ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1))])
        self.up4 = nn.ModuleList([BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2), nn.Sequential(ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1), ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1))])
        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))
        self.point_transforms = nn.ModuleList([nn.Sequential(nn.Linear(cs[0], cs[4]), nn.BatchNorm1d(cs[4]), nn.ReLU(True)), nn.Sequential(nn.Linear(cs[4], cs[6]), nn.BatchNorm1d(cs[6]), nn.ReLU(True)), nn.Sequential(nn.Linear(cs[6], cs[8]), nn.BatchNorm1d(cs[8]), nn.ReLU(True))])
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)
        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)
        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)
        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        out = self.classifier(z3.F)
        return out


def spvcnn(output_dim=16):
    model = SPVCNN(num_classes=output_dim, cr=0.64, pres=0.05, vres=0.05)
    return model


class LOGG3D(nn.Module):

    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()
        self.spvcnn = spvcnn(output_dim=feature_dim)
        self.sop = SOP(signed_sqrt=False, do_fc=False, input_dim=feature_dim, is_tuple=False)

    def forward(self, x):
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.spvcnn(x)
        y = torch.split(x, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.sop(x)
        return x, y[:2]


class PointNetVLAD(nn.Module):

    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVLAD, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat, feature_transform=feature_transform, max_pool=max_pool)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64, output_dim=output_dim, gating=True, add_batch_norm=True, is_training=True)

    def forward(self, x):
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GatingContext,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NetVLADLoupe,
     lambda: ([], {'feature_size': 4, 'max_samples': 4, 'cluster_size': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SOP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

