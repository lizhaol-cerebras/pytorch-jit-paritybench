
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


import numpy as np


import torch


import torch.utils.data as data


import scipy.io as sio


from sklearn.neighbors import NearestNeighbors as nnbrs


from collections import namedtuple


from scipy.spatial import KDTree


import random


from scipy.spatial.transform import Rotation as sciR


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.batchnorm import _BatchNorm


import time


from collections import OrderedDict


from sklearn.neighbors import KDTree


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.optim as optim


class ClsSO3ConvModel(nn.Module):

    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = M.ClsOutBlockPointnet(params['outblock'])
        self.na_in = params['na']
        self.invariance = True

    def forward(self, x, rlabel=None):
        input_x = x
        x = M.preprocess_input(x, self.na_in, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x = self.outblock(x, rlabel)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


class InvSO3ConvModel(nn.Module):

    def __init__(self, params):
        super(InvSO3ConvModel, self).__init__()
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = M.InvOutBlockMVD(params['outblock'])
        self.na_in = params['na']
        self.invariance = True

    def forward(self, x):
        x = M.preprocess_input(x, self.na_in, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x = self.outblock(x)
        return x

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


class RegSO3ConvModel(nn.Module):

    def __init__(self, params):
        super(RegSO3ConvModel, self).__init__()
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
        self.outblock = M.RelSO3OutBlockR(params['outblock'])
        self.na_in = params['na']
        self.invariance = True

    def forward(self, x):
        x = torch.cat((x[:, 0], x[:, 1]), dim=0)
        x = M.preprocess_input(x, self.na_in, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        f1, f2 = torch.chunk(x.feats, 2, dim=0)
        x1, x2 = torch.chunk(x.xyz, 2, dim=0)
        confidence, quats = self.outblock(f1, f2, x1, x2)
        return confidence, quats

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


class IntraSO3ConvBlock(nn.Module):

    def __init__(self, dim_in, dim_out, norm=None, activation='relu', dropout_rate=0):
        super(IntraSO3ConvBlock, self).__init__()
        if norm is not None:
            norm = getattr(nn, norm)
        self.conv = sptk.IntraSO3Conv(dim_in, dim_out)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim__out)
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class PropagationBlock(nn.Module):

    def __init__(self, params, norm=None, activation='relu', dropout_rate=0):
        super(PropagationBlock, self).__init__()
        self.prop = sptk.KernelPropagation(**params)
        if norm is None:
            norm = nn.InstanceNorm2d
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.norm = norm(params['dim_out'], affine=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, frag, clouds):
        x = self.prop(frag, clouds)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class InterSO3ConvBlock(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, radius, sigma, n_neighbor, multiplier, kanchor=60, lazy_sample=None, norm=None, activation='relu', pooling='none', dropout_rate=0):
        super(InterSO3ConvBlock, self).__init__()
        if lazy_sample is None:
            lazy_sample = True
        if norm is not None:
            norm = getattr(nn, norm)
        pooling_method = None if pooling == 'none' else pooling
        self.conv = sptk.InterSO3Conv(dim_in, dim_out, kernel_size, stride, radius, sigma, n_neighbor, kanchor=kanchor, lazy_sample=lazy_sample, pooling=pooling_method)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        input_x = x
        inter_idx, inter_w, sample_idx, x = self.conv(x, inter_idx, inter_w)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, sample_idx, zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class SeparableSO3ConvBlock(nn.Module):

    def __init__(self, params):
        super(SeparableSO3ConvBlock, self).__init__()
        dim_in = params['dim_in']
        dim_out = params['dim_out']
        norm = getattr(nn, params['norm']) if 'norm' in params.keys() else None
        self.use_intra = params['kanchor'] > 1
        self.inter_conv = InterSO3ConvBlock(**params)
        intra_args = {'dim_in': dim_out, 'dim_out': dim_out, 'dropout_rate': params['dropout_rate'], 'activation': params['activation']}
        if self.use_intra:
            self.intra_conv = IntraSO3ConvBlock(**intra_args)
        self.stride = params['stride']
        self.skip_conv = nn.Conv2d(dim_in, dim_out, 1)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)
        self.relu = getattr(F, params['activation'])

    def forward(self, x, inter_idx, inter_w):
        """
            inter, intra conv with skip connection
        """
        skip_feature = x.feats
        inter_idx, inter_w, sample_idx, x = self.inter_conv(x, inter_idx, inter_w)
        if self.use_intra:
            x = self.intra_conv(x)
        if self.stride > 1:
            skip_feature = zptk.functional.batched_index_select(skip_feature, 2, sample_idx.long())
        skip_feature = self.skip_conv(skip_feature)
        skip_feature = self.relu(self.norm(skip_feature))
        x_out = zptk.SphericalPointCloud(x.xyz, x.feats + skip_feature, x.anchors)
        return inter_idx, inter_w, sample_idx, x_out

    def get_anchor(self):
        return torch.from_numpy(sptk.get_anchors())


class BasicSO3ConvBlock(nn.Module):

    def __init__(self, params):
        super(BasicSO3ConvBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.layer_types = []
        for param in params:
            if param['type'] == 'intra_block':
                conv = IntraSO3ConvBlock(**param['args'])
            elif param['type'] == 'inter_block':
                conv = InterSO3ConvBlock(**param['args'])
            elif param['type'] == 'separable_block':
                conv = SeparableSO3ConvBlock(param['args'])
            else:
                raise ValueError(f"No such type of SO3Conv {param['type']}")
            self.layer_types.append(param['type'])
            self.blocks.append(conv)
        self.params = params

    def forward(self, x):
        inter_idx, inter_w = None, None
        for conv, param in zip(self.blocks, self.params):
            if param['type'] in ['inter', 'inter_block', 'separable_block']:
                inter_idx, inter_w, _, x = conv(x, inter_idx, inter_w)
                if param['args']['stride'] > 1:
                    inter_idx, inter_w = None, None
            elif param['type'] in ['intra_block']:
                x = conv(x)
            else:
                raise ValueError(f"No such type of SO3Conv {param['type']}")
        return x

    def get_anchor(self):
        return torch.from_numpy(sptk.get_anchors())


class ClsOutBlockR(nn.Module):

    def __init__(self, params, norm=None):
        super(ClsOutBlockR, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']
        self.outDim = k
        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        if 'intra' in params.keys():
            self.intra = nn.ModuleList()
            self.skipconv = nn.ModuleList()
            for intraparams in params['intra']:
                conv = IntraSO3ConvBlock(**intraparams['args'])
                self.intra.append(conv)
                c_out = intraparams['args']['dim_out']
                self.skipconv.append(nn.Conv2d(c_in, c_out, 1))
                self.norm.append(nn.BatchNorm2d(c_out))
                c_in = c_out
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)
        elif self.pooling_method == 'attention2':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, c_in, 1)
        self.fc1 = nn.ModuleList()
        for c in fc:
            self.fc1.append(nn.Linear(c_in, c))
            c_in = c
        self.fc2 = nn.Linear(c_in, self.outDim)

    def forward(self, feats, label=None):
        x_out = feats
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            norm_cnt += 1
        out_feat = x_out
        x_out = x_out.mean(2, keepdim=True)
        if hasattr(self, 'intra'):
            x_in = zptk.SphericalPointCloud(None, x_out, None)
            for lid, conv in enumerate(self.intra):
                skip_feat = x_in.feats
                x_in = conv(x_in)
                norm = self.norm[norm_cnt]
                skip_feat = self.skipconv[lid](skip_feat)
                skip_feat = F.relu(norm(skip_feat))
                x_in = zptk.SphericalPointCloud(None, skip_feat + x_in.feats, None)
                norm_cnt += 1
            x_out = x_in.feats
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            x_out = x_out.mean(2).max(-1)[0]
        elif label is not None:

            def to_one_hot(label, num_class):
                """
                label: [B,...]
                return [B,...,num_class]
                """
                comp = torch.arange(num_class).long()
                for i in range(label.dim()):
                    comp = comp.unsqueeze(0)
                onehot = label.unsqueeze(-1) == comp
                return onehot.float()
            x_out = x_out.mean(2)
            label = label.squeeze()
            if label.dim() == 2:
                cdim = x_out.shape[1]
                label = label.repeat(1, 5)[:, :cdim]
            confidence = to_one_hot(label, x_out.shape[2])
            if confidence.dim() < 3:
                confidence = confidence.unsqueeze(1)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        elif self.pooling_method.startswith('attention'):
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        else:
            raise NotImplementedError(f'Pooling mode {self.pooling_method} is not implemented!')
        for linear in self.fc1:
            x_out = linear(x_out)
            x_out = F.relu(x_out)
        x_out = self.fc2(x_out)
        return x_out, out_feat.squeeze()


class ClsOutBlockPointnet(nn.Module):

    def __init__(self, params, norm=None, debug=False):
        super(ClsOutBlockPointnet, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']
        na = params['kanchor']
        self.outDim = k
        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na)
        self.norm.append(nn.BatchNorm1d(c_in))
        self.fc2 = nn.Linear(c_in, self.outDim)
        self.debug = debug

    def forward(self, x, label=None):
        x_out = x.feats
        if self.debug:
            return x_out[:, :40].mean(-1).mean(-1), None
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            norm_cnt += 1
        out_feat = x_out
        x_in = zptk.SphericalPointCloud(x.xyz, out_feat, x.anchors)
        x_out = self.pointnet(x_in)
        norm = self.norm[norm_cnt]
        norm_cnt += 1
        x_out = F.relu(norm(x_out))
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'debug':
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            x_out = x_out.max(2)[0]
        elif self.pooling_method.startswith('attention'):
            out_feat = self.attention_layer(x_out)
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        else:
            raise NotImplementedError(f'Pooling mode {self.pooling_method} is not implemented!')
        x_out = self.fc2(x_out)
        return x_out, out_feat.squeeze()


class InvOutBlockR(nn.Module):

    def __init__(self, params, norm=None):
        super(InvOutBlockR, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        if 'intra' in params.keys():
            pass
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']
        self.norm = nn.ModuleList()
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(mlp[-1], 1, 1)
        self.linear = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.InstanceNorm2d(c, affine=False))
            c_in = c

    def forward(self, feats):
        x_out = feats
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if lid != end - 1:
                norm = self.norm[lid]
                x_out = F.relu(norm(x_out))
        out_feat = x_out.mean(2)
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            x_out = x_out.mean(2).max(-1)[0]
        elif self.pooling_method == 'attention':
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            out_feat = confidence.squeeze()
        else:
            raise NotImplementedError(f'Pooling mode {self.pooling_method} is not implemented!')
        return F.normalize(x_out, p=2, dim=1), out_feat


class InvOutBlockPointnet(nn.Module):

    def __init__(self, params, norm=None):
        super(InvOutBlockPointnet, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]
        na = params['kanchor']
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_out, na)
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_out, 1, 1)

    def forward(self, x):
        x_out = self.pointnet(x)
        out_feat = x_out
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'max':
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'attention':
            attw = self.attention_layer(x_out)
            confidence = F.softmax(attw * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            confidence = confidence.squeeze()
        else:
            raise NotImplementedError(f'Pooling mode {self.pooling_method} is not implemented!')
        return F.normalize(x_out, p=2, dim=1), F.normalize(out_feat, p=2, dim=1)


class InvOutBlockMVD(nn.Module):

    def __init__(self, params, norm=None):
        super(InvOutBlockMVD, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]
        na = params['kanchor']
        self.temperature = params['temperature']
        self.attention_layer = nn.Sequential(nn.Conv2d(c_in, c_in, 1), nn.ReLU(inplace=True), nn.Conv2d(c_in, c_in, 1))
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_out, na)

    def forward(self, x):
        nb, nc, np, na = x.feats.shape
        attn = self.attention_layer(x.feats)
        attn = F.softmax(attn, dim=3)
        x_out = (x.feats * attn).sum(-1, keepdim=True)
        x_in = zptk.SphericalPointCloud(x.xyz, x_out, None)
        x_out = self.pointnet(x_in).view(nb, -1)
        return F.normalize(x_out, p=2, dim=1), attn


class SO3OutBlockR(nn.Module):

    def __init__(self, params, norm=None):
        super(SO3OutBlockR, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        self.representation = params['representation']
        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1, 1))
        self.regressor_layer = nn.Conv2d(mlp[-1], 4, (1, 1))
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            c_in = c

    def forward(self, feats):
        x_out = feats
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            x_out = F.relu(x_out)
        x_out = x_out.mean(2)
        attention_wts = self.attention_layer(x_out)
        confidence = F.softmax(attention_wts * self.temperature, dim=2).view(x_out.shape[0], x_out.shape[2])
        y = self.regressor_layer(x_out)
        return confidence, y


class RelSO3OutBlockR(nn.Module):

    def __init__(self, params, norm=None):
        super(RelSO3OutBlockR, self).__init__()
        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na)
        c_in = c_in * 2
        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        rp = params['representation']
        if rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        else:
            raise KeyError('Unrecognized representation of rotation: %s' % rp)
        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1, 1))
        self.regressor_layer = nn.Conv2d(mlp[-1], self.out_channel, (1, 1))
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, (1, 1)))
            c_in = c

    def forward(self, f1, f2, x1, x2):
        sp1 = zptk.SphericalPointCloud(x1, f1, None)
        sp2 = zptk.SphericalPointCloud(x2, f2, None)
        f1 = self._pooling(sp1)
        f2 = self._pooling(sp2)
        nb = f1.shape[0]
        na = f1.shape[2]
        f2_expand = f2.unsqueeze(-1).expand(-1, -1, -1, na).contiguous()
        f1_expand = f1.unsqueeze(-2).expand(-1, -1, na, -1).contiguous()
        x_out = torch.cat((f1_expand, f2_expand), 1)
        for linear in self.linear:
            x_out = linear(x_out)
            x_out = F.relu(x_out)
        attention_wts = self.attention_layer(x_out).view(nb, na, na)
        confidence = F.softmax(attention_wts * self.temperature, dim=1)
        y = self.regressor_layer(x_out)
        return confidence, y

    def _pooling(self, x):
        x_out = self.pointnet(x)
        x_out = F.relu(x_out)
        return x_out


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, pred, label):
        _, pred_label = pred.max(1)
        pred_label = pred_label.reshape(-1)
        label_flattened = label.reshape(-1)
        acc = (pred_label == label_flattened).sum().float() / float(label_flattened.shape[0])
        return self.metric(pred, label), acc


class AttentionCrossEntropyLoss(torch.nn.Module):

    def __init__(self, loss_type, loss_margin):
        super(AttentionCrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyLoss()
        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.iter_counter = 0

    def forward(self, pred, label, wts, rlabel, pretrain_step=2000):
        cls_loss, acc = self.metric(pred, label)
        """
        rlabel: B or Bx60 -> BxC
        wts: BxA or BxCxA -> BxAxC
        """
        if wts.ndimension() == 3:
            if wts.shape[1] <= rlabel.shape[1]:
                rlabel = rlabel[:, :wts.shape[1]]
            else:
                rlabel = rlabel.repeat(1, 10)[:, :wts.shape[1]]
            wts = wts.transpose(1, 2)
        r_loss, racc = self.metric(wts, rlabel)
        m = self.loss_margin
        loss_type = self.loss_type
        if loss_type == 'schedule':
            cls_loss_wts = min(float(self.iter_counter) / pretrain_step, 1.0)
            loss = cls_loss_wts * cls_loss + (m + 1.0 - cls_loss_wts) * r_loss
        elif loss_type == 'default':
            loss = cls_loss + m * r_loss
        elif loss_type == 'no_reg':
            loss = cls_loss
        else:
            raise NotImplementedError(f'{loss_type} is not Implemented!')
        if self.training:
            self.iter_counter += 1
        return loss, cls_loss, r_loss, acc, racc


def acos_safe(x, eps=0.0001):
    sign = torch.sign(x)
    slope = np.arccos(1 - eps) / eps
    return torch.where(abs(x) <= 1 - eps, torch.acos(x), torch.acos(sign * (1 - eps)) - slope * sign * (abs(x) - 1 + eps))


def angle_from_R(R):
    return acos_safe(0.5 * (torch.einsum('bii->b', R) - 1))


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def batched_select_anchor(labels, y, rotation_mapping):
    """
        (b, c, na_tgt, na_src) x (b, na_tgt)
            -> (b, na_src, c)
            -> (b, na, 3, 3)

    """
    b, na = labels.shape
    preds_rs = labels.view(-1)[:, None]
    y_rs = y.transpose(1, 3).contiguous()
    y_rs = y_rs.view(b * na, na, -1)
    y_select = batched_index_select(y_rs, 1, preds_rs).view(b * na, -1)
    pred_RAnchor = rotation_mapping(y_select).view(b, na, 3, 3).contiguous()
    return pred_RAnchor


def compute_rotation_matrix_from_ortho6d(ortho6d):

    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-08])))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if return_mag == True:
            return v, v_mag[:, 0]
        else:
            return v

    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        return out
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix


def compute_rotation_matrix_from_quaternion(quaternion):

    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-08])))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if return_mag == True:
            return v, v_mag[:, 0]
        else:
            return v
    batch = quaternion.shape[0]
    quat = normalize_vector(quaternion).contiguous()
    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw
    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)
    return matrix


def mean_angular_error(pred_R, gt_R):
    R_diff = torch.matmul(pred_R, gt_R.transpose(1, 2).float())
    angles = angle_from_R(R_diff)
    return angles


def so3_mean(Rs, weights=None):
    """Get the mean of the rotations.
        Parameters
        ----------
        Rs: (B,N,3,3)
        weights : array_like shape (B,N,), optional
            Weights describing the relative importance of the rotations. If
            None (default), then all values in `weights` are assumed to be
            equal.
        Returns
        -------
        mean R: (B,3,3)
        -----
        The mean used is the chordal L2 mean (also called the projected or
        induced arithmetic mean). If ``p`` is a set of rotations with mean
        ``m``, then ``m`` is the rotation which minimizes
        ``(weights[:, None, None] * (p.as_matrix() - m.as_matrix())**2).sum()``.

        """
    nb, na, _, _ = Rs.shape
    mask = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).float()
    mask2 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
    mask = mask[None].expand(nb, -1, -1).contiguous()
    mask2 = mask2[None].expand(nb, -1, -1).contiguous()
    if weights is None:
        weights = 1.0
    else:
        weights = weights[:, :, None, None]
    Ce = torch.sum(weights * Rs, dim=1)
    cu, cd, cv = torch.svd(Ce)
    cvT = cv.transpose(1, 2).contiguous()
    dets = torch.det(torch.matmul(cu, cvT))
    D = mask * dets[:, None, None] + mask2
    return torch.einsum('bij,bjk,bkl->bil', cu, D, cvT)


class MultiTaskDetectionLoss(torch.nn.Module):

    def __init__(self, anchors, nr=4, w=10, threshold=1.0):
        super(MultiTaskDetectionLoss, self).__init__()
        self.classifier = CrossEntropyLoss()
        self.anchors = anchors
        self.nr = nr
        assert nr == 4 or nr == 6
        self.w = w
        self.threshold = threshold
        self.iter_counter = 0

    def forward(self, wts, label, y, gt_R, gt_T=None):
        """ setting for alignment regression:
                - label (nb, na):
                    label the targte anchor from the perspective of source anchor na
                - wts (nb, na_tgt, na_src) normalized confidence weights
                - y (nb, nr, na_tgt, na_src) features
                - gt_R (nb, na, 3, 3)
                    relative rotation to regress from the perspective of source anchor na
                    Ra_tgti @ gt_R_i @ Ra_srci.T = gt_T for each i
                - gt_T (nb, 3, 3)
                    ground truth relative rotation: gt_T @ R_tgt = R_src

            setting for canonical regression:
                - label (nb)
                - wts (nb, na) normalized confidence weights
                - y (nb, nr, na) features to be mapped to 3x3 rotation matrices
                - gt_R (nb, na, 3, 3) relative rotation between gtR and each anchor
        """
        b = wts.shape[0]
        nr = self.nr
        na = wts.shape[1]
        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
        true_R = gt_R[:, 29] if gt_T is None else gt_T
        if na == 1:
            target_R = true_R
            cls_loss = torch.zeros(1)
            r_acc = torch.zeros(1) + 1
            pred_R = rotation_mapping(y.view(b, nr))
            l2_loss = torch.pow(pred_R - target_R, 2).mean()
            loss = self.w * l2_loss
        elif gt_T is not None and label.ndimension() == 2:
            wts = wts.view(b, na, na)
            cls_loss, r_acc = self.classifier(wts, label)
            confidence, preds = wts.max(1)
            select_RAnchor = batched_select_anchor(label, y, rotation_mapping)
            pred_RAnchor = batched_select_anchor(preds, y, rotation_mapping)
            confidence = confidence / (1e-06 + torch.sum(confidence, 1, keepdim=True))
            anchors_src = self.anchors[None].expand(b, -1, -1, -1).contiguous()
            pred_Rs = torch.einsum('baij, bajk, balk -> bail', anchors_src, pred_RAnchor, self.anchors[preds])
            pred_R = so3_mean(pred_Rs, confidence)
            l2_loss = torch.pow(gt_R - select_RAnchor, 2).mean()
            loss = cls_loss + self.w * l2_loss
        else:
            wts = wts.view(b, -1)
            cls_loss, r_acc = self.classifier(wts, label)
            pred_RAnchor = rotation_mapping(y.transpose(1, 2).contiguous().view(-1, nr)).view(b, -1, 3, 3)
            gt_bias = angle_from_R(gt_R.view(-1, 3, 3)).view(b, -1)
            mask = (gt_bias < self.threshold)[:, :, None, None].float()
            l2_loss = torch.pow(gt_R * mask - pred_RAnchor * mask, 2).sum()
            loss = cls_loss + self.w * l2_loss
            preds = torch.argmax(wts, 1)
            pred_R = batched_index_select(pred_RAnchor, 1, preds.long().view(b, -1)).view(b, 3, 3)
            pred_R = torch.matmul(self.anchors[preds], pred_R)
        if self.training:
            self.iter_counter += 1
        return loss, cls_loss, self.w * l2_loss, r_acc, mean_angular_error(pred_R, true_R)


def batch_hard_negative_mining(dist_mat):
    M, N = dist_mat.size(0), dist_mat.size(1)
    assert M == N
    labels = torch.arange(N, device=dist_mat.device).view(N, 1).expand(N, N)
    is_neg = labels.ne(labels.t())
    dist_an, _ = torch.min(torch.reshape(dist_mat[is_neg], (N, -1)), 1, keepdim=False)
    return dist_an


def pairwise_distance_matrix(x, y, eps=1e-06):
    M, N = x.size(0), y.size(0)
    x2 = torch.sum(x * x, dim=1, keepdim=True).repeat(1, N)
    y2 = torch.sum(y * y, dim=1, keepdim=True).repeat(1, M)
    dist2 = x2 + torch.t(y2) - 2.0 * torch.matmul(x, torch.t(y))
    dist2 = torch.clamp(dist2, min=eps)
    return torch.sqrt(dist2)


class TripletBatchLoss(nn.Module):

    def __init__(self, opt, anchors, sigma=0.2, interpolation='spherical', alpha=0.0):
        """
            anchors: na x 3 x 3, default anchor rotations
            margin: float, for triplet loss margin value
            sigma: float, sigma for softmax function
            loss: str "none" | "soft" | "hard", for loss mode
            interpolation: str "spherical" | "linear"
        """
        super(TripletBatchLoss, self).__init__()
        self.register_buffer('anchors', anchors)
        self.device = opt.device
        self.loss = opt.train_loss.loss_type
        self.margin = opt.train_loss.margin
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.k_precision = 1
        self.iter_counter = 0

    def forward(self, src, tgt, T, equi_src=None, equi_tgt=None):
        self.gt_idx = torch.arange(src.shape[0], dtype=torch.int32).unsqueeze(1).expand(-1, self.k_precision).contiguous().int()
        if self.alpha > 0 and equi_src is not None and equi_tgt is not None:
            return self._forward_equivariance(src, tgt, equi_src, equi_tgt, T)
        else:
            return self._forward_invariance(src, tgt)

    def _forward_invariance(self, src, tgt):
        """
            src, tgt: [nb, cdim]
        """
        dist_func = lambda a, b: (a - b) ** 2
        bdim = src.size(0)
        all_dist = pairwise_distance_matrix(src, tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)
        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)
        self.match_idx = idx
        self.all_dist = all_dist
        self.fpos = furthest_positive
        self.cneg = closest_negative
        return diff.mean(), accuracy, furthest_positive.mean(), closest_negative.mean()

    def _forward_equivariance(self, src, tgt, equi_src, equi_tgt, T):
        inv_loss, acc, fp, cn = self._forward_invariance(src, tgt)
        dist_func = lambda a, b: (a - b) ** 2
        bdim = src.size(0)
        equi_tgt = self._interpolate(equi_tgt, T, sigma=self.sigma).view(bdim, -1)
        equi_srcR = equi_src.view(bdim, -1)
        all_dist = pairwise_distance_matrix(equi_srcR, equi_tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)
        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)
        inv_info = [inv_loss, acc, fp, cn]
        equi_loss = diff.mean()
        total_loss = inv_loss + self.alpha * equi_loss
        equi_info = [equi_loss, accuracy, furthest_positive.mean(), closest_negative.mean()]
        return total_loss, inv_info, equi_info

    def _forward_attention(self, src, tgt, T, feats):
        """
            src, tgt: [nb, cdim]
            feats: (src_feat, tgt_feat) [nb, 1, na], normalized attention weights to be aligned
        """
        dist_func = lambda a, b: (a - b) ** 2
        src_wts = feats[0].squeeze().clamp(min=1e-05)
        tgt_wts = feats[1].squeeze().clamp(min=1e-05)
        inv_loss, acc, fpos, cneg = self._forward_invariance(src, tgt)
        loss_type = self.attention_params['attention_type']
        m = self.attention_params['attention_margin']
        pretrain_step = self.attention_params['attention_pretrain_step']
        if src_wts.ndimension() == 3:
            src_wts = src_wts.mean(-1)
            tgt_wts = tgt_wts.mean(-1)
        entropy = -(src_wts * src_wts.log() + tgt_wts * tgt_wts.log())
        entropy_loss = 0.01 * entropy.sum()
        if loss_type == 'no_reg':
            loss = inv_loss
        else:
            raise NotImplementedError(f'{loss_type} is not Implemented!')
        if self.training:
            self.iter_counter += 1
        return loss, inv_loss, entropy_loss, acc, fpos, cneg

    def _interpolate(self, feature, T, knn=3, sigma=0.1):
        """
            :param:
                anchors: [na, 3, 3]
                feature: [nb, cdim, na]
                T: [nb, 4, 4] rigid transformations or [nb, 3, 3]
            :return:
                rotated_feature: [nb, cdim, na]
        """
        bdim, cdim, adim = feature.shape
        R = T[:, :3, :3]
        r_anchors = torch.einsum('bij,njk->bnik', R.transpose(1, 2), self.anchors)
        influences, idx = self._rotation_distance(r_anchors, self.anchors, k=knn)
        influences = F.softmax(influences / sigma, 2)[:, None]
        idx = idx.view(-1)
        feat = feature[:, :, idx].reshape(bdim, cdim, adim, knn)
        feat = (feat * influences).sum(-1)
        return feat

    def _rotation_distance(self, r0, r1, k=3):
        diff_r = torch.einsum('bnij, mjk->bnmik', r0, r1.transpose(1, 2))
        traces = torch.einsum('bnmii->bnm', diff_r)
        return traces.topk(k=k, dim=2)


class BasicSO3Conv(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, debug=False):
        super(BasicSO3Conv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        if debug:
            W = torch.zeros(self.dim_out, self.dim_in * self.kernel_size) + 1
            self.register_buffer('W', W)
        else:
            W = torch.empty(self.dim_out, self.dim_in, self.kernel_size)
            nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
            W = W.view(self.dim_out, self.dim_in * self.kernel_size)
            self.register_parameter('W', nn.Parameter(W))

    def forward(self, x):
        bs, np, na = x.shape[0], x.shape[3], x.shape[4]
        x = x.view(bs, self.dim_in * self.kernel_size, np * na)
        x = torch.matmul(self.W, x)
        x = x.view(bs, self.dim_out, np, na)
        return x


KERNEL_CONDENSE_RATIO = 0.7

