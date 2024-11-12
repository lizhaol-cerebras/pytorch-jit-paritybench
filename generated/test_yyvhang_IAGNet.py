
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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch


from torchvision import transforms


import random


from sklearn.metrics import roc_auc_score


from numpy import nan


import pandas as pd


import torch.nn as nn


from torchvision import models


from torchvision.ops import roi_align


from time import time


import logging


class Cross_Attention(nn.Module):

    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.proj_q = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sk = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sv = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ek = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ev = nn.Linear(self.emb_dim, proj_dim)
        self.scale = self.proj_dim ** -0.5
        self.layernorm = nn.LayerNorm(self.emb_dim)

    def forward(self, obj, sub, scene):
        """
        obj: [B,N_p+HW,C]
        others : [B, HW, C]
        """
        B, seq_length, C = obj.size()
        query = self.proj_q(obj)
        s_key = self.proj_sk(sub)
        s_value = self.proj_sv(sub)
        e_key = self.proj_ek(scene)
        e_value = self.proj_ev(scene)
        atten_I1 = torch.bmm(query, s_key.mT) * self.scale
        atten_I1 = atten_I1.softmax(dim=-1)
        I_1 = torch.bmm(atten_I1, s_value)
        atten_I2 = torch.bmm(query, e_key.mT) * self.scale
        atten_I2 = atten_I2.softmax(dim=-1)
        I_2 = torch.bmm(atten_I2, e_value)
        I_1 = self.layernorm(obj + I_1)
        I_2 = self.layernorm(obj + I_2)
        return I_1, I_2


class Inherent_relation(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super(Inherent_relation, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.hidden_size ** 0.5
        attention_weights = nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, values)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.ln(out + x)
        return out


class Joint_Region_Alignment(nn.Module):

    def __init__(self, emb_dim=512, num_heads=4):
        super().__init__()


        class SwapAxes(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)
        self.emb_dim = emb_dim
        self.div_scale = self.emb_dim ** -0.5
        self.num_heads = num_heads
        self.to_common = nn.Sequential(nn.Conv1d(self.emb_dim, 2 * self.emb_dim, 1, 1), nn.BatchNorm1d(2 * self.emb_dim), nn.ReLU(), nn.Conv1d(2 * self.emb_dim, self.emb_dim, 1, 1), nn.BatchNorm1d(self.emb_dim), nn.ReLU())
        self.i_atten = Inherent_relation(self.emb_dim, self.num_heads)
        self.p_atten = Inherent_relation(self.emb_dim, self.num_heads)
        self.joint_atten = Inherent_relation(self.emb_dim, self.num_heads)

    def forward(self, F_i, F_p):
        """
        i_feature: [B, C, H, W]
        p_feature: [B, C, N_p]
        HW = N_i
        """
        B, _, N_p = F_p.size()
        F_i = F_i.view(B, self.emb_dim, -1)
        I = self.to_common(F_i)
        P = self.to_common(F_p)
        phi = torch.bmm(P.permute(0, 2, 1), I) * self.div_scale
        phi_p = F.softmax(phi, dim=1)
        phi_i = F.softmax(phi, dim=-1)
        I_enhance = torch.bmm(P, phi_p)
        P_enhance = torch.bmm(I, phi_i.permute(0, 2, 1))
        I_ = self.i_atten(I_enhance.mT)
        P_ = self.p_atten(P_enhance.mT)
        joint_patch = torch.cat((P_, I_), dim=1)
        F_j = self.joint_atten(joint_patch)
        return F_j


class Affordance_Revealed_Module(nn.Module):

    def __init__(self, emb_dim, proj_dim):


        class SwapAxes(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.cross_atten = Cross_Attention(emb_dim=self.emb_dim, proj_dim=self.proj_dim)
        self.fusion = nn.Sequential(nn.Conv1d(2 * self.emb_dim, self.emb_dim, 1, 1), nn.BatchNorm1d(self.emb_dim), nn.ReLU())

    def forward(self, F_j, F_s, F_e):
        """
        F_j: [B, N_p + N_i, C]
        F_s: [B, H, W, C]
        F_e: [B, H, W, C]
        """
        B, _, C = F_j.size()
        F_s = F_s.view(B, C, -1)
        F_e = F_e.view(B, C, -1)
        Theta_1, Theta_2 = self.cross_atten(F_j, F_s.mT, F_e.mT)
        joint_context = torch.cat((Theta_1.mT, Theta_2.mT), dim=1)
        affordance = self.fusion(joint_context)
        affordance = affordance.permute(0, 2, 1)
        return affordance


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 10000000000.0
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstractionMsg(nn.Module):

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class Point_Encoder(nn.Module):

    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2, 0.4], [16, 32], 256 + 256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]


class Img_Encoder(nn.Module):

    def __init__(self):
        super(Img_Encoder, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        down_1 = self.model.layer2(out)
        down_2 = self.model.layer3(down_1)
        down_3 = self.model.layer4(down_2)
        return down_3


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class Decoder(nn.Module):

    def __init__(self, additional_channel, emb_dim, N_p, N_raw, num_affordance):


        class SwapAxes(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N = N_raw
        self.num_affordance = num_affordance
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + self.emb_dim, mlp=[768, 512])
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512])
        self.fp1 = PointNetFeaturePropagation(in_channel=518 + additional_channel, mlp=[512, 512])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_head = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 8), SwapAxes(), nn.BatchNorm1d(self.emb_dim // 8), nn.ReLU(), SwapAxes(), nn.Linear(self.emb_dim // 8, 1))
        self.cls_head = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim // 2), nn.BatchNorm1d(self.emb_dim // 2), nn.ReLU(), nn.Linear(self.emb_dim // 2, self.num_affordance), nn.BatchNorm1d(self.num_affordance))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_j, affordance, encoder_p):
        """
        obj --->        [F_j]
        affordance ---> [B, N_p + N_i, C]
        encoder_p  ---> [Hierarchy feature]
        """
        B, _, _ = F_j.size()
        p_0, p_1, p_2, p_3 = encoder_p
        P_align, I_align = torch.split(F_j, split_size_or_sections=self.N_p, dim=1)
        F_pa, F_ia = torch.split(affordance, split_size_or_sections=self.N_p, dim=1)
        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], P_align.mT)
        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)
        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]], 1), up_sample)
        F_pa_pool = self.pool(F_pa.mT)
        F_ia_pool = self.pool(F_ia.mT)
        logits = torch.cat((F_pa_pool, F_ia_pool), dim=1)
        logits = self.cls_head(logits.view(B, -1))
        _3daffordance = up_sample * F_pa_pool.expand(-1, -1, self.N)
        _3daffordance = self.out_head(_3daffordance.mT)
        _3daffordance = self.sigmoid(_3daffordance)
        return _3daffordance, logits, [F_ia.mT.contiguous(), I_align.mT.contiguous()]


class IAG(nn.Module):

    def __init__(self, img_model_path=None, pre_train=True, normal_channel=False, local_rank=None, N_p=64, emb_dim=512, proj_dim=512, num_heads=4, N_raw=2048, num_affordance=18):


        class SwapAxes(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.num_affordance = num_affordance
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0
        self.img_encoder = Img_Encoder()
        if pre_train:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)
        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.JRA = Joint_Region_Alignment(self.emb_dim, self.num_heads)
        self.ARM = Affordance_Revealed_Module(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.N_p, self.N_raw, self.num_affordance)

    def forward(self, img, xyz, sub_box, obj_box):
        """
        img: [B, 3, H, W]
        xyz: [B, 3, 2048]
        sub_box: bounding box of the interactive subject
        obj_box: bounding box of the interactive object
        """
        B, C, N = xyz.size()
        if self.local_rank != None:
            device = torch.device('cuda', self.local_rank)
        else:
            device = torch.device('cuda:0')
        F_I = self.img_encoder(img)
        ROI_box = self.get_roi_box(B)
        F_i, F_s, F_e = self.get_mask_feature(img, F_I, sub_box, obj_box, device)
        F_e = roi_align(F_e, ROI_box, output_size=(4, 4))
        F_p_wise = self.point_encoder(xyz)
        F_j = self.JRA(F_i, F_p_wise[-1][1])
        affordance = self.ARM(F_j, F_s, F_e)
        _3daffordance, logits, to_KL = self.decoder(F_j, affordance, F_p_wise)
        return _3daffordance, logits, to_KL

    def get_mask_feature(self, raw_img, img_feature, sub_box, obj_box, device):
        raw_size = raw_img.size(2)
        current_size = img_feature.size(2)
        B = img_feature.size(0)
        scale_factor = current_size / raw_size
        sub_box[:, :] = sub_box[:, :] * scale_factor
        obj_box[:, :] = obj_box[:, :] * scale_factor
        obj_mask = torch.zeros_like(img_feature)
        obj_roi_box = []
        for i in range(B):
            obj_mask[i, :, int(obj_box[i][1] + 0.5):int(obj_box[i][3] + 0.5), int(obj_box[i][0] + 0.5):int(obj_box[i][2] + 0.5)] = 1
            roi_obj = [obj_box[i][0], obj_box[i][1], obj_box[i][2] + 0.5, obj_box[i][3]]
            roi_obj.insert(0, i)
            obj_roi_box.append(roi_obj)
        obj_roi_box = torch.tensor(obj_roi_box).float()
        sub_roi_box = []
        Scene_mask = obj_mask.clone()
        for i in range(B):
            Scene_mask[i, :, int(sub_box[i][1] + 0.5):int(sub_box[i][3] + 0.5), int(sub_box[i][0] + 0.5):int(sub_box[i][2] + 0.5)] = 1
            roi_sub = [sub_box[i][0], sub_box[i][1], sub_box[i][2], sub_box[i][3]]
            roi_sub.insert(0, i)
            sub_roi_box.append(roi_sub)
        Scene_mask = torch.abs(Scene_mask - 1)
        Scene_mask_feature = img_feature * Scene_mask
        sub_roi_box = torch.tensor(sub_roi_box).float()
        obj_feature = roi_align(img_feature, obj_roi_box, output_size=(4, 4), sampling_ratio=4)
        sub_feature = roi_align(img_feature, sub_roi_box, output_size=(4, 4), sampling_ratio=4)
        return obj_feature, sub_feature, Scene_mask_feature

    def get_roi_box(self, batch_size):
        batch_box = []
        roi_box = [0.0, 0.0, 6.0, 6.0]
        for i in range(batch_size):
            roi_box.insert(0, i)
            batch_box.append(roi_box)
            roi_box = roi_box[1:]
        batch_box = torch.tensor(batch_box).float()
        return batch_box


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNet_Estimation(nn.Module):

    def __init__(self, num_classes, normal_channel=False):
        super(PointNet_Estimation, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + additional_channel, mlp=[128, 128])
        self.classifier = nn.ModuleList()
        for i in range(num_classes):
            classifier = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.Conv1d(128, 1, 1))
            self.classifier.append(classifier)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xyz):
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        score = self.classifier[0](l0_points)
        for index, classifier in enumerate(self.classifier):
            if index == 0:
                continue
            score_ = classifier(l0_points)
            score = torch.cat((score, score_), dim=1)
        score = score.permute(0, 2, 1).contiguous()
        score = self.sigmoid(score)
        return score


class HM_Loss(nn.Module):

    def __init__(self):
        super(HM_Loss, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):
        temp1 = -(1 - self.alpha) * torch.mul(pred ** self.gamma, torch.mul(1 - target, torch.log(1 - pred + 1e-06)))
        temp2 = -self.alpha * torch.mul((1 - pred) ** self.gamma, torch.mul(target, torch.log(pred + 1e-06)))
        temp = temp1 + temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))
        intersection_positive = torch.sum(pred * target, 1)
        cardinality_positive = torch.sum(torch.abs(pred) + torch.abs(target), 1)
        dice_positive = (intersection_positive + 1e-06) / (cardinality_positive + 1e-06)
        intersection_negative = torch.sum((1.0 - pred) * (1.0 - target), 1)
        cardinality_negative = torch.sum(2 - torch.abs(pred) - torch.abs(target), 1)
        dice_negative = (intersection_negative + 1e-06) / (cardinality_negative + 1e-06)
        temp3 = torch.mean(1.5 - dice_positive - dice_negative, 0)
        DICELoss = torch.sum(temp3)
        return CELoss + 1.0 * DICELoss


class CrossModalCenterLoss(nn.Module):
    """Center loss.    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim=512, local_rank=None):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.local_rank = local_rank
        if self.local_rank != None:
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda:0')
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        temp = torch.mm(x, self.centers.t())
        distmat = distmat - 2 * temp
        classes = torch.arange(self.num_classes).long()
        classes = classes
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1000000000000.0).sum() / batch_size
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Affordance_Revealed_Module,
     lambda: ([], {'emb_dim': 4, 'proj_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Cross_Attention,
     lambda: ([], {'emb_dim': 4, 'proj_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (HM_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Img_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Inherent_relation,
     lambda: ([], {'hidden_size': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PointNetSetAbstraction,
     lambda: ([], {'npoint': 4, 'radius': 4, 'nsample': 4, 'in_channel': 4, 'mlp': [4, 4], 'group_all': 4}),
     lambda: ([torch.rand([4, 1, 4]), torch.rand([4, 3, 4])], {})),
]

