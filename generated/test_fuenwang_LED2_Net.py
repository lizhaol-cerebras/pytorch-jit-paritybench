
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


import copy


import matplotlib.pyplot as plt


from torch.utils.data import Dataset as TorchDataset


from torch.utils.data import DataLoader as TorchDataLoader


import torch.nn.functional as F


import math


import torchvision.models as models


import functools


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from scipy.optimize import least_squares


from functools import partial


import scipy.misc as sic


import time


import random


from torch.utils.tensorboard import SummaryWriter


class BaseModule(nn.Module):

    def __init__(self, path):
        super().__init__()
        self.path = path
        os.system('mkdir -p %s' % path)
        self.model_lst = [x for x in sorted(os.listdir(self.path)) if x.endswith('.pkl')]
        self.best_model = None
        self.best_accuracy = -float('inf')

    def _loadName(self, epoch=None):
        if len(self.model_lst) == 0:
            None
            return None, 0
        if epoch is not None:
            for i, name in enumerate(self.model_lst):
                if name.endswith('%.5d.pkl' % epoch):
                    None
                    return name, i
            None
            return None, 0
        else:
            None
            return self.model_lst[-1], len(self.model_lst) - 1

    def Load(self, epoch=None):
        name, _ = self._loadName(epoch)
        if name is not None:
            params = torch.load('%s/%s' % (self.path, name))
            self.load_state_dict(params, strict=False)
            self.best_model = name
            epoch = int(self.best_model.split('_')[-1].split('.')[0]) + 1
        else:
            epoch = 0
        return epoch

    def Save(self, epoch, accuracy=None, replace=False):
        if accuracy is None or replace == False:
            aaa = '%.5d' % epoch
            now = 'model_%s.pkl' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
            params = self.state_dict()
            name = '%s/%s' % (self.path, now)
            torch.save(params, name)
            self.best_model = now
        elif accuracy > self.best_accuracy:
            aaa = '%.5d' % epoch
            now = 'model_%s.pkl' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
            params = self.state_dict()
            name = '%s/%s' % (self.path, now)
            if self.best_model is not None:
                os.system('rm %s/%s' % (self.path, self.best_model))
            torch.save(params, name)
            self.best_model = now
            self.best_accuracy = accuracy
            None


class Corner2Depth(nn.Module):

    def __init__(self, grid):
        super(Corner2Depth, self).__init__()
        self.grid = grid

    def setGrid(self, grid):
        self.grid = grid

    def forward(self, corners, nums, shift=None, mode='origin'):
        if mode == 'origin':
            return self.forward_origin(corners, nums, shift)
        else:
            return self.forward_fast(corners, nums, shift)

    def forward_fast(self, corners, nums, shift=None):
        if shift is not None:
            raise NotImplementedError
        grid_origin = self.grid
        eps = 0.01
        depth_maps = []
        normal_maps = []
        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, ...].clone()
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            diff = corners_now[1:, ...] - corners_now[:-1, ...]
            vec_yaxis = torch.zeros_like(diff)
            vec_yaxis[..., 1] = 1
            cross_result = torch.cross(diff, vec_yaxis, dim=1)
            d = -torch.sum(cross_result * corners_now[:-1, ...], dim=1, keepdim=True)
            planes = torch.cat([cross_result, d], dim=1)
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)
            intersec = []
            for idx in range(scale_all.shape[-1]):
                intersec.append((grid * scale_all[..., idx:idx + 1]).unsqueeze(-1))
            intersec = torch.cat(intersec, dim=-1)
            a = corners_now[1:, ...]
            b = corners_now[:-1, ...]
            x_cat = torch.cat([a[:, 0:1], b[:, 0:1]], dim=1)
            z_cat = torch.cat([a[:, 2:], b[:, 2:]], dim=1)
            max_x, min_x = torch.max(x_cat, dim=1)[0], torch.min(x_cat, dim=1)[0]
            max_z, min_z = torch.max(z_cat, dim=1)[0], torch.min(z_cat, dim=1)[0]
            mask_x = (intersec[:, :, :, 0, :] <= max_x + eps) & (intersec[:, :, :, 0, :] >= min_x - eps)
            mask_z = (intersec[:, :, :, 2, :] <= max_z + eps) & (intersec[:, :, :, 2, :] >= min_z - eps)
            mask_valid = scale_all > 0
            mask = ~(mask_x & mask_z & mask_valid)
            scale_all[mask] = float('inf')
            depth, min_idx = torch.min(scale_all, dim=-1)
            _, h, w = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(1, h, w, -1)
            depth_maps.append(depth)
            normal_maps.append(normal)
        depth_maps = torch.cat(depth_maps, dim=0).unsqueeze(1)
        normal_maps = torch.cat(normal_maps, dim=0)
        return depth_maps, normal_maps

    def forward_origin(self, corners, nums, shift=None):
        grid_origin = self.grid
        eps = 0.01
        depth_maps = []
        normal_maps = []
        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, :num, ...].clone()
            if shift is not None:
                corners_now[..., 0] -= shift[i, 0]
                corners_now[..., 2] -= shift[i, 1]
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            planes = []
            for j in range(1, corners_now.shape[0]):
                vec_corner = corners_now[j:j + 1, ...] - corners_now[j - 1:j, ...]
                vec_yaxis = torch.zeros_like(vec_corner)
                vec_yaxis[..., 1] = 1
                cross_result = torch.cross(vec_corner, vec_yaxis)
                cross_result = cross_result / torch.norm(cross_result, p=2, dim=-1)[..., None]
                d = -torch.sum(cross_result * corners_now[j:j + 1, ...], dim=-1)[..., None]
                abcd = torch.cat([cross_result, d], dim=-1)
                planes.append(abcd)
            planes = torch.cat(planes, dim=0)
            assert planes.shape[0] == num
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)
            depth = []
            for j in range(scale_all.shape[-1]):
                scale = scale_all[..., j]
                intersec = scale[..., None] * grid
                a = corners_now[j + 1:j + 2, :]
                b = corners_now[j:j + 1, :]
                rang = torch.cat([a, b], dim=0)
                max_x, min_x = torch.max(rang[:, 0]), torch.min(rang[:, 0])
                max_z, min_z = torch.max(rang[:, 2]), torch.min(rang[:, 2])
                mask_x = (intersec[..., 0] <= max_x + eps) & (intersec[..., 0] >= min_x - eps)
                mask_z = (intersec[..., 2] <= max_z + eps) & (intersec[..., 2] >= min_z - eps)
                mask_valid = scale > 0
                mask = ~(mask_x & mask_z & mask_valid)
                scale[mask] = float('inf')
                depth.append(scale[None, ...])
            depth = torch.cat(depth, dim=1)
            depth, min_idx = torch.min(depth, dim=1)
            [_, h, w] = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(-1, h, w, 3)
            normal_maps.append(normal)
            depth_maps.append(depth[None, ...])
        depth_maps = torch.cat(depth_maps, dim=0)
        normal_maps = torch.cat(normal_maps, dim=0)
        return depth_maps, normal_maps


class RenderLoss(nn.Module):

    def __init__(self, camera_height=1.6):
        super(RenderLoss, self).__init__()
        assert camera_height > 0
        self.cH = camera_height
        self.grid = None
        self.c2d = Corner2Depth(None)
        self.et = Conversion.EquirecTransformer('torch')

    def setGrid(self, grid):
        self.grid = grid
        self.c2d.setGrid(grid)

    def lonlat2xyz_up(self, pred_up, GT_up, up_down_ratio):
        pred_up_xyz = self.et.lonlat2xyz(pred_up)
        GT_up_xyz = self.et.lonlat2xyz(GT_up)
        s = -(self.cH * up_down_ratio[..., None, None]) / pred_up_xyz[..., 1:2].detach()
        pred_up_xyz *= s
        s = -(self.cH * up_down_ratio[..., None, None]) / GT_up_xyz[..., 1:2]
        GT_up_xyz *= s
        return pred_up_xyz, GT_up_xyz

    def lonlat2xyz_down(self, pred_down, dummy1=None, dummy2=None):
        pred_down_xyz = self.et.lonlat2xyz(pred_down)
        s = self.cH / pred_down_xyz[..., 1:2].detach()
        pred_down_xyz *= s
        return pred_down_xyz, None

    def forward(self, pred_up, pred_down, GT_up, corner_nums, up_down_ratio):
        assert self.grid is not None
        pred_up_xyz, GT_up_xyz = self.lonlat2xyz_up(pred_up, GT_up, up_down_ratio)
        pred_down_xyz, _ = self.lonlat2xyz_down(pred_down)
        gt_depth, _ = self.c2d(GT_up_xyz, corner_nums)
        pred_depth_up, _ = self.c2d(pred_up_xyz, corner_nums, mode='fast')
        pred_depth_down, _ = self.c2d(pred_down_xyz, corner_nums, mode='fast')
        GT_up_xyz_dense = self.grid[:, 0, ...] * gt_depth.permute(0, 2, 3, 1)[:, 0, :, :]
        GT_up_xyz_dense[..., 1:2] = -(self.cH * up_down_ratio[..., None, None])
        loss_depth_up = F.l1_loss(pred_depth_up, gt_depth)
        loss_depth_down = F.l1_loss(pred_depth_down, gt_depth)
        return loss_depth_up, loss_depth_down, [pred_up_xyz, pred_down_xyz, GT_up_xyz, GT_up_xyz_dense], [pred_depth_up, pred_depth_down, gt_depth]


class ShiftSampler(nn.Module):

    def __init__(self, dim=256, down_ratio=0.5):
        super(ShiftSampler, self).__init__()
        self.dim = dim
        self.down_ratio = down_ratio
        self.grid_x, self.grid_z = np.meshgrid(range(dim), range(dim))

    def _GetAngle(self, pred):
        [num, _] = pred.shape
        tmp = np.concatenate([pred, pred[0:1, :]], axis=0)
        abs_cos = []

    def forward(self, pred_xyz, pred_corner_num, gt_xyz, gt_corner_num):
        device = pred_xyz.device
        out = np.zeros([pred_xyz.shape[0], 2], dtype=np.float32)
        pred_xyz = pred_xyz.data.cpu().numpy() * self.down_ratio
        pred_corner_num = pred_corner_num.data.cpu().numpy()
        gt_xyz = gt_xyz.data.cpu().numpy() * self.down_ratio
        gt_corner_num = gt_corner_num.data.cpu().numpy()
        for i in range(pred_xyz.shape[0]):
            max_x1 = pred_xyz[i, :pred_corner_num[i], 0].max()
            max_x2 = gt_xyz[i, :gt_corner_num[i], 0].max()
            min_x1 = pred_xyz[i, :pred_corner_num[i], 0].min()
            min_x2 = gt_xyz[i, :gt_corner_num[i], 0].min()
            max_z1 = pred_xyz[i, :pred_corner_num[i], 2].max()
            max_z2 = gt_xyz[i, :gt_corner_num[i], 2].max()
            min_z1 = pred_xyz[i, :pred_corner_num[i], 2].min()
            min_z2 = gt_xyz[i, :gt_corner_num[i], 2].min()
            max_x = np.max([max_x1, max_x2])
            min_x = np.min([min_x1, min_x2])
            max_z = np.max([max_z1, max_z2])
            min_z = np.min([min_z1, min_z2])
            pred_xyz_now_normalized = pred_xyz[i, :pred_corner_num[i], :].copy()
            self._GetAngle(pred_xyz_now_normalized)
            pred_xyz_now_normalized[:, 0] = (pred_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            pred_xyz_now_normalized[:, 2] = (pred_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)
            gt_xyz_now_normalized = gt_xyz[i, :gt_corner_num[i], :].copy()
            gt_xyz_now_normalized[:, 0] = (gt_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            gt_xyz_now_normalized[:, 2] = (gt_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)
            pred_xz_now_normalized = (pred_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)
            gt_xz_now_normalized = (gt_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)
            pred_mask = np.zeros([self.dim, self.dim], np.uint8)
            gt_mask = np.zeros([self.dim, self.dim], np.uint8)
            cv2.drawContours(pred_mask, [pred_xz_now_normalized], -1, 255, cv2.FILLED)
            cv2.drawContours(gt_mask, [gt_xz_now_normalized], -1, 255, cv2.FILLED)
            mask = np.logical_and(pred_mask.astype(np.bool), gt_mask.astype(np.bool))
            x_valid = self.grid_x[mask]
            z_valid = self.grid_z[mask]
            idx_choice = np.random.choice(range(z_valid.shape[0]))
            if False:
                plt.subplot('311')
                plt.imshow(pred_mask)
                plt.subplot('312')
                plt.imshow(gt_mask)
                plt.subplot('313')
                plt.imshow(mask)
                plt.show()
            x_choose = x_valid[idx_choice].astype(np.float32)
            z_choose = z_valid[idx_choice].astype(np.float32)
            out[i, 0] = x_choose / (self.dim - 1) * (max_x - min_x) + min_x
            out[i, 1] = z_choose / (self.dim - 1) * (max_z - min_z) + min_z
        return torch.FloatTensor(out)


def lr_pad(x, padding=1):
    """ Pad left/right-most to each other instead of zero padding """
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    """ Pad left/right-most to each other instead of zero padding """

    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


ENCODER_RESNET = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


class Resnet(nn.Module):

    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features.append(x)
        x = self.encoder.layer2(x)
        features.append(x)
        x = self.encoder.layer3(x)
        features.append(x)
        x = self.encoder.layer4(x)
        features.append(x)
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


ENCODER_DENSENET = ['densenet121', 'densenet169', 'densenet161', 'densenet201']


class Densenet(nn.Module):

    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


class ConvCompressH(nn.Module):
    """ Reduce feature height by factor of two """

    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):

    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(ConvCompressH(in_c, in_c // 2), ConvCompressH(in_c // 2, in_c // 2), ConvCompressH(in_c // 2, in_c // 4), ConvCompressH(in_c // 4, out_c))

    def forward(self, x, out_w):
        x = self.layer(x)
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):

    def __init__(self, c1, c2, c3, c4, out_scale=8):
        """ Process 4 blocks from encoder to single multiscale features """
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([GlobalHeightConv(c1, c1 // out_scale), GlobalHeightConv(c2, c2 // out_scale), GlobalHeightConv(c3, c3 // out_scale), GlobalHeightConv(c4, c4 // out_scale)])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([f(x, out_w).reshape(bs, -1, out_w) for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)], dim=1)
        return feature


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = m.padding[0], 0
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(root, names[-1], nn.Sequential(LR_PAD(w_pad), m))


class Network(BaseModule):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, save_path, backbone):
        super().__init__(save_path)
        self.backbone = backbone
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1 * 8 + c2 * 4 + c3 * 2 + c4 * 1) // self.out_scale
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)
        self.bi_rnn = nn.LSTM(input_size=c_last, hidden_size=self.rnn_hidden_size, num_layers=2, dropout=0.5, batch_first=False, bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size, out_features=3)
        self.linear.bias.data[0 * self.step_cols:1 * self.step_cols].fill_(-1)
        self.linear.bias.data[1 * self.step_cols:2 * self.step_cols].fill_(-0.478)
        self.linear.bias.data[2 * self.step_cols:3 * self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        x = x.clone()
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean
            self.x_std = self.x_std
        x[:, :3] = (x[:, :3] - self.x_mean) / self.x_std
        return x

    def forward(self, x):
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)
        feature = feature.permute(2, 0, 1)
        self.bi_rnn.flatten_parameters()
        output, hidden = self.bi_rnn(feature)
        output = self.drop_out(output)
        output = self.linear(output)
        output = output.permute(1, 2, 0)
        cor = output[:, :1]
        bon = output[:, 1:]
        bon = torch.sigmoid(bon)
        up = bon[:, 0:1, :] * -0.5 * math.pi
        down = bon[:, 1:, :] * 0.5 * math.pi
        bon = torch.cat([up, down], dim=1)
        return bon


class CubePad(nn.Module):

    def __init__(self, pad_size, pad_corner=True):
        super(CubePad, self).__init__()
        self.pad_corner = pad_corner
        if type(pad_size) == int:
            self.up_pad = pad_size
            self.down_pad = pad_size
            self.left_pad = pad_size
            self.right_pad = pad_size
        elif type(pad_size) == list:
            [self.up_pad, self.down_pad, self.left_pad, self.right_pad] = pad_size
        self.relation = {'back': ['top-up_yes_yes_no', 'down-down_yes_yes_no', 'right-right_no_no_no', 'left-left_no_no_no'], 'down': ['front-down_no_no_no', 'back-down_yes_yes_no', 'left-down_yes_no_yes', 'right-down_no_yes_yes'], 'front': ['top-down_no_no_no', 'down-up_no_no_no', 'left-right_no_no_no', 'right-left_no_no_no'], 'left': ['top-left_yes_no_yes', 'down-left_no_yes_yes', 'back-right_no_no_no', 'front-left_no_no_no'], 'right': ['top-right_no_yes_yes', 'down-right_yes_no_yes', 'front-right_no_no_no', 'back-left_no_no_no'], 'top': ['back-up_yes_yes_no', 'front-up_no_no_no', 'left-up_no_yes_yes', 'right-up_yes_no_yes']}

    def forward(self, x):
        [bs, c, h, w] = x.size()
        assert bs % 6 == 0 and h == w
        [up_pad, down_pad, left_pad, right_pad] = [self.up_pad, self.down_pad, self.left_pad, self.right_pad]
        mx_pad = max([up_pad, down_pad, left_pad, right_pad])
        if mx_pad <= 0:
            return x
        faces = {'back': None, 'down': None, 'front': None, 'left': None, 'right': None, 'top': None}
        sides = {'back-up': None, 'back-down': None, 'back-left': None, 'back-right': None, 'down-up': None, 'down-down': None, 'down-left': None, 'down-right': None, 'front-up': None, 'front-down': None, 'front-left': None, 'front-right': None, 'left-up': None, 'left-down': None, 'left-left': None, 'left-right': None, 'right-up': None, 'right-down': None, 'right-left': None, 'right-right': None, 'top-up': None, 'top-down': None, 'top-left': None, 'top-right': None}
        for idx, face in enumerate(['back', 'down', 'front', 'left', 'right', 'top']):
            tmp = x[idx::6, :, :, :]
            faces[face] = tmp
            for side in ['up', 'down', 'left', 'right']:
                if side == 'up':
                    pad_array = tmp[:, :, 0:mx_pad, :]
                elif side == 'down':
                    pad_array = tmp[:, :, h - mx_pad:h, :]
                elif side == 'left':
                    pad_array = tmp[:, :, :, 0:mx_pad]
                elif side == 'right':
                    pad_array = tmp[:, :, :, w - mx_pad:w]
                key = '%s-%s' % (face, side)
                assert key in sides
                sides[key] = pad_array
        out = []
        for idx, f in enumerate(['back', 'down', 'front', 'left', 'right', 'top']):
            face = faces[f]
            new_face = F.pad(face, (left_pad, right_pad, up_pad, down_pad), 'constant', 0)
            [bs, _, new_h, new_w] = new_face.size()
            assert new_h == new_w
            for pad_order, relation in zip(['up', 'down', 'left', 'right'], self.relation[f]):
                pad_side, flip_h, flip_w, transpose = relation.split('_')
                pad_array = sides[pad_side]
                if transpose == 'yes':
                    pad_array = pad_array.transpose(2, 3)
                [_, _, hh, ww] = pad_array.size()
                if flip_h == 'yes':
                    index = Variable(torch.arange(hh - 1, -1, -1).type(torch.LongTensor))
                    pad_array = torch.index_select(pad_array, dim=2, index=index)
                if flip_w == 'yes':
                    index = Variable(torch.arange(ww - 1, -1, -1).type(torch.LongTensor))
                    pad_array = torch.index_select(pad_array, dim=3, index=index)
                if pad_order == 'up' and up_pad != 0:
                    new_face[:, :, 0:up_pad, left_pad:new_w - right_pad] = pad_array[:, :, 0:up_pad, :]
                elif pad_order == 'down' and down_pad != 0:
                    new_face[:, :, new_h - down_pad:new_h, left_pad:new_w - right_pad] = pad_array[:, :, 0:down_pad, :]
                elif pad_order == 'left' and left_pad != 0:
                    new_face[:, :, up_pad:new_h - down_pad, 0:left_pad] = pad_array[:, :, :, 0:left_pad]
                elif pad_order == 'right' and right_pad != 0:
                    new_face[:, :, up_pad:new_h - down_pad, new_w - right_pad:new_w] = pad_array[:, :, :, 0:right_pad]
            out.append(new_face)
        out = torch.cat(out, dim=0)
        [bs, c, h, w] = out.size()
        out2 = out.view(-1, bs // 6, c, h, w).transpose(0, 1).contiguous().view(bs, c, h, w)
        if self.pad_corner:
            for corner in ['left_up', 'right_up', 'left_down', 'right_down']:
                if corner == 'left_up' and (left_pad > 0 and up_pad > 0):
                    out2[:, :, 0:up_pad, 0:left_pad] = out[:, :, 0:up_pad, left_pad:left_pad + 1].repeat(1, 1, 1, left_pad).clone()
                elif corner == 'right_up' and (right_pad > 0 and up_pad > 0):
                    out2[:, :, 0:up_pad, w - right_pad:w] = out[:, :, 0:up_pad, w - right_pad - 1:w - right_pad].repeat(1, 1, 1, right_pad).clone()
                elif corner == 'left_down' and (left_pad > 0 and down_pad > 0):
                    out2[:, :, h - down_pad:h, 0:left_pad] = out[:, :, h - down_pad:h, left_pad:left_pad + 1].repeat(1, 1, 1, left_pad).clone()
                elif corner == 'right_down' and (right_pad > 0 and down_pad > 0):
                    out2[:, :, h - down_pad:h, w - right_pad:w] = out[:, :, h - down_pad:h, w - right_pad - 1:w - right_pad].repeat(1, 1, 1, right_pad).clone()
        return out2


class CustomPad(nn.Module):

    def __init__(self, pad_func):
        super(CustomPad, self).__init__()
        self.pad_func = pad_func

    def forward(self, x):
        return self.pad_func(x)


class NoOp(nn.Module):

    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class ZeroPad(nn.Module):

    def __init__(self, pad_s):
        super(ZeroPad, self).__init__()
        self.pad_s = pad_s

    def forward(self, x):
        x = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s))
        return x


class SpherePadGrid(object):

    def __init__(self, cube_dim, equ_h, FoV=90.0):
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([[0, -180.0, 0], [90.0, 0, 0], [0, 0, 0], [0, 90, 0], [0, -90, 0], [-90, 0, 0]], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        self._getCubeGrid()

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        self.intrisic = {'f': float(f), 'cx': float(cx), 'cy': float(cy)}
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        self.K = K
        self.grids = []
        self.grids_xyz = []
        for R in self.R_lst:
            tmp = xyz @ R
            self.grids_xyz.append(tmp)
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            self.grids.append(torch.FloatTensor(lonlat[None, ...]))

    def __call__(self):
        R_lst = [torch.FloatTensor(x) for x in self.R_lst]
        grids_xyz = [torch.FloatTensor(x).view(1, self.cube_dim, self.cube_dim, 3) for x in self.grids_xyz]
        K = torch.FloatTensor(self.K)
        return R_lst, grids_xyz, self.intrisic


class SpherePad(nn.Module):

    def __init__(self, pad_size):
        super(SpherePad, self).__init__()
        self.pad_size = pad_size
        self.data = {}
        self.relation = {'back': ['top-up_yes_yes_no', 'down-down_yes_yes_no', 'right-right_no_no_no', 'left-left_no_no_no'], 'down': ['front-down_no_no_no', 'back-down_yes_yes_no', 'left-down_yes_no_yes', 'right-down_no_yes_yes'], 'front': ['top-down_no_no_no', 'down-up_no_no_no', 'left-right_no_no_no', 'right-left_no_no_no'], 'left': ['top-left_yes_no_yes', 'down-left_no_yes_yes', 'back-right_no_no_no', 'front-left_no_no_no'], 'right': ['top-right_no_yes_yes', 'down-right_yes_no_yes', 'front-right_no_no_no', 'back-left_no_no_no'], 'top': ['back-up_yes_yes_no', 'front-up_no_no_no', 'left-up_no_yes_yes', 'right-up_yes_no_yes']}

    def _GetLoc(self, R_lst, grid_lst, K):
        out = {}
        pad = self.pad_size
        f, cx, cy = K['f'], K['cx'], K['cy']
        K_mat = torch.FloatTensor(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]))
        grid_front = grid_lst[2]
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        for i, face in enumerate(orders):
            out[face] = {}
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face = self.relation[face][j].split('-')[0]
                idx = orders.index(connected_face)
                R_world_to_connected = R_lst[idx]
                R_world_to_itself = R_lst[i]
                R_itself_to_connected = torch.matmul(R_world_to_connected, R_world_to_itself.transpose(0, 1))
                new_grid = torch.matmul(grid_front, R_itself_to_connected.transpose(0, 1))
                proj = torch.matmul(new_grid, K_mat.transpose(0, 1))
                x = proj[:, :, :, 0:1] / proj[:, :, :, 2:3]
                y = proj[:, :, :, 1:2] / proj[:, :, :, 2:3]
                x = (x - cx) / cx
                y = (y - cy) / cy
                xy = torch.cat([x, y], dim=3)
                out[face][connect_side] = {}
                x = xy[:, :, :, 0:1]
                y = xy[:, :, :, 1:2]
                """
                mask1 = np.logical_and(x >= -1.01, x <= 1.01)
                mask2 = np.logical_and(y >= -1.01, y <= 1.01)
                mask = np.logical_and(mask1, mask2)
                """
                mask1 = (x >= -1.01) & (x <= 1.01)
                mask2 = (y >= -1.01) & (y <= 1.01)
                mask = mask1 & mask2
                xy = torch.clamp(xy, -1, 1)
                if connect_side == 'up':
                    out[face][connect_side]['mask'] = mask[:, :pad, :, :]
                    out[face][connect_side]['xy'] = xy[:, :pad, :, :]
                elif connect_side == 'down':
                    out[face][connect_side]['mask'] = mask[:, -pad:, :, :]
                    out[face][connect_side]['xy'] = xy[:, -pad:, :, :]
                elif connect_side == 'left':
                    out[face][connect_side]['mask'] = mask[:, :, :pad, :]
                    out[face][connect_side]['xy'] = xy[:, :, :pad, :]
                elif connect_side == 'right':
                    out[face][connect_side]['mask'] = mask[:, :, -pad:, :]
                    out[face][connect_side]['xy'] = xy[:, :, -pad:, :]
        return out

    def forward(self, inputs):
        [bs, c, h, w] = inputs.shape
        assert bs % 6 == 0 and h == w
        key = '(%d,%d,%d)' % (h, w, self.pad_size)
        if key not in self.data:
            theta = 2 * np.arctan((0.5 * h + self.pad_size) / (0.5 * h))
            grid_ori = SpherePadGrid(h, 2 * h, 90)
            grid = SpherePadGrid(h + 2 * self.pad_size, 2 * h, theta / np.pi * 180)
            R_lst, grid_lst, _ = grid()
            _, _, K = grid_ori()
            self.data[key] = self._GetLoc(R_lst, grid_lst, K)
        pad = self.pad_size
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        out = []
        for i, face in enumerate(orders):
            this_face = inputs[i::6]
            this_face = F.pad(this_face, (pad, pad, pad, pad))
            repeats = this_face.shape[0]
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face_name = self.relation[face][j].split('-')[0]
                connected_face = inputs[orders.index(connected_face_name)::6]
                mask = self.data[key][face][connect_side]['mask'].cuda().repeat(repeats, 1, 1, c).permute(0, 3, 1, 2)
                xy = self.data[key][face][connect_side]['xy'].cuda().repeat(repeats, 1, 1, 1)
                interpo = F.grid_sample(connected_face, xy, align_corners=True, mode='bilinear')
                if connect_side == 'up':
                    this_face[:, :, :pad, :][mask] = interpo[mask]
                elif connect_side == 'down':
                    this_face[:, :, -pad:, :][mask] = interpo[mask]
                elif connect_side == 'left':
                    this_face[:, :, :, :pad][mask] = interpo[mask]
                elif connect_side == 'right':
                    this_face[:, :, :, -pad:][mask] = interpo[mask]
            out.append(this_face)
        out = torch.cat(out, dim=0)
        [bs, c, h, w] = out.shape
        out = out.view(-1, bs // 6, c, h, w).transpose(0, 1).contiguous().view(bs, c, h, w)
        return out


class EquirecTransformer:

    def __init__(self, mode, clip=False):
        assert mode in ['numpy', 'torch']
        self.mode = mode
        self.clip = clip

    def XY2lonlat(self, xy, shape=(512, 1024)):
        return XY2lonlat(xy, shape, self.mode)

    def lonlat2xyz(self, lonlat, RADIUS=1.0):
        return lonlat2xyz(lonlat, RADIUS, self.mode)

    def XY2xyz(self, xy, shape=(512, 1024)):
        return XY2xyz(xy, shape, self.mode)

    def xyz2lonlat(self, xyz):
        return xyz2lonlat(xyz, self.clip, self.mode)

    def lonlat2XY(self, lonlat, shape=(512, 1024)):
        return lonlat2XY(lonlat, shape, self.mode)

    def xyz2XY(self, xyz, shape=(512, 1024)):
        return xyz2XY(xyz, shape, self.clip, self.mode)


def errorCalculate(ratio, up_norm, down_norm):
    error = np.abs(ratio * up_norm - down_norm)
    return error


class InferHeight(nn.Module):

    def __init__(self, scale=1):
        super().__init__()
        self.scale = 1
        self.et = EquirecTransformer('torch')

    def lonlat2xyz(self, pred_up, pred_down):
        pred_up_xyz = self.et.lonlat2xyz(pred_up)
        s = -self.scale / pred_up_xyz[..., 1:2]
        pred_up_xyz *= s
        pred_down_xyz = self.et.lonlat2xyz(pred_down)
        s = self.scale / pred_down_xyz[..., 1:2]
        pred_down_xyz *= s
        return pred_up_xyz, pred_down_xyz

    def forward(self, pred_up, pred_down):
        pred_up_xyz, pred_down_xyz = self.lonlat2xyz(pred_up, pred_down)
        pred_up_xz = pred_up_xyz[..., ::2]
        pred_down_xz = pred_down_xyz[..., ::2]
        pred_up_norm = torch.norm(pred_up_xz, p=2, dim=-1)
        pred_down_norm = torch.norm(pred_down_xz, p=2, dim=-1)
        ratio = self.lsq_fit(pred_up_xz, pred_down_xz)
        return ratio

    def lsq_fit(self, pred_up_xz, pred_down_xz):
        device = pred_up_xz.device
        pred_up_xz = pred_up_xz.cpu().numpy()
        pred_down_xz = pred_down_xz.cpu().numpy()
        ratio = np.zeros(pred_up_xz.shape[0], dtype=np.float32)
        for i in range(pred_up_xz.shape[0]):
            up_xz = pred_up_xz[i, ...].copy()
            up_norm = np.linalg.norm(up_xz, axis=-1)
            down_xz = pred_down_xz[i, ...].copy()
            down_norm = np.linalg.norm(down_xz, axis=-1)
            init_ratio = 1 / np.mean(up_norm / down_norm, axis=-1)
            error_func = partial(errorCalculate, up_norm=up_norm, down_norm=down_norm)
            ret = least_squares(error_func, init_ratio, verbose=0)
            x = ret.x[0]
            ratio[i] = x
        ratio = torch.FloatTensor(ratio)
        return ratio


class Cube2Equirec(nn.Module):

    def __init__(self, cube_length, equ_h):
        super().__init__()
        self.cube_length = cube_length
        self.equ_h = equ_h
        equ_w = equ_h * 2
        self.equ_w = equ_w
        theta = (np.arange(equ_w) / (equ_w - 1) - 0.5) * 2 * np.pi
        phi = (np.arange(equ_h) / (equ_h - 1) - 0.5) * np.pi
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        planes = np.asarray([[0, 0, 1, 1], [0, 1, 0, -1], [0, 0, 1, -1], [1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 0, 1]])
        r_lst = np.array([[0, 1, 0], [0.5, 0, 0], [0, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [-0.5, 0, 0]]) * np.pi
        f = cube_length / 2.0
        self.K = np.array([[f, 0, (cube_length - 1) / 2.0], [0, f, (cube_length - 1) / 2.0], [0, 0, 1]])
        self.R_lst = [cv2.Rodrigues(x)[0] for x in r_lst]
        self.mask, self.XY = self._intersection(xyz, planes)

    def forward(self, x, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        assert x.shape[0] % 6 == 0
        equ_count = x.shape[0] // 6
        equi = torch.zeros(equ_count, x.shape[1], self.equ_h, self.equ_w)
        for i in range(6):
            now = x[i::6, ...]
            mask = self.mask[i]
            mask = mask[None, ...].repeat(equ_count, x.shape[1], 1, 1)
            XY = (self.XY[i][None, None, :, :].repeat(equ_count, 1, 1, 1) / (self.cube_length - 1) - 0.5) * 2
            sample = F.grid_sample(now, XY, mode=mode, align_corners=True)[..., 0, :]
            equi[mask] = sample.view(-1)
        return equi

    def _intersection(self, xyz, planes):
        abc = planes[:, :-1]
        depth = -planes[:, 3][None, None, ...] / np.dot(xyz, abc.T)
        depth[depth < 0] = np.inf
        arg = np.argmin(depth, axis=-1)
        depth = np.min(depth, axis=-1)
        pts = depth[..., None] * xyz
        mask_lst = []
        mapping_XY = []
        for i in range(6):
            mask = arg == i
            mask = np.tile(mask[..., None], [1, 1, 3])
            XY = np.dot(np.dot(pts[mask].reshape([-1, 3]), self.R_lst[i].T), self.K.T)
            XY = np.clip(XY[..., :2].copy() / XY[..., 2:], 0, self.cube_length - 1)
            mask_lst.append(mask[..., 0])
            mapping_XY.append(XY)
        mask_lst = [torch.BoolTensor(x) for x in mask_lst]
        mapping_XY = [torch.FloatTensor(x) for x in mapping_XY]
        return mask_lst, mapping_XY


class Equirec2Cube(nn.Module):

    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([[0, -180.0, 0], [90.0, 0, 0], [0, 0, 0], [0, 90, 0], [0, -90, 0], [-90, 0, 0]], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        self._getCubeGrid()

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        self.grids = []
        for R in self.R_lst:
            tmp = xyz @ R
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            self.grids.append(torch.FloatTensor(lonlat[None, ...]))

    def forward(self, batch, mode='bilinear'):
        [_, _, h, w] = batch.shape
        assert h == self.equ_h and w == self.equ_w
        assert mode in ['nearest', 'bilinear']
        out = []
        for grid in self.grids:
            grid = grid
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)
            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i::batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out


class EquirecRotate(nn.Module):

    def __init__(self, equ_h):
        super().__init__()
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        X = torch.arange(self.equ_w)[None, :, None].repeat(self.equ_h, 1, 1)
        Y = torch.arange(self.equ_h)[:, None, None].repeat(1, self.equ_w, 1)
        XY = torch.cat([X, Y], dim=-1).unsqueeze(0)
        self.grid = Conversion.XY2xyz(XY, shape=(self.equ_h, self.equ_w), mode='torch')

    def forward(self, equi, axis_angle, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        grid = self.grid.repeat(equi.shape[0], 1, 1, 1)
        R = Conversion.angle_axis_to_rotation_matrix(axis_angle)
        xyz = (R[:, None, None, ...] @ grid[..., None]).squeeze(-1)
        XY = Conversion.xyz2lonlat(xyz, clip=False, mode='torch')
        X, Y = torch.unbind(XY, dim=-1)
        XY = torch.cat([(X / math.pi).unsqueeze(-1), (Y / 0.5 / math.pi).unsqueeze(-1)], dim=-1)
        sample = F.grid_sample(equi, XY, mode=mode, align_corners=True)
        return sample


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvCompressH,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CustomPad,
     lambda: ([], {'pad_func': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Densenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (LR_PAD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NoOp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ZeroPad,
     lambda: ([], {'pad_s': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

