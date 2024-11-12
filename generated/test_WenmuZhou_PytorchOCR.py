
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


import inspect


from functools import partial


import re


from itertools import zip_longest


import numpy


import math


import itertools


import torch.cuda


import copy


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


import numbers


from collections import defaultdict


from torch.utils.data import Dataset


import string


from torch.utils.data import Sampler


import torch.distributed as dist


import random


import time


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import L1Loss


from torch.nn import MSELoss as L2Loss


from torch.nn import SmoothL1Loss


from torch import nn


from torch.nn import functional as F


import torchvision


from torch.nn import Linear


from torch.nn.init import xavier_uniform_


from torch.optim import lr_scheduler


import functools


from numpy.linalg import norm


from numpy.fft import ifft


import logging


class TorchModuleStr(torch.nn.Module):

    def __init__(self, net):
        super(TorchModuleStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


class ACELoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, ignore_index=0, reduction='none', soft_label=True, dim=-1)

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        B, N = predicts.shape[:2]
        div = torch.tensor([N]).astype('float32')
        predicts = nn.functional.softmax(predicts, dim=-1)
        aggregation_preds = torch.sum(predicts, dim=1)
        aggregation_preds = torch.divide(aggregation_preds, div)
        length = batch[2].astype('float32')
        batch = batch[3].astype('float32')
        batch[:, 0] = torch.subtract(div, length)
        batch = torch.divide(batch, div)
        loss = self.loss_func(aggregation_preds, batch)
        return {'loss_ace': loss}


class CELoss(nn.Module):

    def __init__(self, smoothing=False, with_all=False, ignore_index=-1, **kwargs):
        super(CELoss, self).__init__()
        if ignore_index >= 0:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.smoothing = smoothing
        self.with_all = with_all

    def forward(self, pred, batch):
        pred = pred['res']
        if isinstance(pred, dict):
            loss = {}
            loss_sum = []
            for name, logits in pred.items():
                if isinstance(logits, list):
                    logit_num = len(logits)
                    all_tgt = torch.cat([batch[1]] * logit_num, 0)
                    all_logits = torch.cat(logits, 0)
                    flt_logtis = all_logits.reshape([-1, all_logits.shape[2]])
                    flt_tgt = all_tgt.reshape([-1])
                else:
                    flt_logtis = logits.reshape([-1, logits.shape[2]])
                    flt_tgt = batch[1].reshape([-1])
                loss[name + '_loss'] = self.loss_func(flt_logtis, flt_tgt)
                loss_sum.append(loss[name + '_loss'])
            loss['loss'] = sum(loss_sum)
            return loss
        elif self.with_all:
            tgt = batch[1]
            pred = pred.reshape([-1, pred.shape[2]])
            tgt = tgt.reshape([-1])
            loss = self.loss_func(pred, tgt)
            return {'loss': loss}
        else:
            max_len = batch[2].max()
            tgt = batch[1][:, 1:2 + max_len]
            pred = pred.reshape([-1, pred.shape[2]])
            tgt = tgt.reshape([-1])
            if self.smoothing:
                eps = 0.1
                n_class = pred.shape[1]
                one_hot = F.one_hot(tgt, pred.shape[1])
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / -1
                log_prb = F.log_softmax(pred, dim=1)
                non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype, device=tgt.device))
                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).mean()
            else:
                loss = self.loss_func(pred, tgt)
            return {'loss': loss}


class KLJSLoss(object):

    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'], "mode can only be one of ['kl', 'KL', 'js', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction='mean', eps=1e-05):
        if self.mode.lower() == 'kl':
            loss = torch.multiply(p2, torch.log((p2 + eps) / (p1 + eps) + eps))
            loss += torch.multiply(p1, torch.log((p1 + eps) / (p2 + eps) + eps))
            loss *= 0.5
        elif self.mode.lower() == 'js':
            loss = torch.multiply(p2, torch.log((2 * p2 + eps) / (p1 + p2 + eps) + eps))
            loss += torch.multiply(p1, torch.log((2 * p1 + eps) / (p1 + p2 + eps) + eps))
            loss *= 0.5
        else:
            raise ValueError("The mode.lower() if KLJSLoss should be one of ['kl', 'js']")
        if reduction == 'mean':
            loss = torch.mean(loss, dim=[1, 2])
        elif reduction == 'none' or reduction is None:
            return loss
        else:
            loss = torch.sum(loss, dim=[1, 2])
        return loss


class DMLLoss(nn.Module):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ['softmax', 'sigmoid']
        if act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode='kl')

    def _kldiv(self, x, target):
        eps = 1e-10
        loss = target * (torch.log(target + eps) - x)
        loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log:
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Module):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode='l2', **kargs):
        super().__init__()
        assert mode in ['l1', 'l2', 'smooth_l1']
        if mode == 'l1':
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == 'l2':
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == 'smooth_l1':
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class LossFromOutput(nn.Module):

    def __init__(self, key='loss', reduction='none'):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return {'loss': loss}


class KLDivLoss(nn.Module):
    """
    KLDivLoss
    """

    def __init__(self):
        super().__init__()

    def _kldiv(self, x, target, mask=None):
        eps = 1e-10
        loss = target * (torch.log(target + eps) - x)
        if mask is not None:
            loss = loss.flatten(0, 1).sum(dim=1)
            loss = loss.masked_select(mask).mean()
        else:
            loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, logits_s, logits_t, mask=None):
        log_out_s = F.log_softmax(logits_s, dim=-1)
        out_t = F.softmax(logits_t, dim=-1)
        loss = self._kldiv(log_out_s, out_t, mask)
        return loss


class DKDLoss(nn.Module):
    """
    KLDivLoss
    """

    def __init__(self, temperature=1.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdim=True)
        t2 = (t * mask2).sum(dim=1, keepdim=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def _kl_div(self, x, label, mask=None):
        y = (label * (torch.log(label + 1e-10) - x)).sum(dim=1)
        if mask is not None:
            y = y.masked_select(mask).mean()
        else:
            y = y.mean()
        return y

    def forward(self, logits_student, logits_teacher, target, mask=None):
        gt_mask = F.one_hot(target.reshape([-1]), num_classes=logits_student.shape[-1])
        other_mask = 1 - gt_mask
        logits_student = logits_student.flatten(0, 1)
        logits_teacher = logits_teacher.flatten(0, 1)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = self._kl_div(log_pred_student, pred_teacher) * self.temperature ** 2
        pred_teacher_part2 = F.softmax(logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(logits_student / self.temperature - 1000.0 * gt_mask, dim=1)
        nckd_loss = self._kl_div(log_pred_student_part2, pred_teacher_part2) * self.temperature ** 2
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return loss


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn(self.num_classes, self.feat_dim, dtype=torch.float)
        if center_file_path is not None:
            assert os.path.exists(center_file_path), f'center path({center_file_path}) must exist when it is not None.'
            with open(center_file_path, 'rb') as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.from_numpy(char_dict[key])

    def __call__(self, predicts, batch):
        features, predicts = predicts['feat'], predicts['res']
        feats_reshape = torch.reshape(features, [-1, features.shape[-1]])
        label = torch.argmax(predicts, dim=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])
        batch_size = feats_reshape.shape[0]
        square_feat = torch.sum(torch.square(feats_reshape), dim=1, keepdim=True)
        square_feat = square_feat.expand([batch_size, self.num_classes])
        square_center = torch.sum(torch.square(self.centers), dim=1, keepdim=True)
        square_center = square_center.expand([self.num_classes, batch_size])
        square_center = torch.permute(square_center, [1, 0])
        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape, torch.permute(self.centers, [1, 0]))
        distmat = distmat - 2.0 * feat_dot_center
        classes = torch.arange(self.num_classes, dtype=torch.int)
        label = torch.unsqueeze(label, 1).expand((batch_size, self.num_classes))
        mask = torch.equal(classes.expand([batch_size, self.num_classes]), label)
        dist = torch.multiply(distmat, mask)
        loss = torch.sum(torch.clip(dist, min=1e-12, max=1000000000000.0)) / batch_size
        return {'loss_center': loss}


class ClsLoss(nn.Module):

    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predicts, batch):
        label = batch[1].long()
        loss = self.loss_func(predicts['res'], label)
        return {'loss': loss}


class CombinedLoss(nn.Module):
    """
    CombinedLoss:
        a combionation of loss function
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(loss_config_list, list), 'operator config should be a list'
        for config in loss_config_list:
            assert isinstance(config, dict) and len(config) == 1, 'yaml format error'
            name = list(config)[0]
            param = config[name]
            assert 'weight' in param, 'weight must be in param, but param just contains {}'.format(param.keys())
            self.loss_weight.append(param.pop('weight'))
            self.loss_func.append(eval(name)(**param))

    def forward(self, input, batch, **kargs):
        loss_dict = {}
        loss_all = 0.0
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kargs)
            if isinstance(loss, torch.Tensor):
                loss = {'loss_{}_{}'.format(str(loss), idx): loss}
            weight = self.loss_weight[idx]
            loss = {key: (loss[key] * weight) for key in loss}
            if 'loss' in loss:
                loss_all += loss['loss'][0] if loss['loss'].ndim == 1 else loss['loss']
            else:
                loss_all += sum(loss.values())
            loss_dict.update(loss)
        loss_dict['loss'] = loss_all
        return loss_dict


class BCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class DiceLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]
        input = F.sigmoid(input)
        input = input.reshape((batch_size, -1))
        target = target.reshape((batch_size, -1)).float()
        mask = mask.reshape((batch_size, -1)).float()
        input = input * mask
        target = target * mask
        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = 2 * a / (b + c)
        loss = 1 - d
        loss = self.loss_weight * loss
        if reduce:
            loss = torch.mean(loss)
        return loss


class MaskL1Loss(nn.Module):

    def __init__(self, eps=1e-06):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class BalanceLoss(nn.Module):

    def __init__(self, balance_loss=True, main_loss_type='DiceLoss', negative_ratio=3, return_origin=False, eps=1e-06, **kwargs):
        """
               The BalanceLoss for Differentiable Binarization text detection
               args:
                   balance_loss (bool): whether balance loss or not, default is True
                   main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                       'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
                   negative_ratio (int|float): float, default is 3.
                   return_origin (bool): whether return unbalanced loss or not, default is False.
                   eps (float): default is 1e-6.
               """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps
        if self.main_loss_type == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == 'Euclidean':
            self.loss = nn.MSELoss()
        elif self.main_loss_type == 'DiceLoss':
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == 'BCELoss':
            self.loss = BCELoss(reduction='none')
        elif self.main_loss_type == 'MaskL1Loss':
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = ['CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss']
            raise Exception('main_loss_type in BalanceLoss() can only be one of {}'.format(loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)
        if not self.balance_loss:
            return loss
        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss
        return balance_loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        np_coord = np_coord.reshape((-1, 2))
        self.coord = nn.Parameter(torch.from_numpy(np_coord))
        self.coord.requires_grade = False

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-06):
        batch_size = input.shape[0]
        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        loss = loss.reshape((batch_size, -1)).float()
        mask = mask.reshape((batch_size, -1)).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + eps)
        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance, training_mask):
        with torch.no_grad():
            select_distance_list = []
            for i in range(2):
                tmp1 = distance[i, :]
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                select_distance_list.append(tmp2.unsqueeze(0))
            select_distance = torch.cat(select_distance_list, dim=0)
            off_points = self.coord.float() + 10 * select_distance.permute((1, 0))
            off_points = off_points.float()
            off_points = torch.clip(off_points, 0, distance.shape[-1] - 1)
            selected_mask = gt_instance[self.coord[:, 1], self.coord[:, 0]] != gt_kernel_instance[off_points[:, 1], off_points[:, 0]]
            selected_mask = selected_mask.reshape((1, -1, distance.shape[-1])).long()
            selected_training_mask = selected_mask * training_mask
            return selected_training_mask

    def forward(self, distances, gt_instances, gt_kernel_instances, training_masks, gt_distances, reduce=True):
        selected_training_masks = []
        for i in range(distances.shape[0]):
            selected_training_masks.append(self.select_single(distances[i, :, :, :], gt_instances[i, :, :], gt_kernel_instances[i, :, :], training_masks[i, :, :]))
        selected_training_masks = torch.cat(selected_training_masks, 0).float()
        loss = self.forward_single(distances, gt_distances, selected_training_masks, self.beta)
        loss = self.loss_weight * loss
        with torch.no_grad():
            batch_size = distances.shape[0]
            false_num = selected_training_masks.reshape((batch_size, -1))
            false_num = false_num.sum(dim=-1)
            total_num = training_masks.reshape((batch_size, -1)).float()
            total_num = total_num.sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-06)
        if reduce:
            loss = torch.mean(loss)
        return loss, iou_text


def iou_single(a, b, mask, n_class):
    EPS = 1e-06
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()
        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]
    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))
    iou = torch.zeros(batch_size).float()
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)
    if reduce:
        iou = torch.mean(iou)
    return iou


def ohem_single(score, gt_text, training_mask):
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    if pos_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
        return selected_mask
    neg_num = int(torch.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
        return selected_mask
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]
    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))
    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


class CTLoss(nn.Module):

    def __init__(self):
        super(CTLoss, self).__init__()
        self.kernel_loss = DiceLoss()
        self.loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)

    def forward(self, preds, batch):
        imgs = batch[0]
        out = preds['maps']
        gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances = batch[1:]
        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)
        loss_kernel = self.kernel_loss(kernels, gt_kernels, selected_masks, reduce=False)
        iou_kernel = iou((kernels > 0).long(), gt_kernels, training_masks, reduce=False)
        losses = dict(loss_kernels=loss_kernel)
        loss_loc, iou_text = self.loc_loss(distances, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances, reduce=False)
        losses.update(dict(loss_loc=loss_loc))
        loss_all = loss_kernel + loss_loc
        losses = {'loss': loss_all}
        return losses


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self, balance_loss=True, main_loss_type='DiceLoss', alpha=5, beta=10, ohem_ratio=3, eps=1e-06, **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(balance_loss=balance_loss, main_loss_type=main_loss_type, negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts['res']
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]
        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map, label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map, label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps
        if 'distance_maps' in predicts.keys():
            distance_maps = predicts['distance_maps']
            cbn_maps = predicts['cbn_maps']
            cbn_loss = self.bce_loss(cbn_maps[:, 0, :, :], label_shrink_map, label_shrink_mask)
        else:
            dis_loss = torch.tensor([0.0], device=shrink_maps.device)
            cbn_loss = torch.tensor([0.0], device=shrink_maps.device)
        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
        losses = {'loss': loss_all + cbn_loss, 'loss_shrink_maps': loss_shrink_maps, 'loss_threshold_maps': loss_threshold_maps, 'loss_binary_maps': loss_binary_maps, 'loss_cbn': cbn_loss}
        return losses


class DRRGLoss(nn.Module):

    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = 1.0

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert torch.all(pred >= 0) and torch.all(pred <= 1)
        assert torch.all(gt >= 0) and torch.all(gt <= 1)
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = torch.sum(loss * positive)
            negative_loss = loss * negative
            negative_count = min(int(negative.sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = torch.tensor(0.0)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative
            negative_count = 100
        negative_loss, _ = torch.topk(negative_loss.reshape([-1]), negative_count)
        balance_loss = (positive_loss + torch.sum(negative_loss)) / (float(positive_count + negative_count) + 1e-05)
        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.reshape([-1])
        loss = F.cross_entropy(gcn_pred, gt_labels)
        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        batch_size = len(bitmasks)
        results = []
        kernel = []
        for batch_inx in range(batch_size):
            mask = bitmasks[batch_inx]
            mask_sz = mask.shape
            pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
            mask = F.pad(mask, pad, mode='constant', value=0)
            kernel.append(mask)
        kernel = torch.stack(kernel)
        results.append(kernel)
        return results

    def forward(self, preds, labels):
        """Compute Drrg loss.
        """
        assert isinstance(preds, tuple)
        gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map = labels[1:8]
        downsample_ratio = self.downsample_ratio
        pred_maps, gcn_data = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.shape
        mapping = {'gt_text_mask': gt_text_mask.float(), 'gt_center_region_mask': gt_center_region_mask.float(), 'gt_mask': gt_mask.float(), 'gt_top_height_map': gt_top_height_map.float(), 'gt_bot_height_map': gt_bot_height_map.float(), 'gt_sin_map': gt_sin_map.float(), 'gt_cos_map': gt_cos_map.float()}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 0.01:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [(item * downsample_ratio) for item in gt[key]]
            gt[key] = [item for item in gt[key]]
        scale = torch.sqrt(1.0 / (pred_sin_map ** 2 + pred_cos_map ** 2 + 1e-08))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale
        loss_text = self.balance_bce_loss(F.sigmoid(pred_text_region), gt['gt_text_mask'][0], gt['gt_mask'][0])
        text_mask = gt['gt_text_mask'][0] * gt['gt_mask'][0]
        negative_text_mask = (1 - gt['gt_text_mask'][0]) * gt['gt_mask'][0]
        loss_center_map = F.binary_cross_entropy(F.sigmoid(pred_center_region), gt['gt_center_region_mask'][0], reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = torch.sum(loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            loss_center_positive = torch.tensor(0.0)
        loss_center_negative = torch.sum(loss_center_map * negative_text_mask) / torch.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative
        center_mask = gt['gt_center_region_mask'][0] * gt['gt_mask'][0]
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.shape
            ones = torch.ones(map_sz, dtype=torch.float32)
            loss_top = F.smooth_l1_loss(pred_top_height_map / (gt['gt_top_height_map'][0] + 0.01), ones, reduction='none')
            loss_bot = F.smooth_l1_loss(pred_bot_height_map / (gt['gt_bot_height_map'][0] + 0.01), ones, reduction='none')
            gt_height = gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0]
            loss_height = torch.sum(torch.log(gt_height + 1) * (loss_top + loss_bot) * center_mask) / torch.sum(center_mask)
            loss_sin = torch.sum(F.smooth_l1_loss(pred_sin_map, gt['gt_sin_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(F.smooth_l1_loss(pred_cos_map, gt['gt_cos_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
        else:
            loss_height = torch.tensor(0.0)
            loss_sin = torch.tensor(0.0)
            loss_cos = torch.tensor(0.0)
        loss_gcn = self.gcn_loss(gcn_data)
        loss = loss_text + loss_center + loss_height + loss_sin + loss_cos + loss_gcn
        results = dict(loss=loss, loss_text=loss_text, loss_center=loss_center, loss_height=loss_height, loss_sin=loss_sin, loss_cos=loss_cos, loss_gcn=loss_gcn)
        return results


class EASTLoss(nn.Module):
    """
    """

    def __init__(self, eps=1e-06, **kwargs):
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        l_score, l_geo, l_mask = labels[1:]
        f_score = predicts['f_score']
        f_geo = predicts['f_geo']
        dice_loss = self.dice_loss(f_score, l_score, l_mask)
        channels = 8
        l_geo_split = torch.split(l_geo, split_size_or_sections=channels + 1, dim=1)
        f_geo_split = torch.split(f_geo, split_size_or_sections=channels, dim=1)
        smooth_l1 = 0
        for i in range(0, channels):
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = (abs_geo_diff < l_score).float()
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = torch.mean(smooth_l1 * l_score)
        dice_loss = dice_loss * 0.01
        total_loss = dice_loss + smooth_l1_loss
        losses = {'loss': total_loss, 'dice_loss': dice_loss, 'smooth_l1_loss': smooth_l1_loss}
        return losses


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class FCELoss(nn.Module):
    """The class for implementing FCENet loss
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.0):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio

    def forward(self, preds, labels):
        assert isinstance(preds, dict)
        preds = preds['levels']
        p3_maps, p4_maps, p5_maps = labels[1:]
        assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5, 'fourier degree not equal in FCEhead and FCEtarget'
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = torch.tensor(np.stack(maps))
        losses = multi_apply(self.forward_single, preds, gts)
        loss_tr = torch.tensor(0.0).float()
        loss_tcl = torch.tensor(0.0).float()
        loss_reg_x = torch.tensor(0.0).float()
        loss_reg_y = torch.tensor(0.0).float()
        loss_all = torch.tensor(0.0).float()
        for idx, loss in enumerate(losses):
            loss_all += sum(loss)
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)
        results = dict(loss=loss_all, loss_text=loss_tr, loss_center=loss_tcl, loss_reg_x=loss_reg_x, loss_reg_y=loss_reg_y)
        return results

    def forward_single(self, pred, gt):
        cls_pred = torch.permute(pred[0], (0, 2, 3, 1))
        reg_pred = torch.permute(pred[1], (0, 2, 3, 1))
        gt = torch.permute(gt, (0, 2, 3, 1))
        k = 2 * self.fourier_degree + 1
        tr_pred = torch.permute(cls_pred[:, :, :, :2], (-1, 2))
        tcl_pred = torch.permute(cls_pred[:, :, :, 2:], (-1, 2))
        x_pred = torch.permute(reg_pred[:, :, :, 0:k], (-1, k))
        y_pred = torch.permute(reg_pred[:, :, :, k:2 * k], (-1, k))
        tr_mask = gt[:, :, :, :1].reshape([-1])
        tcl_mask = gt[:, :, :, 1:2].reshape([-1])
        train_mask = gt[:, :, :, 2:3].reshape([-1])
        x_map = torch.permute(gt[:, :, :, 3:3 + k], (-1, k))
        y_map = torch.permute(gt[:, :, :, 3 + k:], (-1, k))
        tr_train_mask = (train_mask * tr_mask).astype('bool')
        tr_train_mask2 = torch.cat([tr_train_mask.unsqueeze(1), tr_train_mask.unsqueeze(1)], dim=1)
        loss_tr = self.ohem(tr_pred, tr_mask, train_mask)
        loss_tcl = torch.tensor(0.0).float()
        tr_neg_mask = tr_train_mask.logical_not()
        tr_neg_mask2 = torch.cat([tr_neg_mask.unsqueeze(1), tr_neg_mask.unsqueeze(1)], dim=1)
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred.masked_select(tr_train_mask2).reshape([-1, 2]), tcl_mask.masked_select(tr_train_mask).astype('int64'))
            loss_tcl_neg = F.cross_entropy(tcl_pred.masked_select(tr_neg_mask2).reshape([-1, 2]), tcl_mask.masked_select(tr_neg_mask).astype('int64'))
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg
        loss_reg_x = torch.tensor(0.0).float()
        loss_reg_y = torch.tensor(0.0).float()
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask.masked_select(tr_train_mask.astype('bool')).astype('float32') + tcl_mask.masked_select(tr_train_mask.astype('bool')).astype('float32')) / 2
            weight = weight.reshape([-1, 1])
            ft_x, ft_y = self.fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier2poly(x_pred, y_pred)
            dim = ft_x.shape[1]
            tr_train_mask3 = torch.cat([tr_train_mask.unsqueeze(1) for i in range(dim)], dim=1)
            loss_reg_x = torch.mean(weight * F.smooth_l1_loss(ft_x_pre.masked_select(tr_train_mask3).reshape([-1, dim]), ft_x.masked_select(tr_train_mask3).reshape([-1, dim]), reduction='none'))
            loss_reg_y = torch.mean(weight * F.smooth_l1_loss(ft_y_pre.masked_select(tr_train_mask3).reshape([-1, dim]), ft_y.masked_select(tr_train_mask3).reshape([-1, dim]), reduction='none'))
        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def ohem(self, predict, target, train_mask):
        pos = (target * train_mask).astype('bool')
        neg = ((1 - target) * train_mask).astype('bool')
        pos2 = torch.concat([pos.unsqueeze(1), pos.unsqueeze(1)], dim=1)
        neg2 = torch.concat([neg.unsqueeze(1), neg.unsqueeze(1)], dim=1)
        n_pos = pos.astype('float32').sum()
        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict.masked_select(pos2).reshape([-1, 2]), target.masked_select(pos).astype('int64'), reduction='sum')
            loss_neg = F.cross_entropy(predict.masked_select(neg2).reshape([-1, 2]), target.masked_select(neg).astype('int64'), reduction='none')
            n_neg = min(int(neg.astype('float32').sum().item()), int(self.ohem_ratio * n_pos.astype('float32')))
        else:
            loss_pos = torch.tensor(0.0)
            loss_neg = F.cross_entropy(predict.masked_select(neg2).reshape([-1, 2]), target.masked_select(neg).astype('int64'), reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).astype('float32')

    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """
        k_vect = torch.arange(-self.fourier_degree, self.fourier_degree + 1, dtype=torch.float).reshape([-1, 1])
        i_vect = torch.arange(0, self.num_sample, dtype=torch.float).reshape([1, -1])
        transform_matrix = 2 * np.pi / self.num_sample * torch.matmul(k_vect, i_vect)
        x1 = torch.einsum('ak, kn-> an', real_maps, torch.cos(transform_matrix))
        x2 = torch.einsum('ak, kn-> an', imag_maps, torch.sin(transform_matrix))
        y1 = torch.einsum('ak, kn-> an', real_maps, torch.sin(transform_matrix))
        y2 = torch.einsum('ak, kn-> an', imag_maps, torch.cos(transform_matrix))
        x_maps = x1 - x2
        y_maps = y1 + y2
        return x_maps, y_maps


class PSELoss(nn.Module):

    def __init__(self, alpha, ohem_ratio=3, kernel_sample_mask='pred', reduction='sum', eps=1e-06, **kwargs):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none']
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs, labels):
        predicts = outputs['maps']
        predicts = F.interpolate(predicts, scale_factor=4)
        texts = predicts[:, 0, :, :]
        kernels = predicts[:, 1:, :, :]
        gt_texts, gt_kernels, training_masks = labels[1:]
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.dice_loss(texts, gt_texts, selected_masks)
        iou_text = iou((texts > 0).astype('int64'), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)
        loss_kernels = []
        if self.kernel_sample_mask == 'gt':
            selected_masks = gt_texts * training_masks
        elif self.kernel_sample_mask == 'pred':
            selected_masks = (F.sigmoid(texts) > 0.5).float() * training_masks
        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).astype('int64'), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))
        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernels
        losses['loss'] = loss
        if self.reduction == 'sum':
            losses = {x: torch.sum(v) for x, v in losses.items()}
        elif self.reduction == 'mean':
            losses = {x: torch.mean(v) for x, v in losses.items()}
        return losses

    def dice_loss(self, input, target, mask):
        input = F.sigmoid(input)
        input = input.reshape([input.shape[0], -1])
        target = target.reshape([target.shape[0], -1])
        mask = mask.reshape([mask.shape[0], -1])
        input = input * mask
        target = target * mask
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps
        d = 2 * a / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask, ohem_ratio=3):
        pos_num = int(torch.sum((gt_text > 0.5).astype('float32'))) - int(torch.sum(torch.logical_and(gt_text > 0.5, training_mask <= 0.5).float()))
        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
            return selected_mask
        neg_num = int(torch.sum((gt_text <= 0.5).astype('float32')))
        neg_num = int(min(pos_num * ohem_ratio, neg_num))
        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
            return selected_mask
        neg_score = torch.masked_select(score, gt_text <= 0.5)
        neg_score_sorted = torch.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = torch.logical_and(torch.logical_or(score >= threshold, gt_text > 0.5), training_mask > 0.5)
        selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks, ohem_ratio=3):
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :], ohem_ratio))
        selected_masks = torch.cat(selected_masks, 0).float()
        return selected_masks


class SASTLoss(nn.Module):
    """
    """

    def __init__(self, eps=1e-06, **kwargs):
        super(SASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor
        """
        f_score = predicts['f_score']
        f_border = predicts['f_border']
        f_tvo = predicts['f_tvo']
        f_tco = predicts['f_tco']
        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]
        intersection = torch.sum(f_score * l_score * l_mask)
        union = torch.sum(f_score * l_mask) + torch.sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-05)
        l_border_split, l_border_norm = torch.split(l_border, split_size_or_sections=[4, 1], dim=1)
        f_border_split = f_border
        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = l_border_norm.expand(border_ex_shape)
        l_border_score = l_score.expand(border_ex_shape)
        l_border_mask = l_mask.expand(border_ex_shape)
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.float()
        border_sign.requires_grade = False
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / (torch.sum(l_border_score * l_border_mask) + 1e-05)
        l_tvo_split, l_tvo_norm = torch.split(l_tvo, split_size_or_sections=[8, 1], dim=1)
        f_tvo_split = f_tvo
        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        l_tvo_norm_split = l_tvo_norm.expand(tvo_ex_shape)
        l_tvo_score = l_tvo_norm.expand(tvo_ex_shape)
        l_tvo_mask = l_tvo_norm.expand(tvo_ex_shape)
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = torch.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = tvo_sign.float()
        tvo_sign.requires_grade = False
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = torch.sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / (torch.sum(l_tvo_score * l_tvo_mask) + 1e-05)
        l_tco_split, l_tco_norm = torch.split(l_tco, split_size_or_sections=[2, 1], dim=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])
        l_tco_norm_split = l_tco_norm.expand(tco_ex_shape)
        l_tco_score = l_score.expand(tco_ex_shape)
        l_tco_mask = l_mask.expand(tco_ex_shape)
        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = torch.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = tco_sign.float()
        tco_sign.requires_grade = False
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = torch.sum(tco_out_loss * l_tco_score * l_tco_mask) / (torch.sum(l_tco_score * l_tco_mask) + 1e-05)
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = score_loss * score_lw + border_loss * border_lw + tvo_loss * tvo_lw + tco_loss * tco_lw
        losses = {'loss': total_loss, 'score_loss': score_loss, 'border_loss': border_loss, 'tvo_loss': tvo_loss, 'tco_loss': tco_loss}
        return losses


def _sum_loss(loss_dict):
    if 'loss' in loss_dict.keys():
        return loss_dict
    else:
        loss_dict['loss'] = 0.0
        for k, value in loss_dict.items():
            if k == 'loss':
                continue
            else:
                loss_dict['loss'] += value[0] if value.ndim == 1 else value
        return loss_dict


class DistillationDMLLoss(DMLLoss):
    """
    """

    def __init__(self, model_name_pairs=[], act=None, use_log=False, key=None, multi_head=False, dis_head='ctc', maps_name=None, name='dml'):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == 'thrink_maps':
                new_outs[k] = outs[:, 0, :, :]
            elif k == 'threshold_maps':
                new_outs[k] = outs[:, 1, :, :]
            elif k == 'binary_maps':
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    loss = super().forward(out1[self.dis_head]['res'], out2[self.dis_head]['res'])
                else:
                    loss = super().forward(out1['res'], out2['res'])
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict['{}_{}_{}_{}'.format(key, pair[0], pair[1], idx)] = loss[key]
                else:
                    loss_dict['{}_{}'.format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict['{}_{}_{}_{}_{}'.format(key, pair[0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        loss_dict['{}_{}_{}'.format(self.name, self.maps_name[_c], idx)] = loss
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationKLDivLoss(KLDivLoss):
    """
    """

    def __init__(self, model_name_pairs=[], key=None, multi_head=False, dis_head='ctc', maps_name=None, name='kl_div'):
        super().__init__()
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == 'thrink_maps':
                new_outs[k] = outs[:, 0, :, :]
            elif k == 'threshold_maps':
                new_outs[k] = outs[:, 1, :, :]
            elif k == 'binary_maps':
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1:2 + max_len]
                    tgt = tgt.reshape([-1])
                    non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype))
                    loss = super().forward(out1[self.dis_head], out2[self.dis_head], non_pad_mask)
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict['{}_{}_{}_{}'.format(key, pair[0], pair[1], idx)] = loss[key]
                else:
                    loss_dict['{}_{}'.format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict['{}_{}_{}_{}_{}'.format(key, pair[0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        loss_dict['{}_{}_{}'.format(self.name, self.maps_name[_c], idx)] = loss
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDKDLoss(DKDLoss):
    """
    """

    def __init__(self, model_name_pairs=[], key=None, multi_head=False, dis_head='ctc', maps_name=None, name='dkd', temperature=1.0, alpha=1.0, beta=1.0):
        super().__init__(temperature, alpha, beta)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == 'thrink_maps':
                new_outs[k] = outs[:, 0, :, :]
            elif k == 'threshold_maps':
                new_outs[k] = outs[:, 1, :, :]
            elif k == 'binary_maps':
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1:2 + max_len]
                    tgt = tgt.reshape([-1])
                    non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype, device=tgt.device))
                    loss = super().forward(out1[self.dis_head], out2[self.dis_head], tgt, non_pad_mask)
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict['{}_{}_{}_{}'.format(key, pair[0], pair[1], idx)] = loss[key]
                else:
                    loss_dict['{}_{}'.format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict['{}_{}_{}_{}_{}'.format(key, pair[0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        loss_dict['{}_{}_{}'.format(self.name, self.maps_name[_c], idx)] = loss
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationNRTRDMLLoss(DistillationDMLLoss):
    """
    """

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.multi_head:
                max_len = batch[3].max()
                tgt = batch[2][:, 1:2 + max_len]
                tgt = tgt.reshape([-1])
                non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype))
                loss = super().forward(out1[self.dis_head], out2[self.dis_head], non_pad_mask)
            else:
                loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict['{}_{}_{}_{}'.format(key, pair[0], pair[1], idx)] = loss[key]
            else:
                loss_dict['{}_{}'.format(self.name, idx)] = loss
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationNRTRLoss(CELoss):

    def __init__(self, model_name_list=[], key=None, multi_head=False, smoothing=True, name='loss_nrtr', **kwargs):
        super().__init__(smoothing=smoothing)
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert 'nrtr' in out, 'multi head has multi out'
                loss = super().forward({'res': out['nrtr']}, batch[:1] + batch[2:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict['{}_{}_{}'.format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict['{}_{}'.format(self.name, model_name)] = loss
        return loss_dict


class DistillationDBLoss(DBLoss):

    def __init__(self, model_name_list=[], balance_loss=True, main_loss_type='DiceLoss', alpha=5, beta=10, ohem_ratio=3, eps=1e-06, name='db', **kwargs):
        super().__init__(balance_loss, main_loss_type, alpha, beta, ohem_ratio, eps)
        self.model_name_list = model_name_list
        self.name = name
        self.key = None

    def forward(self, predicts, batch):
        loss_dict = {}
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss.keys():
                    if key == 'loss':
                        continue
                    name = '{}_{}_{}'.format(self.name, model_name, key)
                    loss_dict[name] = loss[key]
            else:
                loss_dict['{}_{}'.format(self.name, model_name)] = loss
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDilaDBLoss(DBLoss):

    def __init__(self, model_name_pairs=[], key=None, balance_loss=True, main_loss_type='DiceLoss', alpha=5, beta=10, ohem_ratio=3, eps=1e-06, kd_loss=None, name='dila_dbloss'):
        super().__init__(balance_loss, main_loss_type, alpha, beta, ohem_ratio, eps)
        self.model_name_pairs = model_name_pairs
        self.name = name
        self.key = key
        self.kd_loss = None
        if kd_loss is not None:
            self.kd_loss = eval(kd_loss.pop('name'))(**kd_loss)

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            stu_outs = predicts[pair[0]]
            tch_outs = predicts[pair[1]]
            if self.key is not None:
                stu_preds = stu_outs[self.key]
                tch_preds = tch_outs[self.key]
            stu_shrink_maps = stu_preds[:, 0, :, :]
            stu_binary_maps = stu_preds[:, 2, :, :]
            dilation_w = np.array([[1, 1], [1, 1]])
            th_shrink_maps = tch_preds[:, 0, :, :]
            if self.kd_loss is not None:
                B = stu_shrink_maps.shape[0]
                kd_loss = self.kd_loss(stu_shrink_maps.reshape([B, -1]), th_shrink_maps.reshape([B, -1]))
                k = f'{self.name}_{pair[0]}_{pair[1]}_kd'
                loss_dict[k] = kd_loss
            th_shrink_maps = th_shrink_maps.cpu().numpy() > 0.3
            dilate_maps = np.zeros_like(th_shrink_maps).astype(np.float32)
            for i in range(th_shrink_maps.shape[0]):
                dilate_maps[i] = cv2.dilate(th_shrink_maps[i, :, :].astype(np.uint8), dilation_w)
            th_shrink_maps = torch.tensor(dilate_maps, device=stu_shrink_maps.device)
            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = batch[1:]
            bce_loss = self.alpha * self.bce_loss(stu_shrink_maps, th_shrink_maps, label_shrink_mask)
            loss_binary_maps = self.dice_loss(stu_binary_maps, th_shrink_maps, label_shrink_mask)
            k = '{}_{}_{}'.format(self.name, pair[0], pair[1])
            loss_dict[k] = bce_loss + loss_binary_maps
        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDistanceLoss(DistanceLoss):
    """
    """

    def __init__(self, mode='l2', model_name_pairs=[], key=None, name='loss_distance', **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + '_l2'

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict['{}_{}_{}'.format(self.name, key, idx)] = loss[key]
            else:
                loss_dict['{}_{}_{}_{}'.format(self.name, pair[0], pair[1], idx)] = loss
        return loss_dict


class DistillationLossFromOutput(LossFromOutput):

    def __init__(self, reduction='none', model_name_list=[], dist_key=None, key='loss', name='loss_re'):
        super().__init__(key=key, reduction=reduction)
        self.model_name_list = model_name_list
        self.name = name
        self.dist_key = dist_key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.dist_key is not None:
                out = out[self.dist_key]
            loss = super().forward(out, batch)
            loss_dict['{}_{}'.format(self.name, model_name)] = loss['loss']
        return loss_dict


class DistillationSERDMLLoss(DMLLoss):
    """
    """

    def __init__(self, act='softmax', use_log=True, num_classes=7, model_name_pairs=[], key=None, name='loss_dml_ser'):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.name = name
        self.num_classes = num_classes
        self.model_name_pairs = model_name_pairs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            out1 = out1.reshape([-1, out1.shape[-1]])
            out2 = out2.reshape([-1, out2.shape[-1]])
            attention_mask = batch[2]
            if attention_mask is not None:
                active_output = attention_mask.reshape([-1]) == 1
                out1 = out1[active_output]
                out2 = out2[active_output]
            loss_dict['{}_{}'.format(self.name, idx)] = super().forward(out1, out2)
        return loss_dict


class DistillationVQADistanceLoss(DistanceLoss):

    def __init__(self, mode='l2', model_name_pairs=[], key=None, index=None, name='loss_distance', **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.index = index
        self.model_name_pairs = model_name_pairs
        self.name = name + '_l2'

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            attention_mask = batch[2]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
                if self.index is not None:
                    out1 = out1[:, self.index, :, :]
                    out2 = out2[:, self.index, :, :]
                if attention_mask is not None:
                    max_len = attention_mask.shape[-1]
                    out1 = out1[:, :max_len]
                    out2 = out2[:, :max_len]
                out1 = out1.reshape([-1, out1.shape[-1]])
                out2 = out2.reshape([-1, out2.shape[-1]])
            if attention_mask is not None:
                active_output = attention_mask.reshape([-1]) == 1
                out1 = out1[active_output]
                out2 = out2[active_output]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict['{}_{}nohu_{}'.format(self.name, key, idx)] = loss[key]
            else:
                loss_dict['{}_{}_{}_{}'.format(self.name, pair[0], pair[1], idx)] = loss
        return loss_dict


class CTCDKDLoss(nn.Module):
    """
    KLDivLoss
    """

    def __init__(self, temperature=0.5, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-06
        self.t = temperature
        self.act = nn.Softmax(dim=-1)
        self.use_log = True

    def kl_loss(self, p1, p2):
        loss = torch.multiply(p2, torch.log((p2 + self.eps) / (p1 + self.eps) + self.eps))
        bs = loss.shape[0]
        loss = torch.sum(loss) / bs
        return loss

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdim=True)
        t2 = (t * mask2).sum(dim=1, keepdim=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def multi_label_mask(self, targets):
        targets = targets.astype('int32')
        res = F.one_hot(targets, num_classes=11465)
        mask = torch.clip(torch.sum(res, dim=1), 0, 1)
        mask[:, 0] = 0
        return mask

    def forward(self, logits_student, logits_teacher, targets, mask=None):
        gt_mask = self.multi_label_mask(targets)
        other_mask = torch.ones_like(gt_mask) - gt_mask
        pred_student = F.softmax(logits_student / self.temperature, dim=-1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)
        pred_student = torch.mean(pred_student, dim=1)
        pred_teacher = torch.mean(pred_teacher, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        tckd_loss = self.kl_loss(pred_student, pred_teacher)
        gt_mask_ex = gt_mask.expand_as(logits_teacher)
        pred_teacher_part2 = F.softmax(logits_teacher / self.temperature - 1000.0 * gt_mask_ex, dim=-1)
        pred_student_part2 = F.softmax(logits_student / self.temperature - 1000.0 * gt_mask_ex, dim=-1)
        pred_teacher_part2 = torch.mean(pred_teacher_part2, dim1)
        pred_student_part2 = torch.mean(pred_student_part2, dim=1)
        nckd_loss = self.kl_loss(pred_student_part2, pred_teacher_part2)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return loss


class KLCTCLogits(nn.Module):

    def __init__(self, weight=1.0, reduction='mean', mode='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-06
        self.t = 0.5
        self.act = nn.Softmax(dim=-1)
        self.use_log = True
        self.mode = mode
        self.ctc_dkd_loss = CTCDKDLoss()

    def kl_loss(self, p1, p2):
        loss = torch.multiply(p2, torch.log((p2 + self.eps) / (p1 + self.eps) + self.eps))
        bs = loss.shape[0]
        loss = torch.sum(loss) / bs
        return loss

    def forward_meanmax(self, stu_out, tea_out):
        stu_out = torch.mean(F.softmax(stu_out / self.t, dim=-1), dim=1)
        tea_out = torch.mean(F.softmax(tea_out / self.t, dim=-1), dim=1)
        loss = self.kl_loss(stu_out, tea_out)
        return loss

    def forward_meanlog(self, stu_out, tea_out):
        stu_out = torch.mean(F.softmax(stu_out / self.t, dim=-1), dim=1)
        tea_out = torch.mean(F.softmax(tea_out / self.t, dim=-1), dim=1)
        if self.use_log is True:
            log_out1 = torch.log(stu_out)
            log_out2 = torch.log(tea_out)
            loss = (self._kldiv(log_out1, tea_out) + self._kldiv(log_out2, stu_out)) / 2.0
        return loss

    def forward_sum(self, stu_out, tea_out):
        stu_out = torch.sum(F.softmax(stu_out / self.t, dim=-1), dim=1)
        tea_out = torch.sum(F.softmax(tea_out / self.t, dim=-1), dim=1)
        stu_out = torch.log(stu_out)
        bs = stu_out.shape[0]
        loss = tea_out * (torch.log(tea_out + self.eps) - stu_out)
        loss = torch.sum(loss, dim=1) / loss.shape[0]
        return loss

    def _kldiv(self, x, target):
        eps = 1e-10
        loss = target * (torch.log(target + eps) - x)
        loss = torch.sum(torch.mean(loss, dim=1)) / loss.shape[0]
        return loss

    def forward(self, stu_out, tea_out, targets=None):
        if self.mode == 'log':
            return self.forward_log(stu_out, tea_out)
        elif self.mode == 'mean':
            blank_mask = torch.ones_like(stu_out)
            blank_mask.requires_grade = False
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanmax(stu_out, tea_out)
        elif self.mode == 'sum':
            return self.forward_sum(stu_out, tea_out)
        elif self.mode == 'meanlog':
            blank_mask = torch.ones_like(stu_out)
            blank_mask.requires_grade = False
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanlog(stu_out, tea_out)
        elif self.mode == 'ctcdkd':
            blank_mask = torch.ones_like(stu_out)
            blank_mask.requires_grade = False
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.ctc_dkd_loss(stu_out, tea_out, targets)
        else:
            raise ValueError('error!!!!!!')

    def forward_log(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log is True:
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        return loss


class DistillCTCLogits(KLCTCLogits):

    def __init__(self, model_name_pairs=[], key=None, name='ctc_logits', reduction='mean'):
        super().__init__(reduction=reduction)
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.key = key
        self.name = name

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]['ctc']
                out2 = out2[self.key]['ctc']
            ctc_label = batch[1]
            loss = super().forward(out1, out2, ctc_label)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict['{}_{}_{}'.format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict['{}_{}'.format(self.name, idx)] = loss
        return loss_dict


def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_bs):
    """
    """
    pos_lists_, pos_masks_, label_lists_ = [], [], []
    img_bs = batch_size
    ngpu = int(batch_size / img_bs)
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    pos_lists_split, pos_masks_split, label_lists_split = [], [], []
    for i in range(ngpu):
        pos_lists_split.append([])
        pos_masks_split.append([])
        label_lists_split.append([])
    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs)
        img_id = img_id % img_bs
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(copy.deepcopy(label_lists[i]))
    for i in range(ngpu):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_bs:
            for j in range(0, tcl_bs - vp_len):
                pos_list = pos_lists_split[i][j].copy()
                pos_lists_split[i].append(pos_list)
                pos_mask = pos_masks_split[i][j].copy()
                pos_masks_split[i].append(pos_mask)
                label_list = copy.deepcopy(label_lists_split[i][j])
                label_lists_split[i].append(label_list)
        else:
            for j in range(0, vp_len - tcl_bs):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)
    for i in range(ngpu):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])
    return pos_lists_, pos_masks_, label_lists_


def pre_process(label_list, pos_list, pos_mask, max_text_length, max_text_nums, pad_num, tcl_bs):
    label_list = label_list.numpy()
    batch, _, _, _ = label_list.shape
    pos_list = pos_list.numpy()
    pos_mask = pos_mask.numpy()
    pos_list_t = []
    pos_mask_t = []
    label_list_t = []
    for i in range(batch):
        for j in range(max_text_nums):
            if pos_mask[i, j].any():
                pos_list_t.append(pos_list[i][j])
                pos_mask_t.append(pos_mask[i][j])
                label_list_t.append(label_list[i][j])
    pos_list, pos_mask, label_list = org_tcl_rois(batch, pos_list_t, pos_mask_t, label_list_t, tcl_bs)
    label = []
    tt = [l.tolist() for l in label_list]
    for i in range(tcl_bs):
        k = 0
        for j in range(max_text_length):
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        label.append(k)
    label = torch.tensor(label)
    label = label.long()
    pos_list = torch.tensor(pos_list)
    pos_mask = torch.tensor(pos_mask)
    label_list = torch.squeeze(torch.tensor(label_list), dim=2)
    label_list = label_list.int()
    return pos_list, pos_mask, label_list, label


class PGLoss(nn.Module):

    def __init__(self, tcl_bs, max_text_length, max_text_nums, pad_num, eps=1e-06, **kwargs):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = torch.split(l_border, split_size_or_sections=[4, 1], dim=1)
        f_border_split = f_border
        b, c, h, w = l_border_norm.shape
        l_border_norm_split = l_border_norm.expand([b, 4 * c, h, w])
        b, c, h, w = l_score.shape
        l_border_score = l_score.expand([b, 4 * c, h, w])
        b, c, h, w = l_mask.shape
        l_border_mask = l_mask.expand([b, 4 * c, h, w])
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.float()
        border_sign.requires_grade = False
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / (torch.sum(l_border_score * l_border_mask) + 1e-05)
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = torch.split(l_direction, split_size_or_sections=[2, 1], dim=1)
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = l_direction_norm.expand([b, 2 * c, h, w])
        b, c, h, w = l_score.shape
        l_direction_score = l_score.expand([b, 2 * c, h, w])
        b, c, h, w = l_mask.shape
        l_direction_mask = l_mask.expand([b, 2 * c, h, w])
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = direction_sign.float()
        direction_sign.requires_grade = False
        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = torch.sum(direction_out_loss * l_direction_score * l_direction_mask) / (torch.sum(l_direction_score * l_direction_mask) + 1e-05)
        return direction_loss

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = torch.permute(f_char, [0, 2, 3, 1])
        tcl_pos = torch.permute(tcl_pos, [-1, 3]).int()
        f_tcl_char = torch.gather(f_char, tcl_pos)
        f_tcl_char = torch.reshape(f_tcl_char, [-1, 64, self.pad_num + 1])
        f_tcl_char_fg, f_tcl_char_bg = torch.split(f_tcl_char, [self.pad_num, 1], dim=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        b, c, l = tcl_mask.shape
        tcl_mask_fg = tcl_mask.expand([b, c, self.pad_num * l])
        tcl_mask_fg.requires_grade = False
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * -20.0
        f_tcl_char_mask = torch.cat([f_tcl_char_fg, f_tcl_char_bg], dim=2)
        f_tcl_char_ld = torch.permute(f_tcl_char_mask, (1, 0, 2))
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.tensor([N] * B, dtype=torch.int64)
        cost = torch.nn.functional.ctc_loss(log_probs=f_tcl_char_ld, targets=tcl_label, input_lengths=input_lengths, target_lengths=label_t, blank=self.pad_num, reduction='none')
        cost = cost.mean()
        return cost

    def forward(self, predicts, labels):
        images, tcl_maps, tcl_label_maps, border_maps, direction_maps, training_masks, label_list, pos_list, pos_mask = labels
        pos_list, pos_mask, label_list, label_t = pre_process(label_list, pos_list, pos_mask, self.max_text_length, self.max_text_nums, self.pad_num, self.tcl_bs)
        f_score, f_border, f_direction, f_char = predicts['f_score'], predicts['f_border'], predicts['f_direction'], predicts['f_char']
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(f_direction, direction_maps, tcl_maps, training_masks)
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss
        losses = {'loss': loss_all, 'score_loss': score_loss, 'border_loss': border_loss, 'direction_loss': direction_loss, 'ctc_loss': ctc_loss}
        return losses


class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.T = T

    def knowledge_distillation_kl_div_loss(self, pred, soft_label, T, detach_target=True):
        """Loss function for knowledge distilling using KL divergence.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            T (int): Temperature for distillation.
            detach_target (bool): Remove soft_label from automatic differentiation
        """
        assert pred.shape == soft_label.shape
        target = F.softmax(soft_label / T, dim=1)
        if detach_target:
            target = target.detach()
        kd_loss = F.kl_div(F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (T * T)
        return kd_loss

    def forward(self, pred, soft_label, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_kd_out = self.knowledge_distillation_kl_div_loss(pred, soft_label, T=self.T)
        if weight is not None:
            loss_kd_out = weight * loss_kd_out
        if avg_factor is None:
            if reduction == 'none':
                loss = loss_kd_out
            elif reduction == 'mean':
                loss = loss_kd_out.mean()
            elif reduction == 'sum':
                loss = loss_kd_out.sum()
        elif reduction == 'mean':
            loss = loss_kd_out.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
        loss_kd = self.loss_weight * loss
        return loss_kd


class SDMGRLoss(nn.Module):

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=0):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def pre_process(self, gts, tag):
        gts, tag = gts.numpy(), tag.numpy().tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_gts.append(torch.tensor(gts[i, :num, :num + 1], dtype=torch.int64))
        return temp_gts

    def accuracy(self, pred, target, topk=1, thresh=None):
        """Calculate accuracy according to the prediction and target.

        Args:
            pred (torch.Tensor): The model prediction, shape (N, num_class)
            target (torch.Tensor): The target of each prediction, shape (N, )
            topk (int | tuple[int], optional): If the predictions in ``topk``
                matches the target, the predictions will be regarded as
                correct ones. Defaults to 1.
            thresh (float, optional): If not None, predictions with scores under
                this threshold are considered incorrect. Default to None.

        Returns:
            float | tuple[float]: If the input ``topk`` is a single integer,
                the function will return a single float as accuracy. If
                ``topk`` is a tuple containing multiple integers, the
                function will return a tuple containing accuracies of
                each ``topk`` number.
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = topk,
            return_single = True
        else:
            return_single = False
        maxk = max(topk)
        if pred.shape[0] == 0:
            accu = [pred.new_tensor(0.0) for i in range(len(topk))]
            return accu[0] if return_single else accu
        pred_value, pred_label = torch.topk(pred, maxk, dim=1)
        pred_label = pred_label.transpose([1, 0])
        correct = torch.equal(pred_label, target.reshape([1, -1]).expand_as(pred_label))
        res = []
        for k in topk:
            correct_k = torch.sum(correct[:k].reshape([-1]).astype('float32'), dim=0, keepdim=True)
            res.append(torch.multiply(correct_k, torch.tensor(100.0 / pred.shape[0])))
        return res[0] if return_single else res

    def forward(self, pred, batch):
        node_preds, edge_preds = pred
        gts, tag = batch[4], batch[5]
        gts = self.pre_process(gts, tag)
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        node_gts = torch.cat(node_gts)
        edge_gts = torch.cat(edge_gts)
        node_valids = torch.nonzero(node_gts != self.ignore).reshape([-1])
        edge_valids = torch.nonzero(edge_gts != -1).reshape([-1])
        loss_node = self.loss_node(node_preds, node_gts)
        loss_edge = self.loss_edge(edge_preds, edge_gts)
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge
        return dict(loss=loss, loss_node=loss_node, loss_edge=loss_edge, acc_node=self.accuracy(torch.gather(node_preds, node_valids), torch.gather(node_gts, node_valids)), acc_edge=self.accuracy(torch.gather(edge_preds, edge_valids), torch.gather(edge_gts, edge_valids)))


class CosineEmbeddingLoss(nn.Module):

    def __init__(self, margin=0.0):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-12

    def forward(self, x1, x2, target):
        similarity = torch.sum(x1 * x2, dim=-1) / (torch.norm(x1, dim=-1) * torch.norm(x2, dim=-1) + self.epsilon)
        one_list = torch.full_like(target, fill_value=1)
        out = torch.mean(torch.where(torch.equal(target, one_list), 1.0 - similarity, torch.maximum(torch.zeros_like(similarity), similarity - self.margin)))
        return out


class AsterLoss(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, sequence_normalize=False, sample_normalize=True, **kwargs):
        super(AsterLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        self.loss_sem = CosineEmbeddingLoss()
        self.is_cosin_loss = True
        self.loss_func_rec = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        targets = batch[1].astype('int64')
        label_lengths = batch[2].astype('int64')
        sem_target = batch[3].astype('float32')
        embedding_vectors = predicts['embedding_vectors']
        rec_pred = predicts['rec_pred']
        if not self.is_cosin_loss:
            sem_loss = torch.sum(self.loss_sem(embedding_vectors, sem_target))
        else:
            label_target = torch.ones([embedding_vectors.shape[0]])
            sem_loss = torch.sum(self.loss_sem(embedding_vectors, sem_target, label_target))
        batch_size, def_max_length = targets.shape[0], targets.shape[1]
        mask = torch.zeros([batch_size, def_max_length])
        for i in range(batch_size):
            mask[i, :label_lengths[i]] = 1
        mask = mask.float()
        max_length = max(label_lengths)
        assert max_length == rec_pred.shape[1]
        targets = targets[:, :max_length]
        mask = mask[:, :max_length]
        rec_pred = torch.reshape(rec_pred, [-1, rec_pred.shape[2]])
        input = nn.functional.log_softmax(rec_pred, dim=1)
        targets = torch.reshape(targets, [-1, 1])
        mask = torch.reshape(mask, [-1, 1])
        output = -torch.index_select(input, index=targets) * mask
        output = torch.sum(output)
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size
        loss = output + sem_loss * 0.1
        return {'loss': loss}


class AttentionLoss(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        predicts = predicts['res'][:, :-1]
        targets = batch[1].long()[:, 1:]
        assert len(targets.shape) == len(list(predicts.shape)) - 1, "The target's shape and inputs's shape is [N, d] and [N, num_steps]"
        inputs = torch.reshape(predicts, [-1, predicts.shape[-1]])
        targets = torch.reshape(targets, [-1])
        return {'loss': torch.sum(self.loss_func(inputs, targets))}


def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    counting_labels = np.zeros([b, channel])
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    counting_labels = torch.tensor(counting_labels, dtype=torch.float32)
    return counting_labels


class CANLoss(nn.Module):
    """
    CANLoss is consist of two part:
        word_average_loss: average accuracy of the symbol
        counting_loss: counting loss of every symbol
    """

    def __init__(self):
        super(CANLoss, self).__init__()
        self.use_label_mask = False
        self.out_channel = 111
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.ratio = 16

    def forward(self, preds, batch):
        word_probs = preds[0]
        counting_preds = preds[1]
        counting_preds1 = preds[2]
        counting_preds2 = preds[3]
        labels = batch[2]
        labels_mask = batch[3]
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) + self.counting_loss(counting_preds, counting_labels)
        word_loss = self.cross(torch.reshape(word_probs, [-1, word_probs.shape[-1]]), torch.reshape(labels, [-1]))
        word_average_loss = torch.sum(torch.reshape(word_loss * labels_mask, [-1])) / (torch.sum(labels_mask) + 1e-10) if self.use_label_mask else word_loss
        loss = word_average_loss + counting_loss
        return {'loss': loss}


class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        predicts = predicts['res']
        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1 - weight
            weight = torch.square(weight)
            loss = loss * weight
        loss = loss.mean()
        return {'loss': loss}


class EnhancedCTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, use_ace_loss=False, ace_loss_weight=0.1, use_center_loss=False, center_loss_weight=0.05, num_classes=6625, feat_dim=96, init_center=False, center_file_path=None, **kwargs):
        super(EnhancedCTCLoss, self).__init__()
        self.ctc_loss_func = CTCLoss(use_focal_loss=use_focal_loss)
        self.use_ace_loss = False
        if use_ace_loss:
            self.use_ace_loss = use_ace_loss
            self.ace_loss_func = ACELoss()
            self.ace_loss_weight = ace_loss_weight
        self.use_center_loss = False
        if use_center_loss:
            self.use_center_loss = use_center_loss
            self.center_loss_func = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, center_file_path=center_file_path)
            self.center_loss_weight = center_loss_weight

    def __call__(self, predicts, batch):
        loss = self.ctc_loss_func(predicts, batch)['loss']
        if self.use_center_loss:
            center_loss = self.center_loss_func(predicts, batch)['loss_center'] * self.center_loss_weight
            loss = loss + center_loss
        if self.use_ace_loss:
            ace_loss = self.ace_loss_func(predicts, batch)['loss_ace'] * self.ace_loss_weight
            loss = loss + ace_loss
        return {'enhanced_ctc_loss': loss}


class MultiLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func({'res': predicts['ctc']}, batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func({'res': predicts['sar']}, batch[:1] + batch[2:])['loss'] * self.weight_2
            elif name == 'NRTRLoss':
                loss = loss_func({'res': predicts['nrtr']}, batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                raise NotImplementedError('{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        return self.total_loss


class NRTRLoss(nn.Module):

    def __init__(self, smoothing=True, ignore_index=0, **kwargs):
        super(NRTRLoss, self).__init__()
        if ignore_index >= 0 and not smoothing:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        self.smoothing = smoothing

    def forward(self, pred, batch):
        pred = pred['res']
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]
        pred = pred.reshape([-1, pred.shape[2]])
        tgt = tgt.reshape([-1])
        if self.smoothing:
            eps = 0.1
            n_class = pred.shape[1]
            one_hot = F.one_hot(tgt.long(), num_classes=pred.shape[1])
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype, device=tgt.device))
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            loss = self.loss_func(pred, tgt)
        return {'loss': loss}


class PRENLoss(nn.Module):

    def __init__(self, **kwargs):
        super(PRENLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, predicts, batch):
        loss = self.loss_func(predicts, batch[1].astype('int64'))
        return {'loss': loss}


class RFLLoss(nn.Module):

    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()
        self.cnt_loss = nn.MSELoss(**kwargs)
        self.seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        if isinstance(predicts, tuple) or isinstance(predicts, list):
            cnt_outputs, seq_outputs = predicts
        else:
            cnt_outputs, seq_outputs = predicts, None
        if cnt_outputs is not None:
            cnt_loss = self.cnt_loss(cnt_outputs, batch[3].float())
            self.total_loss['cnt_loss'] = cnt_loss
            total_loss += cnt_loss
        if seq_outputs is not None:
            targets = batch[1].astype('int64')
            label_lengths = batch[2].astype('int64')
            batch_size, num_steps, num_classes = seq_outputs.shape[0], seq_outputs.shape[1], seq_outputs.shape[2]
            assert len(targets.shape) == len(list(seq_outputs.shape)) - 1, "The target's shape and inputs's shape is [N, d] and [N, num_steps]"
            inputs = seq_outputs[:, :-1, :]
            targets = targets[:, 1:]
            inputs = torch.reshape(inputs, [-1, inputs.shape[-1]])
            targets = torch.reshape(targets, [-1])
            seq_loss = self.seq_loss(inputs, targets)
            self.total_loss['seq_loss'] = seq_loss
            total_loss += seq_loss
        self.total_loss['loss'] = total_loss
        return self.total_loss


class SARLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predicts = predicts['res']
        predict = predicts[:, :-1, :]
        label = batch[1].long()[:, 1:]
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, "The target's shape and inputs's shape is [N, d] and [N, num_steps]"
        inputs = predict.reshape([-1, num_classes])
        targets = label.reshape([-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss}


class SATRNLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SATRNLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)
        self.loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predict = predicts[:, :-1, :]
        label = batch[1].astype('int64')[:, 1:]
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, "The target's shape and inputs's shape is [N, d] and [N, num_steps]"
        inputs = torch.reshape(predict, [-1, num_classes])
        targets = torch.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss.mean()}


class SPINAttentionLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100, **kwargs):
        super(SPINAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction=reduction, ignore_index=ignore_index)

    def forward(self, predicts, batch):
        targets = batch[1].astype('int64')
        targets = targets[:, 1:]
        label_lengths = batch[2].astype('int64')
        batch_size, num_steps, num_classes = predicts.shape[0], predicts.shape[1], predicts.shape[2]
        assert len(targets.shape) == len(list(predicts.shape)) - 1, "The target's shape and inputs's shape is [N, d] and [N, num_steps]"
        inputs = torch.reshape(predicts, [-1, predicts.shape[-1]])
        targets = torch.reshape(targets, [-1])
        return {'loss': self.loss_func(inputs, targets)}


class SRNLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SRNLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predicts, batch):
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        label = batch[1]
        casted_label = label.long()
        casted_label = label.reshape(x=casted_label, shape=[-1, 1])
        cost_word = self.loss_func(word_predict, label=casted_label)
        cost_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        cost_vsfd = self.loss_func(predict, label=casted_label)
        cost_word = label.reshape(x=label.sum(cost_word), shape=[1])
        cost_gsrm = label.reshape(x=label.sum(cost_gsrm), shape=[1])
        cost_vsfd = label.reshape(x=label.sum(cost_vsfd), shape=[1])
        sum_cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15
        return {'loss': sum_cost, 'word_loss': cost_word, 'img_loss': cost_vsfd}


class VLLoss(nn.Module):

    def __init__(self, mode='LF_1', weight_res=0.5, weight_mas=0.5, **kwargs):
        super(VLLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        assert mode in ['LF_1', 'LF_2', 'LA']
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    def flatten_label(self, target):
        label_flatten = []
        label_length = []
        for i in range(0, target.shape[0]):
            cur_label = target[i].tolist()
            label_flatten += cur_label[:cur_label.index(0) + 1]
            label_length.append(cur_label.index(0) + 1)
        label_flatten = torch.tensor(label_flatten, dtype=torch.int64)
        label_length = torch.tensor(label_length, dtype=torch.int32)
        return label_flatten, label_length

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def forward(self, predicts, batch):
        text_pre = predicts[0]
        target = batch[1].astype('int64')
        label_flatten, length = self.flatten_label(target)
        text_pre = self._flatten(text_pre, length)
        if self.mode == 'LF_1':
            loss = self.loss_func(text_pre, label_flatten)
        else:
            text_rem = predicts[1]
            text_mas = predicts[2]
            target_res = batch[2].astype('int64')
            target_sub = batch[3].astype('int64')
            label_flatten_res, length_res = self.flatten_label(target_res)
            label_flatten_sub, length_sub = self.flatten_label(target_sub)
            text_rem = self._flatten(text_rem, length_res)
            text_mas = self._flatten(text_mas, length_sub)
            loss_ori = self.loss_func(text_pre, label_flatten)
            loss_res = self.loss_func(text_rem, label_flatten_res)
            loss_mas = self.loss_func(text_mas, label_flatten_sub)
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        return {'loss': loss}


class StrokeFocusLoss(nn.Module):

    def __init__(self, character_dict_path=None, **kwargs):
        super(StrokeFocusLoss, self).__init__(character_dict_path)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_stroke_alphabet = '0123456789'
        self.english_stroke_dict = {}
        for index in range(len(self.english_stroke_alphabet)):
            self.english_stroke_dict[self.english_stroke_alphabet[index]] = index
        stroke_decompose_lines = open(character_dict_path, 'r').readlines()
        self.dic = {}
        for line in stroke_decompose_lines:
            line = line.strip()
            character, sequence = line.split()
            self.dic[character] = sequence

    def forward(self, pred, data):
        sr_img = pred['sr_img']
        hr_img = pred['hr_img']
        mse_loss = self.mse_loss(sr_img, hr_img)
        word_attention_map_gt = pred['word_attention_map_gt']
        word_attention_map_pred = pred['word_attention_map_pred']
        hr_pred = pred['hr_pred']
        sr_pred = pred['sr_pred']
        attention_loss = torch.nn.functional.l1_loss(word_attention_map_gt, word_attention_map_pred)
        loss = (mse_loss + attention_loss * 50) * 100
        return {'mse_loss': mse_loss, 'attention_loss': attention_loss, 'loss': loss}


class TableAttentionLoss(nn.Module):

    def __init__(self, structure_weight, loc_weight, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype('int64')
        structure_targets = structure_targets[:, 1:]
        structure_probs = torch.reshape(structure_probs, [-1, structure_probs.shape[-1]])
        structure_targets = torch.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)
        structure_loss = torch.mean(structure_loss) * self.structure_weight
        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype('float32')
        loc_targets_mask = batch[3].astype('float32')
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight
        total_loss = structure_loss + loc_loss
        return {'loss': total_loss, 'structure_loss': structure_loss, 'loc_loss': loc_loss}


class SLALoss(nn.Module):

    def __init__(self, structure_weight, loc_weight, loc_loss='mse', **kwargs):
        super(SLALoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.loc_loss = loc_loss
        self.eps = 1e-12

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype('int64')
        structure_targets = structure_targets[:, 1:]
        structure_loss = self.loss_func(structure_probs, structure_targets)
        structure_loss = torch.mean(structure_loss) * self.structure_weight
        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype('float32')
        loc_targets_mask = batch[3].astype('float32')
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.smooth_l1_loss(loc_preds * loc_targets_mask, loc_targets * loc_targets_mask, reduction='sum') * self.loc_weight
        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
        total_loss = structure_loss + loc_loss
        return {'loss': total_loss, 'structure_loss': structure_loss, 'loc_loss': loc_loss}


class TableMasterLoss(nn.Module):

    def __init__(self, ignore_index=-1):
        super(TableMasterLoss, self).__init__()
        self.structure_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        self.box_loss = nn.L1Loss(reduction='sum')
        self.eps = 1e-12

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1]
        structure_targets = structure_targets[:, 1:]
        structure_probs = structure_probs.reshape([-1, structure_probs.shape[-1]])
        structure_targets = structure_targets.reshape([-1])
        structure_loss = self.structure_loss(structure_probs, structure_targets)
        structure_loss = structure_loss.mean()
        losses = dict(structure_loss=structure_loss)
        bboxes_preds = predicts['loc_preds']
        bboxes_targets = batch[2][:, 1:, :]
        bbox_masks = batch[3][:, 1:]
        masked_bboxes_preds = bboxes_preds * bbox_masks
        masked_bboxes_targets = bboxes_targets * bbox_masks
        horizon_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 0::2], masked_bboxes_targets[:, :, 0::2])
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        vertical_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 1::2], masked_bboxes_targets[:, :, 1::2])
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)
        horizon_loss = horizon_loss.mean()
        vertical_loss = vertical_loss.mean()
        all_loss = structure_loss + horizon_loss + vertical_loss
        losses.update({'loss': all_loss, 'horizon_bbox_loss': horizon_loss, 'vertical_bbox_loss': vertical_loss})
        return losses


standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def load_confuse_matrix(confuse_dict_path):
    f = open(confuse_dict_path, 'rb')
    data = pkl.load(f)
    f.close()
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1, 62))
    pad = np.ones((63, 1))
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data == np.inf] = 1
    rearrange_data = torch.tensor(rearrange_data)
    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j + 26])
    rearrange_data = rearrange_data[:37, :37]
    return rearrange_data


def weight_cross_entropy(pred, gt, weight_table):
    batch = gt.shape[0]
    weight = weight_table[gt]
    pred_exp = torch.exp(pred)
    pred_exp_weight = weight * pred_exp
    loss = 0
    for i in range(len(gt)):
        loss -= torch.log(pred_exp_weight[i][gt[i]] / torch.sum(pred_exp_weight, 1)[i])
    return loss / batch


class TelescopeLoss(nn.Module):

    def __init__(self, confuse_dict_path):
        super(TelescopeLoss, self).__init__()
        self.weight_table = load_confuse_matrix(confuse_dict_path)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, data):
        sr_img = pred['sr_img']
        hr_img = pred['hr_img']
        sr_pred = pred['sr_pred']
        text_gt = pred['text_gt']
        word_attention_map_gt = pred['word_attention_map_gt']
        word_attention_map_pred = pred['word_attention_map_pred']
        mse_loss = self.mse_loss(sr_img, hr_img)
        attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
        recognition_loss = weight_cross_entropy(sr_pred, text_gt, self.weight_table)
        loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
        return {'mse_loss': mse_loss, 'attention_loss': attention_loss, 'loss': loss}


def build_backbone(config, model_type):
    if model_type == 'det' or model_type == 'table':
        support_dict = ['MobileNetV3', 'ResNet', 'ResNet_vd', 'ResNet_SAST', 'PPLCNet', 'PPLCNetV3', 'PPHGNet_small']
        if model_type == 'table':
            support_dict.append('TableResNetExtra')
    elif model_type == 'rec' or model_type == 'cls':
        support_dict = ['MobileNetV1Enhance', 'MobileNetV3', 'ResNet', 'ResNetFPN', 'MTB', 'ResNet31', 'ResNet45', 'ResNet_ASTER', 'MicroNet', 'EfficientNetb3_PREN', 'SVTRNet', 'ViTSTR', 'ResNet32', 'ResNetRFL', 'DenseNet', 'ShallowCNN', 'PPLCNetV3', 'PPHGNet_small']
    elif model_type == 'e2e':
        support_dict = ['ResNet']
    elif model_type == 'kie':
        support_dict = ['Kie_backbone', 'LayoutLMForSer', 'LayoutLMv2ForSer', 'LayoutLMv2ForRe', 'LayoutXLMForSer', 'LayoutXLMForRe']
    elif model_type == 'table':
        support_dict = ['ResNet', 'MobileNetV3']
    else:
        raise NotImplementedError
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('when model typs is {}, backbone only support {}'.format(model_type, support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def build_head(config):
    support_dict = ['DBHead', 'PSEHead', 'FCEHead', 'EASTHead', 'SASTHead', 'CTCHead', 'ClsHead', 'AttentionHead', 'SRNHead', 'PGHead', 'Transformer', 'TableAttentionHead', 'SARHead', 'AsterHead', 'SDMGRHead', 'PRENHead', 'MultiHead', 'ABINetHead', 'TableMasterHead', 'SPINAttentionHead', 'VLHead', 'SLAHead', 'RobustScannerHead', 'CT_Head', 'RFLHead', 'DRRGHead', 'CANHead', 'SATRNHead', 'PFHeadLocal']
    if config['name'] == 'DRRGHead':
        support_dict.append('DRRGHead')
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def build_neck(config):
    support_dict = ['FPN', 'FCEFPN', 'LKPAN', 'DBFPN', 'RSEFPN', 'EASTFPN', 'SASTFPN', 'SequenceEncoder', 'PGFPN', 'TableFPN', 'PRENFPN', 'CSPPAN', 'CTFPN', 'RFAdaptor', 'FPN_UNet']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def build_transform(config):
    support_dict = ['TPS', 'STN_ON', 'GA_SPIN', 'TSRN', 'TBSRN']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class BaseModel(nn.Module):

    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config['Backbone']['in_channels'] = in_channels
            self.backbone = build_backbone(config['Backbone'], model_type)
            in_channels = self.backbone.out_channels
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config['Head']['in_channels'] = in_channels
            self.head = build_head(config['Head'])
        self.return_all_feats = config.get('return_all_feats', False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y['backbone_out'] = x
        final_name = 'backbone_out'
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y['neck_out'] = x
            final_name = 'neck_out'
        if self.use_head:
            x = self.head(x, data=data)
            if 'ctc_neck' in x.keys():
                y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
            final_name = 'head_out'
        if self.return_all_feats:
            if self.training:
                return y
            else:
                return {final_name: x}
        else:
            return x


logger_initialized = {}


@functools.lru_cache()
def get_logger(name='torchocr', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    if log_file is not None and rank == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger


def load_pretrained_params(model, pretrained_model):
    logger = get_logger()
    loaded_state_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))['state_dict']
    current_model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k not in current_model_dict.keys():
            logger.info(f'ignore loading parameter: {k}, because it is not in current model')
            continue
        if current_model_dict[k].size() != v.size():
            logger.info(f'ignore loading parameter: {k}, because of size mismatch, current size: {current_model_dict[k].size()}, pretrained size: {v.size()}')
            continue
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)


class DistillationModel(nn.Module):

    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        logger = get_logger()
        self.model_list = nn.ModuleDict()
        for key in config['Models']:
            model_config = config['Models'][key]
            freeze_params = False
            pretrained = None
            if 'freeze_params' in model_config:
                freeze_params = model_config.pop('freeze_params')
            if 'pretrained' in model_config:
                pretrained = model_config.pop('pretrained')
            model = BaseModel(model_config)
            if pretrained is not None:
                load_pretrained_params(model, pretrained)
                logger.info('{}, load pretrained model from {}'.format(key, pretrained))
            if freeze_params:
                for param in model.parameters():
                    param.requires_grad = False
            self.model_list.add_module(key, model)

    def forward(self, x, data=None):
        result_dict = dict()
        for model_name in self.model_list:
            result_dict[model_name] = self.model_list[model_name](x, data)
        return result_dict


class GELU(nn.Module):

    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):

    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'hard_sigmoid':
            self.act = nn.Hardsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = nn.Hardswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Activation(act) if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def hardsigmoid(x):
    return F.relu6(x + 3.0, inplace=True) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        x = torch.mul(inputs, outputs)
        return x


class ResidualUnit(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act=None, name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se
        self.expand_conv = ConvBNLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, if_act=True, act=act)
        self.bottleneck_conv = ConvBNLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2), groups=mid_channels, if_act=True, act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + '_se')
        self.linear_conv = ConvBNLayer(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, if_act=False, act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Module):

    def __init__(self, in_channels=3, model_name='small', scale=0.5, large_stride=None, small_stride=None, **kwargs):
        super(MobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]
        assert isinstance(large_stride, list), 'large_stride type must be list but got {}'.format(type(large_stride))
        assert isinstance(small_stride, list), 'small_stride type must be list but got {}'.format(type(small_stride))
        assert len(large_stride) == 4, 'large_stride length must be 4 but got {}'.format(len(large_stride))
        assert len(small_stride) == 4, 'small_stride length must be 4 but got {}'.format(len(small_stride))
        if model_name == 'large':
            cfg = [[3, 16, 16, False, 'relu', large_stride[0]], [3, 64, 24, False, 'relu', (large_stride[1], 1)], [3, 72, 24, False, 'relu', 1], [5, 72, 40, True, 'relu', (large_stride[2], 1)], [5, 120, 40, True, 'relu', 1], [5, 120, 40, True, 'relu', 1], [3, 240, 80, False, 'hard_swish', 1], [3, 200, 80, False, 'hard_swish', 1], [3, 184, 80, False, 'hard_swish', 1], [3, 184, 80, False, 'hard_swish', 1], [3, 480, 112, True, 'hard_swish', 1], [3, 672, 112, True, 'hard_swish', 1], [5, 672, 160, True, 'hard_swish', (large_stride[3], 1)], [5, 960, 160, True, 'hard_swish', 1], [5, 960, 160, True, 'hard_swish', 1]]
            cls_ch_squeeze = 960
        elif model_name == 'small':
            cfg = [[3, 16, 16, True, 'relu', (small_stride[0], 1)], [3, 72, 24, False, 'relu', (small_stride[1], 1)], [3, 88, 24, False, 'relu', 1], [5, 96, 40, True, 'hard_swish', (small_stride[2], 1)], [5, 240, 40, True, 'hard_swish', 1], [5, 240, 40, True, 'hard_swish', 1], [5, 120, 48, True, 'hard_swish', 1], [5, 144, 48, True, 'hard_swish', 1], [5, 288, 96, True, 'hard_swish', (small_stride[3], 1)], [5, 576, 96, True, 'hard_swish', 1], [5, 576, 96, True, 'hard_swish', 1]]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError('mode[' + model_name + '_model] is not implemented!')
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, 'supported scales are {} but input scale is {}'.format(supported_scale, scale)
        inplanes = 16
        self.conv1 = ConvBNLayer(in_channels=in_channels, out_channels=make_divisible(inplanes * scale), kernel_size=3, stride=2, padding=1, groups=1, if_act=True, act='hard_swish')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            block_list.append(ResidualUnit(in_channels=inplanes, mid_channels=make_divisible(scale * exp), out_channels=make_divisible(scale * c), kernel_size=k, stride=s, use_se=se, act=nl, name='conv' + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)
        self.conv2 = ConvBNLayer(in_channels=inplanes, out_channels=make_divisible(scale * cls_ch_squeeze), kernel_size=1, stride=1, padding=0, groups=1, if_act=True, act='hard_swish')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class DeformableConvV2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias_attr=None, skip_quant=False):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2 * groups
        self.mask_channel = kernel_size ** 2 * groups
        if bias_attr:
            dcn_bias_attr = True
        else:
            dcn_bias_attr = False
        self.conv_dcn = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation, groups=groups // 2 if groups > 1 else 1, bias=dcn_bias_attr)
        self.conv_offset = nn.Conv2d(in_channels, groups * 3 * kernel_size ** 2, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True)
        if skip_quant:
            self.conv_offset.skip_quant = True

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = torch.split(offset_mask, split_size_or_sections=[self.offset_channel, self.mask_channel], dim=1)
        mask = torch.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super(BottleneckBlock, self).__init__()
        self.scale = 4
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act='relu')
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu')
        self.conv2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels * self.scale, kernel_size=1, act=None)
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels * self.scale, kernel_size=1, stride=stride, is_vd_mode=not if_first and stride[0] != 1)
        self.shortcut = shortcut
        self.out_channels = out_channels * self.scale

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv2
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, shortcut=True, if_first=False, name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.scale = 1
        self.conv0 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, act='relu')
        self.conv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None)
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, is_vd_mode=not if_first and stride[0] != 1)
        self.shortcut = shortcut
        self.out_channels = out_channels * self.scale

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv1
        y = F.relu(y)
        return y


class ResNet_vd(nn.Module):

    def __init__(self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs):
        super(ResNet_vd, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.dcn_stage = dcn_stage if dcn_stage is not None else [False, False, False, False]
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act='relu')
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu')
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    bottleneck_block = BottleneckBlock(in_channels=num_channels[block] if i == 0 else num_filters[block] * 4, out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0, is_dcn=is_dcn)
                    shortcut = True
                    block_list.append(bottleneck_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    basic_block = BasicBlock(in_channels=num_channels[block] if i == 0 else num_filters[block], out_channels=num_filters[block], stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut, if_first=block == i == 0)
                    shortcut = True
                    block_list.append(basic_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out


class ConvBNAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ESEModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x * identity


class HG_Block(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, layer_num, identity=False):
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        self.layers.append(ConvBNAct(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1))
        for _ in range(layer_num - 1):
            self.layers.append(ConvBNAct(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1))
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_conv = ConvBNAct(in_channels=total_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.att = ESEModule(out_channels)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_conv(x)
        x = self.att(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_num, layer_num, downsample=True, stride=[2, 1]):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, groups=in_channels, use_act=False)
        blocks_list = []
        blocks_list.append(HG_Block(in_channels, mid_channels, out_channels, layer_num, identity=False))
        for _ in range(block_num - 1):
            blocks_list.append(HG_Block(out_channels, mid_channels, out_channels, layer_num, identity=True))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNet(nn.Module):
    """
    PPHGNet
    Args:
        stem_channels: list. Stem channel list of PPHGNet.
        stage_config: dict. The configuration of each stage of PPHGNet. such as the number of channels, stride, etc.
        layer_num: int. Number of layers of HG_Block.
        use_last_conv: boolean. Whether to use a 1x1 convolutional layer before the classification layer.
        class_expand: int=2048. Number of channels for the last 1x1 convolutional layer.
        dropout_prob: float. Parameters of dropout, 0.0 means dropout is not used.
        class_num: int=1000. The number of classes.
    Returns:
        model: nn.Layer. Specific PPHGNet model depends on args.
    """

    def __init__(self, stem_channels, stage_config, layer_num, in_channels=3, det=False, out_indices=None):
        super().__init__()
        self.det = det
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]
        stem_channels.insert(0, in_channels)
        self.stem = nn.Sequential(*[ConvBNAct(in_channels=stem_channels[i], out_channels=stem_channels[i + 1], kernel_size=3, stride=2 if i == 0 else 1) for i in range(len(stem_channels) - 1)])
        if self.det:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.out_channels = []
        for block_id, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, stride = stage_config[k]
            self.stages.append(HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, stride))
            if block_id in self.out_indices:
                self.out_channels.append(out_channels)
        if not self.det:
            self.out_channels = stage_config['stage4'][2]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        if self.det:
            x = self.pool(x)
        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.det and i in self.out_indices:
                out.append(x)
        if self.det:
            return out
        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class LearnableAffineBlock(nn.Module):

    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([scale_value]))
        self.bias = nn.Parameter(torch.Tensor([bias_value]))

    def forward(self, x):
        return self.scale * x + self.bias


class Act(nn.Module):

    def __init__(self, act='hard_swish', lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        assert act in ['hard_swish', 'relu']
        self.act = Activation(act)
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))


class LearnableRepLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, num_conv_branches=1, lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2
        self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.conv_kxk = nn.ModuleList([ConvBNLayer(in_channels, out_channels, kernel_size, stride, groups=groups, lr_mult=lr_mult) for _ in range(self.num_conv_branches)])
        self.conv_1x1 = ConvBNLayer(in_channels, out_channels, 1, stride, groups=groups, lr_mult=lr_mult) if kernel_size > 1 else None
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out
        out = 0
        if self.identity is not None:
            out += self.identity(x)
        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)
        for conv in self.conv_kxk:
            out += conv(x)
        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

    def rep(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.groups)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        self.is_repped = True

    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        if not isinstance(kernel1x1, torch.Tensor):
            return 0
        else:
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self):
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1, self.kernel_size // 2)
        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)
        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias
        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(self, branch):
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=branch.weight.dtype)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, padding=0)
        self.hardsigmoid = Activation('hard_sigmoid')

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = x * identity
        return x


class LCNetV3Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, conv_kxk_num=4, lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=dw_size, stride=stride, groups=in_channels, num_conv_branches=conv_kxk_num, lr_mult=lr_mult, lab_lr=lab_lr)
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, num_conv_branches=conv_kxk_num, lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


NET_CONFIG_det = {'blocks2': [[3, 16, 32, 1, False]], 'blocks3': [[3, 32, 64, 2, False], [3, 64, 64, 1, False]], 'blocks4': [[3, 64, 128, 2, False], [3, 128, 128, 1, False]], 'blocks5': [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]], 'blocks6': [[5, 256, 512, 2, True], [5, 512, 512, 1, True], [5, 512, 512, 1, False], [5, 512, 512, 1, False]]}


NET_CONFIG_rec = {'blocks2': [[3, 16, 32, 1, False]], 'blocks3': [[3, 32, 64, 1, False], [3, 64, 64, 1, False]], 'blocks4': [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]], 'blocks5': [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]], 'blocks6': [[5, 256, 512, (2, 1), True], [5, 512, 512, 1, True], [5, 512, 512, (2, 1), False], [5, 512, 512, 1, False]]}


class PPLCNetV3(nn.Module):

    def __init__(self, scale=1.0, conv_kxk_num=4, lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], lab_lr=0.1, det=False, **kwargs):
        super().__init__()
        self.scale = scale
        self.lr_mult_list = lr_mult_list
        self.det = det
        self.net_config = NET_CONFIG_det if self.det else NET_CONFIG_rec
        assert isinstance(self.lr_mult_list, (list, tuple)), 'lr_mult_list should be in (list, tuple) but got {}'.format(type(self.lr_mult_list))
        assert len(self.lr_mult_list) == 6, 'lr_mult_list length should be 6 but got {}'.format(len(self.lr_mult_list))
        self.conv1 = ConvBNLayer(in_channels=3, out_channels=make_divisible(16 * scale), kernel_size=3, stride=2, lr_mult=self.lr_mult_list[0])
        self.blocks2 = nn.Sequential(*[LCNetV3Block(in_channels=make_divisible(in_c * scale), out_channels=make_divisible(out_c * scale), dw_size=k, stride=s, use_se=se, conv_kxk_num=conv_kxk_num, lr_mult=self.lr_mult_list[1], lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(self.net_config['blocks2'])])
        self.blocks3 = nn.Sequential(*[LCNetV3Block(in_channels=make_divisible(in_c * scale), out_channels=make_divisible(out_c * scale), dw_size=k, stride=s, use_se=se, conv_kxk_num=conv_kxk_num, lr_mult=self.lr_mult_list[2], lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(self.net_config['blocks3'])])
        self.blocks4 = nn.Sequential(*[LCNetV3Block(in_channels=make_divisible(in_c * scale), out_channels=make_divisible(out_c * scale), dw_size=k, stride=s, use_se=se, conv_kxk_num=conv_kxk_num, lr_mult=self.lr_mult_list[3], lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(self.net_config['blocks4'])])
        self.blocks5 = nn.Sequential(*[LCNetV3Block(in_channels=make_divisible(in_c * scale), out_channels=make_divisible(out_c * scale), dw_size=k, stride=s, use_se=se, conv_kxk_num=conv_kxk_num, lr_mult=self.lr_mult_list[4], lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(self.net_config['blocks5'])])
        self.blocks6 = nn.Sequential(*[LCNetV3Block(in_channels=make_divisible(in_c * scale), out_channels=make_divisible(out_c * scale), dw_size=k, stride=s, use_se=se, conv_kxk_num=conv_kxk_num, lr_mult=self.lr_mult_list[5], lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(self.net_config['blocks6'])])
        self.out_channels = make_divisible(512 * scale)
        if self.det:
            mv_c = [16, 24, 56, 480]
            self.out_channels = [make_divisible(self.net_config['blocks3'][-1][2] * scale), make_divisible(self.net_config['blocks4'][-1][2] * scale), make_divisible(self.net_config['blocks5'][-1][2] * scale), make_divisible(self.net_config['blocks6'][-1][2] * scale)]
            self.layer_list = nn.ModuleList([nn.Conv2d(self.out_channels[0], int(mv_c[0] * scale), 1, 1, 0), nn.Conv2d(self.out_channels[1], int(mv_c[1] * scale), 1, 1, 0), nn.Conv2d(self.out_channels[2], int(mv_c[2] * scale), 1, 1, 0), nn.Conv2d(self.out_channels[3], int(mv_c[3] * scale), 1, 1, 0)])
            self.out_channels = [int(mv_c[0] * scale), int(mv_c[1] * scale), int(mv_c[2] * scale), int(mv_c[3] * scale)]

    def forward(self, x):
        out_list = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)
        if self.det:
            out_list[0] = self.layer_list[0](out_list[0])
            out_list[1] = self.layer_list[1](out_list[1])
            out_list[2] = self.layer_list[2](out_list[2])
            out_list[3] = self.layer_list[3](out_list[3])
            return out_list
        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class DepthwiseSeparable(nn.Module):

    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3, padding=1, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels, num_filters=int(num_filters1 * scale), filter_size=dw_size, stride=stride, padding=padding, num_groups=int(num_groups * scale))
        self._se = None
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(num_channels=int(num_filters1 * scale), filter_size=1, num_filters=int(num_filters2 * scale), stride=1, padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self._se is not None:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Module):

    def __init__(self, in_channels=3, scale=0.5, last_conv_stride=1, last_pool_type='max', **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []
        self.conv1 = ConvBNLayer(num_channels=in_channels, filter_size=3, num_filters=int(32 * scale), stride=2, padding=1)
        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale), num_filters1=32, num_filters2=64, num_groups=32, stride=1, scale=scale)
        self.block_list.append(conv2_1)
        conv2_2 = DepthwiseSeparable(num_channels=int(64 * scale), num_filters1=64, num_filters2=128, num_groups=64, stride=1, scale=scale)
        self.block_list.append(conv2_2)
        conv3_1 = DepthwiseSeparable(num_channels=int(128 * scale), num_filters1=128, num_filters2=128, num_groups=128, stride=1, scale=scale)
        self.block_list.append(conv3_1)
        conv3_2 = DepthwiseSeparable(num_channels=int(128 * scale), num_filters1=128, num_filters2=256, num_groups=128, stride=(2, 1), scale=scale)
        self.block_list.append(conv3_2)
        conv4_1 = DepthwiseSeparable(num_channels=int(256 * scale), num_filters1=256, num_filters2=256, num_groups=256, stride=1, scale=scale)
        self.block_list.append(conv4_1)
        conv4_2 = DepthwiseSeparable(num_channels=int(256 * scale), num_filters1=256, num_filters2=512, num_groups=256, stride=(2, 1), scale=scale)
        self.block_list.append(conv4_2)
        for _ in range(5):
            conv5 = DepthwiseSeparable(num_channels=int(512 * scale), num_filters1=512, num_filters2=512, num_groups=512, stride=1, dw_size=5, padding=2, scale=scale, use_se=False)
            self.block_list.append(conv5)
        conv5_6 = DepthwiseSeparable(num_channels=int(512 * scale), num_filters1=512, num_filters2=1024, num_groups=512, stride=(2, 1), dw_size=5, padding=2, scale=scale, use_se=True)
        self.block_list.append(conv5_6)
        conv6 = DepthwiseSeparable(num_channels=int(1024 * scale), num_filters1=1024, num_filters2=1024, num_groups=1024, stride=last_conv_stride, dw_size=5, padding=2, use_se=True, scale=scale)
        self.block_list.append(conv6)
        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


class MTB(nn.Module):

    def __init__(self, cnn_num, in_channels):
        super(MTB, self).__init__()
        self.block = nn.Sequential()
        self.out_channels = in_channels
        self.cnn_num = cnn_num
        if self.cnn_num == 2:
            for i in range(self.cnn_num):
                self.block.add_module('conv_{}'.format(i), nn.Conv2d(in_channels=in_channels if i == 0 else 32 * 2 ** (i - 1), out_channels=32 * 2 ** i, kernel_size=3, stride=2, padding=1))
                self.block.add_module('relu_{}'.format(i), nn.ReLU())
                self.block.add_module('bn_{}'.format(i), nn.BatchNorm2d(32 * 2 ** i))

    def forward(self, images):
        x = self.block(images)
        if self.cnn_num == 2:
            x = x.permute(0, 3, 2, 1)
            x_shape = x.shape
            x = torch.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3]))
        return x


class ResNet31(nn.Module):
    """
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self, in_channels=3, layers=[1, 2, 5, 3], channels=[64, 128, 256, 256, 512, 512, 512], out_indices=None, last_stage_pool=False):
        super(ResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)
        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool
        self.conv1_1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=True)
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(channels[4], channels[4], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU(inplace=True)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU(inplace=True)
        self.out_channels = channels[-1]

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(output_channels))
            layers.append(BasicBlock(input_channels, output_channels, downsample=downsample))
            input_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        outs = []
        for i in range(4):
            layer_index = i + 2
            pool_layer = getattr(self, 'pool{}'.format(layer_index))
            block_layer = getattr(self, 'block{}'.format(layer_index))
            conv_layer = getattr(self, 'conv{}'.format(layer_index))
            bn_layer = getattr(self, 'bn{}'.format(layer_index))
            relu_layer = getattr(self, 'relu{}'.format(layer_index))
            if pool_layer is not None:
                x = pool_layer(x)
            x = block_layer(x)
            x = conv_layer(x)
            x = bn_layer(x)
            x = relu_layer(x)
            outs.append(x)
        if self.out_indices is not None:
            return tuple([outs[i] for i in self.out_indices])
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        if layers >= 50:
            block_class = BottleneckBlock
        else:
            block_class = BasicBlock
        num_filters = [64, 128, 256, 512]
        self.conv1_1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, act='relu')
        self.conv1_2 = ConvBNLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu')
        self.conv1_3 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_list = nn.Sequential()
        in_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if layers in [101, 152, 200] and block == 2:
                    if i == 0:
                        conv_name = 'res' + str(block + 2) + 'a'
                    else:
                        conv_name = 'res' + str(block + 2) + 'b' + str(i)
                else:
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                if i == 0 and block != 0:
                    stride = 2, 1
                else:
                    stride = 1, 1
                block_instance = block_class(in_channels=in_channels, out_channels=num_filters[block], stride=stride, shortcut=shortcut, if_first=block == i == 0, name=conv_name)
                shortcut = True
                in_channels = block_instance.out_channels
                self.block_list.add_module('bb_%d_%d' % (block, i), block_instance)
            self.out_channels = num_filters[block]
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.out_pool(y)
        return y


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = torch.as_tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Activation(act_type=act_layer, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):

    def __init__(self, dim, num_heads=8, HW=[8, 25], local_k=[3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0] // 2, local_k[1] // 2], groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, mixer='Global', HW=[8, 25], local_k=[7, 11], qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        self.mask = None
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask1 = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], fill_value=float('-Inf'), dtype=torch.float32)
            mask = torch.where(mask1 < 1, mask1, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(1)
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mask is not None:
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn.matmul(v).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mixer='Global', local_mixer=[7, 11], HW=None, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer='gelu', norm_layer='nn.LayerNorm', epsilon=1e-06, prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(dim, num_heads=num_heads, mixer=mixer, HW=HW, local_k=local_mixer, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=768, sub_num=2, patch_size=[4, 4], mode='pope'):
        super().__init__()
        num_patches = img_size[1] // 2 ** sub_num * (img_size[0] // 2 ** sub_num)
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(ConvBNLayer(in_channels=in_channels, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True))
            if sub_num == 3:
                self.proj = nn.Sequential(ConvBNLayer(in_channels=in_channels, out_channels=embed_dim // 4, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 4, out_channels=embed_dim // 2, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True), ConvBNLayer(in_channels=embed_dim // 2, out_channels=embed_dim, kernel_size=3, stride=2, padding=1, act='gelu', bias_attr=True))
        elif mode == 'linear':
            self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], "Input image size ({}*{}) doesn't match model ({}*{}).".format(H, W, self.img_size[0], self.img_size[1])
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x


class SubSample(nn.Module):

    def __init__(self, in_channels, out_channels, types='Pool', stride=[2, 1], sub_norm='nn.LayerNorm', act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class SVTRNet(nn.Module):

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=[64, 128, 256], depth=[3, 6, 3], num_heads=[2, 4, 8], mixer=['Local'] * 6 + ['Global'] * 6, local_mixer=[[7, 11], [7, 11], [7, 11]], patch_merging='Conv', mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, last_drop=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer='nn.LayerNorm', sub_norm='nn.LayerNorm', epsilon=1e-06, out_channels=192, out_char_num=25, block_unit='Block', act='gelu', last_stage=True, sub_num=2, prenorm=True, use_lenhead=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim[0], sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // 2 ** sub_num, img_size[1] // 2 ** sub_num]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([Block_unit(dim=embed_dim[0], num_heads=num_heads[0], mixer=mixer[0:depth[0]][i], HW=self.HW, local_mixer=local_mixer[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[0:depth[0]][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[0])])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1], types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([Block_unit(dim=embed_dim[1], num_heads=num_heads[1], mixer=mixer[depth[0]:depth[0] + depth[1]][i], HW=HW, local_mixer=local_mixer[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[depth[0]:depth[0] + depth[1]][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[1])])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1], types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList([Block_unit(dim=embed_dim[2], num_heads=num_heads[2], mixer=mixer[depth[0] + depth[1]:][i], HW=HW, local_mixer=local_mixer[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=act, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1]:][i], norm_layer=norm_layer, epsilon=epsilon, prenorm=prenorm) for i in range(depth[2])])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(in_channels=embed_dim[2], out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.hardswish = Activation('hard_swish', inplace=True)
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = Activation('hard_swish', inplace=True)
            self.dropout_len = nn.Dropout(p=last_drop)
        torch.nn.init.xavier_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.permute(0, 2, 1).reshape([-1, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.permute(0, 2, 1).reshape([-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(x.permute(0, 2, 1).reshape([-1, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x


class ClsHead(nn.Module):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, class_dim)

    def forward(self, x, data=None):
        x = self.pool(x)
        x = torch.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return {'res': x}


class Head(nn.Module):

    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=kernel_list[0], padding=int(kernel_list[0] // 2), bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=kernel_list[1], stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=kernel_list[2], stride=2)

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = F.relu(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x, data=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'res': shrink_maps}
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'res': y}


class LocalModule(nn.Module):

    def __init__(self, in_c, mid_c):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act='relu')
        self.last_1 = nn.Conv2d(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = torch.cat([init_map, x], dim=1)
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):

    def __init__(self, in_channels, k=50, mode='small', **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode
        self.up_conv = nn.Upsample(scale_factor=2, mode='nearest')
        if self.mode == 'large':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == 'small':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, data=None):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = F.sigmoid(cbn_maps)
        if not self.training:
            return {'res': 0.5 * (base_maps + cbn_maps), 'cbn_maps': cbn_maps}
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.concat([cbn_maps, threshold_maps, binary_maps], dim=1)
        return {'res': y, 'distance_maps': cbn_maps, 'cbn_maps': binary_maps}


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        	ext{MultiHead}(Q, K, V) = 	ext{Concat}(head_1,\\dots,head_h)W^O
        	ext{where} head_i = 	ext{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        qN = query.shape[1]
        if self.self_attn:
            qkv = self.qkv(query)
            qkv = qkv.reshape((qkv.shape[0], qN, 3, self.num_heads, self.head_dim)).permute((2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.shape[1]
            q = self.q(query)
            q = q.reshape([q.shape[0], qN, self.num_heads, self.head_dim]).permute([0, 2, 1, 3])
            kv = self.kv(key)
            kv = kv.reshape((kv.shape[0], kN, 2, self.num_heads, self.head_dim)).permute((2, 0, 3, 1, 4))
            k, v = kv[0], kv[1]
        attn = q.matmul(k.permute((0, 1, 3, 2))) * self.scale
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn.matmul(v).permute((0, 2, 1, 3))
        x = x.reshape((x.shape[0], qN, self.embed_dim))
        x = self.out_proj(x)
        return x


class AttentionGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)
        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = torch.permute(alpha, [0, 2, 1])
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden[0])
        return (cur_hidden, cur_hidden), alpha


class AttentionHead(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = AttentionGRUCell(in_channels, hidden_size, out_channels)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.long(), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, data=None, batch_max_length=25):
        batch_size = inputs.size(0)
        num_steps = batch_max_length
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device), torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        output_hiddens = []
        if self.training:
            targets = data[0]
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(torch.unsqueeze(hidden[0], axis=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros(batch_size, dtype=torch.int32, device=inputs.device)
            probs = torch.zeros(batch_size, num_steps, self.num_classes, device=inputs.device)
            char_onehots = None
            outputs = None
            alpha = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        if not self.training:
            probs = F.softmax(probs, dim=2)
        return {'res': probs}


class AttentionLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[1]), dim=1)
        res = torch.tanh(batch_H_proj + prev_hidden_proj)
        e = self.score(res)
        alpha = F.softmax(e, dim=1)
        alpha = torch.permute(alpha, [0, 2, 1])
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.attention_cell = AttentionLSTMCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, data=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length
        hidden = torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if self.training:
            targets = data[0]
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                hidden = hidden[1][0], hidden[1][1]
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros(batch_size, device=inputs.device, dtype=torch.int32)
            probs = None
            char_onehots = None
            alpha = None
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = hidden[1][0], hidden[1][1]
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat([probs, torch.unsqueeze(probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        if not self.training:
            probs = F.softmax(probs, dim=2)
        return probs


class CTCHead(nn.Module):

    def __init__(self, in_channels, out_channels=6625, mid_channels=None, return_feats=False, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, data=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        result = {'res': predicts}
        if self.return_feats:
            result['feat'] = x
        if not self.training:
            result['res'] = F.softmax(result['res'], dim=2)
        return result


class FCTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        if self.only_transpose:
            return x.permute([0, 2, 1])
        else:
            return self.fc(x.permute([0, 2, 1]))


class Im2Seq(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)
        return x


class EncoderWithFC(nn.Module):

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(in_channels, hidden_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithRNN(nn.Module):

    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithSVTR(nn.Module):

    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120, use_guide=False, num_heads=8, qkv_bias=True, mlp_ratio=2.0, drop_rate=0.1, attn_drop_rate=0.1, drop_path=0.0, kernel_size=[3, 3], qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, kernel_size=kernel_size, padding=[kernel_size[0] // 2, kernel_size[1] // 2], act='swish')
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act='swish')
        self.svtr_block = nn.ModuleList([Block(dim=hidden_dims, num_heads=num_heads, mixer='Global', HW=None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer='swish', attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer='nn.LayerNorm', epsilon=1e-05, prenorm=False) for i in range(depth)])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-06)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act='swish')
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, kernel_size=kernel_size, padding=[kernel_size[0] // 2, kernel_size[1] // 2], act='swish')
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act='swish')
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        B, C, H, W = z.shape
        z = z.flatten(2).permute([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape([-1, H, W, C]).permute([0, 3, 1, 2])
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Module):

    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {'reshape': Im2Seq, 'fc': EncoderWithFC, 'rnn': EncoderWithRNN, 'svtr': EncoderWithSVTR}
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(encoder_type, support_encoder_dict.keys())
            if encoder_type == 'svtr':
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x


class Beam:
    """ Beam search """

    def __init__(self, size, device=False):
        self.size = size
        self._done = False
        self.scores = torch.zeros((size,), dtype=torch.float32)
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.full((size,), 0, dtype=torch.int64)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """Update beam status and check if finished or not."""
        num_words = word_prob.shape[1]
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.reshape([-1])
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        if self.next_ys[-1][0] == 3:
            self._done = True
            self.all_scores.append(self.scores)
        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return self.scores, torch.tensor([i for i in range(int(self.scores.shape[0]))], dtype=torch.int32)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [([2] + h) for h in hyps]
            dec_seq = torch.tensor(hyps, dtype=torch.int64)
        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = np.random.normal(0.0, d_model ** -0.5, (vocab, d_model)).astype(np.float32)
        self.embedding.weight.data = torch.from_numpy(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        	ext{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        	ext{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        	ext{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        pe = torch.permute(pe, [1, 0, 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x.permute([1, 0, 2])
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).permute([1, 0, 2])


class TransformerBlock(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, attention_dropout_rate=0.0, residual_dropout_rate=0.1, with_self_attn=True, with_cross_attn=False, epsilon=1e-05):
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=with_self_attn)
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.mlp = Mlp(in_features=d_model, hidden_features=dim_feedforward, act_layer='relu', drop=residual_dropout_rate)
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))
        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class Transformer(nn.Module):
    """A transformer model. User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, beam_size=0, num_decoder_layers=6, max_len=25, dim_feedforward=1024, attention_dropout_rate=0.0, residual_dropout_rate=0.1, in_channels=0, out_channels=0, scale_embedding=True):
        super(Transformer, self).__init__()
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = Embeddings(d_model=d_model, vocab=self.out_channels, padding_idx=0, scale_embedding=scale_embedding)
        self.positional_encoding = PositionalEncoding(dropout=residual_dropout_rate, dim=d_model)
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate, with_self_attn=True, with_cross_attn=False) for i in range(num_encoder_layers)])
        else:
            self.encoder = None
        self.decoder = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward, attention_dropout_rate, residual_dropout_rate, with_self_attn=True, with_cross_attn=True) for i in range(num_decoder_layers)])
        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias=False)
        w0 = np.random.normal(0.0, d_model ** -0.5, (d_model, self.out_channels)).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src
        else:
            memory = src
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, data=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """
        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            res = self.forward_train(src, tgt)
        elif self.beam_size > 0:
            res = self.forward_beam(src)
        else:
            res = self.forward_test(src)
        return {'res': res}

    def forward_test(self, src):
        bs = src.size(0)
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src
        else:
            memory = src
        dec_seq = torch.full((bs, 1), 2, dtype=torch.int64, device=src.device)
        dec_prob = torch.full((bs, 1), 1.0, dtype=torch.float32, device=src.device)
        for len_dec_seq in range(1, self.max_len):
            dec_seq_embed = self.embedding(dec_seq)
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.size(1), dec_seq_embed.device)
            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            preds_prob, preds_idx = torch.max(word_prob, dim=-1)
            if torch.equal(preds_idx, torch.full_like(preds_idx, 3, dtype=torch.int64)):
                break
            dec_seq = torch.cat([dec_seq, preds_idx.reshape([-1, 1])], dim=1)
            dec_prob = torch.concat([dec_prob, preds_prob.reshape([-1, 1])], dim=1)
        return [dec_seq, dec_prob]

    def forward_beam(self, images):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """ Collect tensor parts associated to active instances. """
            beamed_tensor_shape = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = n_curr_active_inst * n_bm, beamed_tensor_shape[1], beamed_tensor_shape[2]
            beamed_tensor = beamed_tensor.reshape([n_prev_active_inst, -1])
            beamed_tensor = beamed_tensor.index_select(curr_active_inst_idx, axis=0)
            beamed_tensor = beamed_tensor.reshape(new_shape)
            return beamed_tensor

        def collate_active_info(src_enc, inst_idx_to_position_map, active_inst_idx_list):
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.tensor(active_inst_idx, dtype=torch.int64)
            active_src_enc = collect_active_part(src_enc.permute([1, 0, 2]), active_inst_idx, n_prev_active_inst, n_bm).permute([1, 0, 2])
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
            """ Decode and update beam status, and then return active beam idx """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.reshape([-1, len_dec_seq])
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
                dec_seq = self.embedding(dec_seq)
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.size(1), dec_seq.device)
                tgt = dec_seq
                for decoder_layer in self.decoder:
                    tgt = decoder_layer(tgt, enc_output, self_mask=tgt_mask)
                dec_output = tgt
                dec_output = dec_output[:, -1, :]
                word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.reshape([n_active_inst, n_bm, -1])
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list
            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores
        with torch.no_grad():
            if self.encoder is not None:
                src = self.positional_encoding(images)
                src_enc = self.encoder(src)
            else:
                src_enc = images
            n_bm = self.beam_size
            inst_dec_beams = [Beam(n_bm) for _ in range(1)]
            active_inst_idx_list = list(range(1))
            src_enc = torch.tile(src_enc, [1, n_bm, 1])
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            for len_dec_seq in range(1, self.max_len):
                src_enc_copy = src_enc.clone()
                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_enc_copy, inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break
                src_enc, inst_idx_to_position_map = collate_active_info(src_enc_copy, inst_idx_to_position_map, active_inst_idx_list)
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_hyp = []
        hyp_scores = []
        for bs_hyp, score in zip(batch_hyp, batch_scores):
            l = len(bs_hyp[0])
            bs_hyp_pad = bs_hyp[0] + [3] * (25 - l)
            result_hyp.append(bs_hyp_pad)
            score = float(score) / l
            hyp_score = [score for _ in range(25)]
            hyp_scores.append(hyp_score)
        return [torch.from_numpy(np.array(result_hyp), dtype=torch.int64), torch.from_numpy(hyp_scores)]

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(torch.full(size=(sz, sz), dtype=torch.float32, fill_value=-torch.inf), diagonal=1)
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0)


class MultiHead(nn.Module):

    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')
        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                sar_args = self.head_list[idx][name]
                self.sar_head = eval(name)(in_channels=in_channels, out_channels=out_channels_list['SARLabelDecode'], **sar_args)
            elif name == 'NRTRHead':
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get('max_text_length', 25)
                nrtr_dim = gtc_args.get('nrtr_dim', 256)
                num_decoder_layers = gtc_args.get('num_decoder_layers', 4)
                self.before_gtc = nn.Sequential(nn.Flatten(2), FCTranspose(in_channels, nrtr_dim))
                self.gtc_head = Transformer(d_model=nrtr_dim, nhead=nrtr_dim // 32, num_encoder_layers=-1, beam_size=-1, num_decoder_layers=num_decoder_layers, max_len=max_text_length, dim_feedforward=nrtr_dim * 4, out_channels=out_channels_list['NRTRLabelDecode'])
            elif name == 'CTCHead':
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx][name]['Neck']
                encoder_type = neck_args.pop('name')
                self.ctc_encoder = SequenceEncoder(in_channels=in_channels, encoder_type=encoder_type, **neck_args)
                head_args = self.head_list[idx][name].get('Head', {})
                if head_args is None:
                    head_args = {}
                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels, out_channels=out_channels_list['CTCLabelDecode'], **head_args)
            else:
                raise NotImplementedError('{} is not supported in MultiHead yet'.format(name))

    def forward(self, x, data=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)['res']
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['res'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        if not self.training:
            return {'res': ctc_out}
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, data[1:])['res']
            head_out['sar'] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), data[1:])['res']
            head_out['nrtr'] = gtc_out
        return head_out


class PositionalEncoding_2d(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        	ext{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        	ext{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        	ext{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(torch.unsqueeze(pe, 0), [1, 0, 2])
        self.register_buffer('pe', pe)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = torch.permute(w_pe, [1, 2, 0])
        w_pe = torch.unsqueeze(w_pe, 2)
        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = torch.permute(h_pe, [1, 2, 0])
        h_pe = torch.unsqueeze(h_pe, 3)
        x = x + w_pe + h_pe
        x = torch.permute(torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]), [2, 0, 1])
        return self.dropout(x)


class SAREncoder(nn.Module):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self, enc_bi_rnn=False, enc_drop_rnn=0.0, enc_gru=False, d_model=512, d_enc=512, mask=True, **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)
        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask
        kwargs = dict(input_size=d_model, hidden_size=d_enc, num_layers=2, batch_first=True, dropout=enc_drop_rnn, bidirectional=enc_bi_rnn)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.size(0)
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        h_feat = feat.size(2)
        feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)
        feat_v = feat_v.permute(0, 2, 1).contiguous()
        holistic_feat = self.rnn_encoder(feat_v)[0]
        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i in range(valid_ratios.size(0)):
                valid_step = torch.minimum(torch.tensor(T), torch.ceil(T * valid_ratios[i]).int()) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]
        holistic_feat = self.linear(valid_hf)
        return holistic_feat


class BaseDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self, feat, out_enc, label=None, img_metas=None, train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """

    def __init__(self, out_channels, enc_bi_rnn=False, dec_bi_rnn=False, dec_drop_rnn=0.0, dec_gru=False, d_model=512, d_enc=512, d_k=64, pred_dropout=0.0, max_text_length=30, mask=True, pred_concat=True, **kwargs):
        super().__init__()
        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)
        kwargs = dict(input_size=encoder_rnn_out_size, hidden_size=encoder_rnn_out_size, num_layers=2, batch_first=True, dropout=dec_drop_rnn, bidirectional=dec_bi_rnn)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)
        self.embedding = nn.Embedding(self.num_classes, encoder_rnn_out_size, padding_idx=self.padding_idx)
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    def _2d_attention(self, decoder_input, feat, holistic_feat, valid_ratios=None):
        y = self.rnn_decoder(decoder_input)[0]
        attn_query = self.conv1x1_1(y)
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)
        attn_key = self.conv3x3_1(feat)
        attn_key = attn_key.unsqueeze(1)
        attn_weight = torch.tanh(torch.add(attn_key, attn_query))
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        attn_weight = self.conv1x1_2(attn_weight)
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1
        if valid_ratios is not None:
            for i in range(valid_ratios.size(0)):
                valid_width = torch.minimum(torch.tensor(w), torch.ceil(w * valid_ratios[i]).int())
                if valid_width < w:
                    attn_weight[i, :, :, valid_width:, :] = float('-inf')
        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        attn_feat = torch.sum(torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        if self.pred_concat:
            hf_c = holistic_feat.shape[-1]
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        if self.train_mode:
            y = self.pred_dropout(y)
        return y

    def forward_train(self, feat, out_enc, label, img_metas):
        """
        img_metas: [label, valid_ratio]
        """
        if img_metas is not None:
            assert img_metas[0].size(0) == feat.size(0)
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        lab_embedding = self.embedding(label)
        out_enc = out_enc.unsqueeze(1)
        in_dec = torch.cat((out_enc, lab_embedding), dim=1)
        out_dec = self._2d_attention(in_dec, feat, out_enc, valid_ratios=valid_ratios)
        return out_dec[:, 1:, :]

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]
        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]
        seq_len = self.max_seq_len
        bsz = feat.size(0)
        start_token = torch.full((bsz,), fill_value=self.start_idx, device=feat.device, dtype=torch.long)
        start_token = self.embedding(start_token)
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1).expand(bsz, seq_len, emb_dim)
        out_enc = out_enc.unsqueeze(1)
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding
        outputs = torch.stack(outputs, 1)
        return outputs


class SARHead(nn.Module):

    def __init__(self, in_channels, out_channels, enc_dim=512, max_text_length=30, enc_bi_rnn=False, enc_drop_rnn=0.1, enc_gru=False, dec_bi_rnn=False, dec_drop_rnn=0.0, dec_gru=False, d_k=512, pred_dropout=0.1, pred_concat=True, **kwargs):
        super(SARHead, self).__init__()
        self.encoder = SAREncoder(enc_bi_rnn=enc_bi_rnn, enc_drop_rnn=enc_drop_rnn, enc_gru=enc_gru, d_model=in_channels, d_enc=enc_dim)
        self.decoder = ParallelSARDecoder(out_channels=out_channels, enc_bi_rnn=enc_bi_rnn, dec_bi_rnn=dec_bi_rnn, dec_drop_rnn=dec_drop_rnn, dec_gru=dec_gru, d_model=in_channels, d_enc=enc_dim, d_k=d_k, pred_dropout=pred_dropout, max_text_length=max_text_length, pred_concat=pred_concat)

    def forward(self, feat, data=None):
        """
        img_metas: [label, valid_ratio]
        """
        holistic_feat = self.encoder(feat, data)
        if self.training:
            label = data[0]
            final_out = self.decoder(feat, holistic_feat, label, img_metas=data)
        else:
            final_out = self.decoder(feat, holistic_feat, label=None, img_metas=data, train_mode=False)
        return {'res': final_out}


class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, groups=None, act='relu', **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.act = act
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * 4), kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(in_channels * 4))
        self.conv3 = nn.Conv2d(in_channels=int(in_channels * 4), out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.act = None
        if act:
            self.act = Activation(act)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.act:
            x = self.act(x)
        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class ASFBlock(nn.Module):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.spatial_scale = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=1), nn.ReLU(), nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False), nn.Sigmoid())
        self.channel_scale = nn.Sequential(nn.Conv2d(in_channels=inter_channels, out_channels=out_features_num, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num
        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        return torch.cat(out_list, dim=1)


class DBFPN(nn.Module):

    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf
        self.in2_conv = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.p5_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        out4 = in4 + F.interpolate(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode='nearest')
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.interpolate(p5, scale_factor=8, mode='nearest')
        p4 = F.interpolate(p4, scale_factor=4, mode='nearest')
        p3 = F.interpolate(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])
        return fuse


class RSELayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=kernel_size, padding=int(kernel_size // 2), bias=False)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        for i in range(len(in_channels)):
            self.ins_conv.append(RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut))
            self.inp_conv.append(RSELayer(out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut))

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)
        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')
        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)
        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class IntraCLBlock(nn.Module):

    def __init__(self, in_channels=96, reduce_factor=4):
        super(IntraCLBlock, self).__init__()
        self.channels = in_channels
        self.rf = reduce_factor
        self.conv1x1_reduce_channel = nn.Conv2d(self.channels, self.channels // self.rf, kernel_size=1, stride=1, padding=0)
        self.conv1x1_return_channel = nn.Conv2d(self.channels // self.rf, self.channels, kernel_size=1, stride=1, padding=0)
        self.v_layer_7x1 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        self.v_layer_5x1 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.v_layer_3x1 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.q_layer_1x7 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        self.q_layer_1x5 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.q_layer_1x3 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.c_layer_7x7 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.c_layer_5x5 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.c_layer_3x3 = nn.Conv2d(self.channels // self.rf, self.channels // self.rf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_new = self.conv1x1_reduce_channel(x)
        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q
        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q
        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q
        x_relation = self.conv1x1_return_channel(x_3)
        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)
        return x + x_relation


class LKPAN(nn.Module):

    def __init__(self, in_channels, out_channels, mode='large', **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()
        if mode.lower() == 'lite':
            p_layer = DSConv
        elif mode.lower() == 'large':
            p_layer = nn.Conv2d
        else:
            raise ValueError("mode can only be one of ['lite', 'large'], but received {}".format(mode))
        for i in range(len(in_channels)):
            self.ins_conv.append(nn.Conv2d(in_channels=in_channels[i], out_channels=self.out_channels, kernel_size=1, bias=False))
            self.inp_conv.append(p_layer(in_channels=self.out_channels, out_channels=self.out_channels // 4, kernel_size=9, padding=4, bias=False))
            if i > 0:
                self.pan_head_conv.append(nn.Conv2d(in_channels=self.out_channels // 4, out_channels=self.out_channels // 4, kernel_size=3, padding=1, stride=2, bias=False))
            self.pan_lat_conv.append(p_layer(in_channels=self.out_channels // 4, out_channels=self.out_channels // 4, kernel_size=9, padding=4, bias=False))
        self.intracl = False
        if 'intracl' in kwargs.keys() and kwargs['intracl'] is True:
            self.intracl = kwargs['intracl']
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)
        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')
        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)
        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)
        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)
        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)
        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')
        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class LocalizationNetwork(nn.Module):

    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == 'large':
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64
        self.block_list = nn.ModuleList()
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            conv = ConvBNLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=3, act='relu')
            self.block_list.append(conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2d(1)
            else:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.append(pool)
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, F * 2)
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        self.fc2.bias.data = torch.tensor(initial_bias, dtype=torch.float32)
        nn.init.zeros_(self.fc2.weight.data)
        self.out_channels = F * 2

    def forward(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias


class GridGenerator(nn.Module):

    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-06
        self.F = num_fiducial
        self.fc = nn.Linear(in_channels, 6)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        self.fc.weight.requires_grad = False
        self.fc.bias.requires_grad = False

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)
        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).float()
        P_hat_tensor = self.build_P_hat_paddle(C, torch.tensor(P)).float()
        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)
        batch_C_prime_with_zeros = torch.cat([batch_C_prime, batch_C_ex_part_tensor], dim=1)
        batch_T = torch.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = torch.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = torch.linspace(-1.0, 1.0, int(F / 2), dtype=torch.float64)
        ctrl_pts_y_top = -1 * torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_y_bottom = torch.ones([int(F / 2)], dtype=torch.float64)
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        C = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        return C

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = (torch.arange(-I_r_width, I_r_width, 2) + 1.0) / torch.tensor(np.array([I_r_width]))
        I_r_grid_y = (torch.arange(-I_r_height, I_r_height, 2) + 1.0) / torch.tensor(np.array([I_r_height]))
        P = torch.stack(torch.meshgrid(I_r_grid_x, I_r_grid_y), dim=2)
        P = torch.permute(P, [1, 0, 2])
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_eye = torch.eye(F)
        hat_C = torch.norm(C.reshape([1, F, 2]) - C.reshape([F, 1, 2]), dim=2) + hat_eye
        hat_C = hat_C ** 2 * torch.log(hat_C)
        delta_C = torch.cat([torch.cat([torch.ones((F, 1)), C, hat_C], dim=1), torch.concat([torch.zeros((2, 3)), C.transpose(0, 1)], dim=1), torch.concat([torch.zeros((1, 3)), torch.ones((1, F))], dim=1)], axis=0)
        inv_delta_C = torch.inverse(delta_C)
        return inv_delta_C

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]
        P_tile = torch.tile(torch.unsqueeze(P, dim=1), (1, F, 1))
        C_tile = torch.unsqueeze(C, dim=0)
        P_diff = P_tile - C_tile
        rbf_norm = torch.norm(P_diff, p=2, dim=2, keepdim=False)
        rbf = torch.multiply(torch.square(rbf_norm), torch.log(rbf_norm + eps))
        P_hat = torch.cat([torch.ones((n, 1)), P, rbf], dim=1)
        return P_hat

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Module):

    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr, model_name)
        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)
        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape([-1, image.shape[2], image.shape[3], 2])
        is_fp16 = False
        if batch_P_prime.dtype != torch.float32:
            data_type = batch_P_prime.dtype
            image = image.float()
            batch_P_prime = batch_P_prime.float()
            is_fp16 = True
        batch_I_r = F.grid_sample(image, grid=batch_P_prime)
        if is_fp16:
            batch_I_r = batch_I_r.astype(data_type)
        return batch_I_r


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ASFBlock,
     lambda: ([], {'in_channels': 4, 'inter_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Act,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttentionGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_embeddings': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {})),
    (AttentionHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CTCHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClsHead,
     lambda: ([], {'in_channels': 4, 'class_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DBHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DMLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DSConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DeformableConvV2,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillationDistanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillationLossFromOutput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillationNRTRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillationSERDMLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillationVQADistanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ESEModule,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EncoderWithFC,
     lambda: ([], {'in_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EncoderWithRNN,
     lambda: ([], {'in_channels': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FCTranspose,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HG_Block,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'layer_num': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HG_Stage,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'block_num': 4, 'layer_num': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Head,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IntraCLBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {})),
    (KLCTCLogits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (KLDivLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (KnowledgeDistillationKLDivLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LearnableAffineBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LocalModule,
     lambda: ([], {'in_c': 4, 'mid_c': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LocalizationNetwork,
     lambda: ([], {'in_channels': 4, 'num_fiducial': 4, 'loc_lr': 4, 'model_name': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (LossFromOutput,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MTB,
     lambda: ([], {'cnn_num': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'dropout': 0.5, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionalEncoding_2d,
     lambda: ([], {'dropout': 0.5, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RFLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RSELayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SARHead,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SubSample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TPS,
     lambda: ([], {'in_channels': 4, 'num_fiducial': 4, 'loc_lr': 4, 'model_name': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (TransformerBlock,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

