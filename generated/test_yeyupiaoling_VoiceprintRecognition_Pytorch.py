
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


import numpy as np


from torch import nn


from collections import defaultdict


from torch.utils.data import Sampler


from torch.utils.data import RandomSampler


import random


from torch.utils.data import Dataset


import math


import torch.nn as nn


from collections import OrderedDict


import torch.nn.functional as F


import torch.utils.checkpoint as cp


from torch.optim import *


from torch.optim.lr_scheduler import *


from sklearn.metrics.pairwise import cosine_similarity


import time


import torch.distributed as dist


from torch.utils.data import DataLoader


from torch.utils.data import BatchSampler


from torch.utils.data.distributed import DistributedSampler


class KaldiFbank(nn.Module):

    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Feature, Length]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank


class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param use_hf_model: 是否使用HF上的Wav2Vec2类似模型提取音频特征
    :type use_hf_model: bool
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='MelSpectrogram', use_hf_model=False, method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        self.use_hf_model = use_hf_model
        if self.use_hf_model:
            use_gpu = torch.cuda.is_available() and method_args.get('use_gpu', True)
            self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
            self.processor = AutoFeatureExtractor.from_pretrained(feature_method)
            self.feature_model = AutoModel.from_pretrained(feature_method)
            logger.info(f'使用模型【{feature_method}】提取特征，使用【{self.device}】设备提取')
            inputs = self.processor(np.ones(16000 * 1, dtype=np.float32), sampling_rate=16000, return_tensors='pt')
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
                self.output_channels = outputs.extract_features.shape[2]
        else:
            if feature_method == 'MelSpectrogram':
                self.feat_fun = MelSpectrogram(**method_args)
            elif feature_method == 'Spectrogram':
                self.feat_fun = Spectrogram(**method_args)
            elif feature_method == 'MFCC':
                self.feat_fun = MFCC(**method_args)
            elif feature_method == 'Fbank':
                self.feat_fun = KaldiFbank(**method_args)
            else:
                raise Exception(f'预处理方法 {self._feature_method} 不存在!')
            logger.info(f'使用【{feature_method}】提取特征')

    def forward(self, waveforms, input_lens_ratio=None):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
        if self.use_hf_model:
            if isinstance(waveforms, torch.Tensor):
                waveforms = waveforms.numpy()
            inputs = self.processor(waveforms, sampling_rate=16000, return_tensors='pt')
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
                feature = outputs.extract_features.cpu().detach()
        else:
            feature = self.feat_fun(waveforms)
            feature = feature.transpose(2, 1)
        feature = feature - feature.mean(1, keepdim=True)
        if input_lens_ratio is not None:
            input_lens = input_lens_ratio * feature.shape[1]
            mask_lens = torch.round(input_lens).long()
            mask_lens = mask_lens.unsqueeze(1)
            idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
            mask = idxs < mask_lens
            mask = mask.unsqueeze(-1)
            feature = torch.where(mask, feature, torch.zeros_like(feature))
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self.use_hf_model:
            return self.output_channels
        if self._feature_method == 'MelSpectrogram':
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 400) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))


class AAMLoss(nn.Module):

    def __init__(self, margin=0.2, scale=32, easy_margin=False, label_smoothing=0.0):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.3.
            scale (float, optional): scale factor. Defaults to 32.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AAMLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        sine = torch.sqrt(1.0 - torch.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(logits > 0, phi, logits)
        else:
            phi = torch.where(logits > self.th, phi, logits - self.mmm)
        one_hot = torch.zeros(logits.size()).type_as(logits)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * logits
        output *= self.scale
        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AMLoss(nn.Module):

    def __init__(self, margin=0.3, scale=32, label_smoothing=0.0):
        super(AMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        label_view = labels.view(-1, 1)
        delt_costh = torch.zeros(logits.size(), device=labels.device).scatter_(1, label_view, self.margin)
        costh_m = logits - delt_costh
        predictions = self.scale * costh_m
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin


class ARMLoss(nn.Module):

    def __init__(self, margin=0.3, scale=32, label_smoothing=0.0):
        super(ARMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        label_view = labels.view(-1, 1)
        delt_costh = torch.zeros(logits.size(), device=labels.device).scatter_(1, label_view, self.margin)
        costh_m = logits - delt_costh
        costh_m_s = self.scale * costh_m
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1, costh_m_s.size()[1])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin


class CELoss(nn.Module):

    def __init__(self, label_smoothing=0.0):
        super(CELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss = self.criterion(logits, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        pass


class SphereFace2(nn.Module):

    def __init__(self, margin=0.2, scale=32.0, lanbuda=0.7, t=3, margin_type='C'):
        """Implement of sphereface2 for speaker verification:
            Reference:
                [1] Exploring Binary Classification Loss for Speaker Verification
                https://ieeexplore.ieee.org/abstract/document/10094954
                [2] Sphereface2: Binary classification is all you need for deep face recognition
                https://arxiv.org/pdf/2108.01513
            Args:
                scale: norm of input feature
                margin: margin
                lanbuda: weight of positive and negative pairs
                t: parameter for adjust score distribution
                margin_type: A:cos(theta+margin) or C:cos(theta)-margin
            Recommend margin:
                training: 0.2 for C and 0.15 for A
                LMF: 0.3 for C and 0.25 for A
        """
        super(SphereFace2, self).__init__()
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: 'int'):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        if self.margin_type == 'A':
            sin = torch.sqrt(1.0 - torch.pow(logits, 2))
            cos_m_theta_p = self.scale * self.fun_g(torch.where(logits > self.th, logits * self.cos_m - sin * self.sin_m, logits - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(logits * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:
            cos_m_theta_p = self.scale * (self.fun_g(logits, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(logits, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        target_mask = torch.zeros(logits.size()).type_as(logits)
        target_mask.scatter_(1, labels.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class SubCenterLoss(nn.Module):
    """Implement of large margin arc distance with subcenter:
    Reference:Sub-center ArcFace: Boosting Face Recognition byLarge-Scale Noisy
     Web Faces.https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

     Args:
        margin (float, optional): margin factor. Defaults to 0.3.
        scale (float, optional): scale factor. Defaults to 32.0.
        easy_margin (bool, optional): easy_margin flag. Defaults to False.
        K: number of sub-centers, same classifier K.
    """

    def __init__(self, margin=0.2, scale=32, easy_margin=False, K=3, label_smoothing=0.0):
        super(SubCenterLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.K = K
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        cosine = torch.reshape(logits, (-1, logits.shape[1] // self.K, self.K))
        cosine, _ = torch.max(cosine, 2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
        one_hot = logits.new_zeros(cosine.size())
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.scale
        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class TripletAngularMarginLoss(nn.Module):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to True.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.8.
        an_value (float, optional): weight for d(a, n). Defaults to 0.4.
    """

    def __init__(self, margin=0.5, normalize_feature=True, add_absolute=True, absolute_loss_weight=1.0, ap_value=0.8, an_value=0.4, label_smoothing=0.0):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss_ce = self.criterion(logits, labels)
        if self.normalize_feature:
            features = torch.divide(features, torch.norm(features, p=2, dim=-1, keepdim=True))
        bs = features.size(0)
        dist = torch.matmul(features, features.t())
        is_pos = labels.expand(bs, bs).eq(labels.expand(bs, bs).t())
        is_neg = labels.expand(bs, bs).ne(labels.expand(bs, bs).t())
        dist_ap = dist[is_pos].view(bs, -1).min(dim=1, keepdim=True)[0]
        dist_an = dist[is_neg].view(bs, -1).max(dim=1, keepdim=True)[0]
        dist_ap = torch.squeeze(dist_ap, dim=1)
        dist_an = torch.squeeze(dist_an, dim=1)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)
        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = torch.where(absolut_loss_ap > 0, absolut_loss_ap, torch.zeros_like(absolut_loss_ap))
            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = torch.where(absolut_loss_an > 0, absolut_loss_an, torch.ones_like(absolut_loss_an))
            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()) * self.absolute_loss_weight + loss.mean()
        loss = loss + loss_ce
        return loss

    def update(self, margin=0.5):
        self.ranking_loss.margin = margin


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=0.01):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):

    def forward(self, x):
        return statistics_pooling(x)


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):

    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x, use_reentrant=False)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels, out_channels=out_channels, bn_channels=bn_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, config_str=config_str, memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False, config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):

    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * math.ceil(feat_dim / 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):

    def __init__(self, input_size, embd_dim=512, growth_rate=32, bn_size=4, init_channels=128, config_str='batchnorm-relu', memory_efficient=True):
        super(CAMPPlus, self).__init__()
        self.head = FCM(feat_dim=input_size)
        channels = self.head.out_channels
        self.embd_dim = embd_dim
        self.xvector = nn.Sequential(OrderedDict([('tdnn', TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str))]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, out_channels=growth_rate, bn_channels=bn_size * growth_rate, kernel_size=kernel_size, dilation=dilation, config_str=config_str, memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module('transit%d' % (i + 1), TransitLayer(channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2
        self.xvector.add_module('out_nonlinear', get_nonlinear(config_str, channels))
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module('dense', DenseLayer(channels * 2, embd_dim, config_str='batchnorm_'))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        return x


class BatchNorm1d(nn.Module):

    def __init__(self, input_size, eps=1e-05, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.norm(x)


def get_padding_elem(L_in: 'int', stride: 'int', kernel_size: 'int', dilation: 'int'):
    if stride > 1:
        n_steps = math.ceil((L_in - kernel_size * dilation) / stride + 1)
        padding = [kernel_size // 2, kernel_size // 2]
    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1
        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding


class Conv1d(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels, stride=1, dilation=1, padding='same', groups=1, bias=True, padding_mode='reflect'):
        """_summary_

        Args:
            in_channels (int): intput channel or input data dimensions
            out_channels (int): output channel or output data dimensions
            kernel_size (int): kernel size of 1-d convolution
            stride (int, optional): strid in 1-d convolution . Defaults to 1.
            padding (str, optional): padding value. Defaults to "same".
            dilation (int, optional): dilation in 1-d convolution. Defaults to 1.
            groups (int, optional): groups in 1-d convolution. Defaults to 1.
            bias (bool, optional): bias in 1-d convolution . Defaults to True.
            padding_mode (str, optional): padding mode. Defaults to "reflect".
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(in_channels, out_channels, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)
        elif self.padding == 'causal':
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))
        elif self.padding == 'valid':
            pass
        else:
            raise ValueError(f"Padding must be 'same', 'valid' or 'causal'. Got {self.padding}")
        wx = self.conv(x)
        return wx

    def _manage_padding(self, x, kernel_size: 'int', dilation: 'int', stride: 'int'):
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)
        return x


class TDNNBlock(nn.Module):
    """An implementation of TDNN.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=nn.ReLU, groups=1):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, groups=groups)
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        """Implementation of Res2Net Block with dilation
           The paper is refered as "Res2Net: A New Multi-scale Backbone Architecture",
           whose url is https://arxiv.org/abs/1904.01169
        Args:
            in_channels (int): input channels or input dimensions
            out_channels (int): output channels or output dimensions
            scale (int, optional): scale in res2net bolck. Defaults to 8.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
        """
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList([TDNNBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation) for i in range(scale - 1)])
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1
    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is None:
        dtype = length.dtype
    if device is None:
        device = length.device
    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class SEBlock(nn.Module):

    def __init__(self, in_channels, se_channels, out_channels):
        """Implementation of SEBlock
           The paper is refered as "Squeeze-and-Excitation Networks"
           whose url is https://arxiv.org/abs/1709.01507
        Args:
            in_channels (int): input channels or input data dimensions
            se_channels (_type_): _description_
            out_channels (int): output channels or output data dimensions
        """
        super(SEBlock, self).__init__()
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class SERes2NetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1, activation=torch.nn.ReLU, groups=1):
        """Implementation of Squeeze-Extraction Res2Blocks in ECAPA-TDNN network model
           The paper is refered "Squeeze-and-Excitation Networks"
           whose url is: https://arxiv.org/pdf/1709.01507.pdf
        Args:
            in_channels (int): input channels or input data dimensions
            out_channels (int): output channels or output data dimensions
            res2net_scale (int, optional): scale in the res2net block. Defaults to 8.
            se_channels (int, optional): embedding dimensions of res2net block. Defaults to 128.
            kernel_size (int, optional): kernel size of 1-d convolution in TDNN block. Defaults to 1.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1, dilation=1, activation=activation, groups=groups)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1, dilation=1, activation=activation, groups=groups)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """ASP
    This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std
        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)
        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats


class SelfAttentivePooling(nn.Module):
    """SAP"""

    def __init__(self, in_dim, bottleneck_dim=128):
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        return mean


class TemporalAveragePooling(nn.Module):

    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = torch.mean(x, dim=2)
        x = x.flatten(start_dim=1)
        return x


class TemporalStatisticsPooling(nn.Module):

    def __init__(self):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, dim=2)
        var = torch.var(x, dim=2)
        x = torch.cat((mean, var), dim=1)
        return x


class EcapaTdnn(torch.nn.Module):

    def __init__(self, input_size, embd_dim=192, pooling_type='ASP', activation=nn.ReLU, channels=[512, 512, 512, 512, 1536], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1], attention_channels=128, res2net_scale=8, se_channels=128, global_context=True, groups=[1, 1, 1, 1, 1]):
        """Implementation of ECAPA-TDNN backbone model network
           The paper is refered as "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
           whose url is: https://arxiv.org/abs/2005.07143
        Args:
            input_size (_type_): input fature dimension
            embd_dim (int, optional): speaker embedding size. Defaults to 192.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
            channels (list, optional): inter embedding dimension. Defaults to [512, 512, 512, 512, 1536].
            kernel_sizes (list, optional): kernel size of 1-d convolution in TDNN block . Defaults to [5, 3, 3, 3, 1].
            dilations (list, optional): dilations of 1-d convolution in TDNN block. Defaults to [1, 2, 3, 4, 1].
            attention_channels (int, optional): attention dimensions. Defaults to 128.
            res2net_scale (int, optional): scale value in res2net. Defaults to 8.
            se_channels (int, optional): dimensions of squeeze-excitation block. Defaults to 128.
            global_context (bool, optional): global context flag. Defaults to True.
        """
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.blocks.append(TDNNBlock(input_size, channels[0], kernel_sizes[0], dilations[0], activation, groups[0]))
        for i in range(1, len(channels) - 1):
            self.blocks.append(SERes2NetBlock(channels[i - 1], channels[i], res2net_scale=res2net_scale, se_channels=se_channels, kernel_size=kernel_sizes[i], dilation=dilations[i], activation=activation, groups=groups[i]))
        self.mfa = TDNNBlock(channels[-1], channels[-1], kernel_sizes[-1], dilations[-1], activation, groups=groups[-1])
        cat_channels = channels[-1]
        self.embd_dim = embd_dim
        if pooling_type == 'ASP':
            self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels=attention_channels, global_context=global_context)
            self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
            self.fc = Conv1d(in_channels=channels[-1] * 2, out_channels=self.embd_dim, kernel_size=1)
        elif pooling_type == 'SAP':
            self.asp = SelfAttentivePooling(cat_channels, 128)
            self.asp_bn = nn.BatchNorm1d(cat_channels)
            self.fc = Conv1d(in_channels=cat_channels, out_channels=self.embd_dim, kernel_size=1)
        elif pooling_type == 'TAP':
            self.asp = TemporalAveragePooling()
            self.asp_bn = nn.BatchNorm1d(cat_channels)
            self.fc = Conv1d(in_channels=cat_channels, out_channels=self.embd_dim, kernel_size=1)
        elif pooling_type == 'TSP':
            self.asp = TemporalStatisticsPooling()
            self.asp_bn = nn.BatchNorm1d(cat_channels * 2)
            self.fc = Conv1d(in_channels=cat_channels * 2, out_channels=self.embd_dim, kernel_size=1)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.asp_bn(x)
        x = x.unsqueeze(2)
        x = self.fc(x).squeeze(-1)
        return x


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.SiLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)
        return xo


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockERes2Net(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class TemporalStatsPool(nn.Module):
    """TSTP
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self):
        super(TemporalStatsPool, self).__init__()

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-08)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats


class ERes2Net(nn.Module):

    def __init__(self, input_size, block=BasicBlockERes2Net, block_fuse=BasicBlockERes2Net_diff_AFF, num_blocks=[3, 4, 6, 3], m_channels=32, mul_channel=1, expansion=2, base_width=32, scale=2, embd_dim=192, two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.expansion = expansion
        self.feat_dim = input_size
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale)
        self.layer1_downsample = nn.Conv2d(m_channels * 2 * mul_channel, m_channels * 4 * mul_channel, kernel_size=3, padding=1, stride=2, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4 * mul_channel, m_channels * 8 * mul_channel, kernel_size=3, padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8 * mul_channel, m_channels * 16 * mul_channel, kernel_size=3, padding=1, stride=2, bias=False)
        self.fuse_mode12 = AFF(channels=m_channels * 4 * mul_channel)
        self.fuse_mode123 = AFF(channels=m_channels * 8 * mul_channel)
        self.fuse_mode1234 = AFF(channels=m_channels * 16 * mul_channel)
        self.n_stats = 2
        self.pooling = TemporalStatsPool()
        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pooling(fuse_out1234)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


class BasicBlockERes2NetV2(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=26, scale=2):
        super(BasicBlockERes2NetV2, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockERes2NetV2_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=26, scale=2):
        super(BasicBlockERes2NetV2_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width, r=4))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class ERes2NetV2(nn.Module):

    def __init__(self, input_size, block=BasicBlockERes2NetV2, block_fuse=BasicBlockERes2NetV2_AFF, num_blocks=[3, 4, 6, 3], m_channels=32, expansion=2, base_width=26, scale=2, embd_dim=192, two_emb_layer=False):
        super(ERes2NetV2, self).__init__()
        self.in_planes = m_channels
        self.expansion = expansion
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale)
        self.layer3_ds = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.fuse34 = AFF(channels=m_channels * 16, r=4)
        self.n_stats = 2
        self.pooling = TemporalStatsPool()
        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        stats = self.pooling(fuse_out34)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


class SpeakerIdentification(nn.Module):

    def __init__(self, input_dim, num_speakers, classifier_type='Cosine', K=1, num_blocks=0, inter_dim=512):
        """The speaker identification model, which includes the speaker backbone network
           and the a linear transform to speaker class num in training

        Args:
            input_dim (nn.Module, class): embedding model output dim.
            num_speakers (_type_): the speaker class num in the training dataset
            classifier_type (str, optional): type of output layer to uses.
            K (int, optional): SubCenterLoss function parameter. It has to match the K of the classifier.
            num_blocks (int, optional): the linear layer transform between the embedding and the final linear layer. Defaults to 0.
            inter_dim (int, optional): the output dimension of dense layer. Defaults to 512.
        """
        super(SpeakerIdentification, self).__init__()
        self.classifier_type = classifier_type
        self.blocks = nn.ModuleList()
        for index in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, config_str='batchnorm'))
            input_dim = inter_dim
        if self.classifier_type == 'Cosine':
            self.weight = nn.Parameter(torch.FloatTensor(num_speakers * K, input_dim))
            nn.init.xavier_uniform_(self.weight)
        elif self.classifier_type == 'Linear':
            self.output = nn.Linear(input_dim, num_speakers)
        else:
            raise ValueError(f'不支持该输出层：{self.classifier_type}')

    def forward(self, features):
        x = features
        for layer in self.blocks:
            x = layer(x)
        if self.classifier_type == 'Cosine':
            logits = F.linear(F.normalize(x), F.normalize(self.weight))
        else:
            logits = self.output(x)
        return {'features': features, 'logits': logits}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):

    def __init__(self, input_size, m_channels=32, layers=[3, 4, 6, 3], base_width=32, scale=2, embd_dim=192, pooling_type='ASP'):
        super(Res2Net, self).__init__()
        self.inplanes = m_channels
        self.base_width = base_width
        self.scale = scale
        self.embd_dim = embd_dim
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=7, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottle2neck, m_channels, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, m_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, m_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, m_channels * 8, layers[3], stride=2)
        cat_channels = m_channels * 8 * Bottle2neck.expansion * (input_size // base_width)
        if pooling_type == 'ASP':
            self.pooling = AttentiveStatisticsPooling(cat_channels, attention_channels=128)
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'SAP':
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TAP':
            self.pooling = TemporalAveragePooling()
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TSP':
            self.pooling = TemporalStatisticsPooling()
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.base_width, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        None
        x = self.pooling(x)
        None
        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
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
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetSE(nn.Module):

    def __init__(self, input_size, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256], embd_dim=192, pooling_type='ASP'):
        super(ResNetSE, self).__init__()
        self.inplanes = num_filters[0]
        self.embd_dim = embd_dim
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(SEBottleneck, num_filters[0], layers[0])
        self.layer2 = self._make_layer(SEBottleneck, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBottleneck, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBottleneck, num_filters[3], layers[3], stride=(2, 2))
        cat_channels = num_filters[3] * SEBottleneck.expansion * (input_size // 8)
        if pooling_type == 'ASP':
            self.pooling = AttentiveStatisticsPooling(cat_channels, attention_channels=128)
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'SAP':
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TAP':
            self.pooling = TemporalAveragePooling()
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TSP':
            self.pooling = TemporalStatisticsPooling()
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.pooling(x)
        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x)
        return x


class TDNN(nn.Module):

    def __init__(self, input_size, channels=512, embd_dim=192, pooling_type='ASP'):
        super(TDNN, self).__init__()
        self.embd_dim = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=channels, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.td_layer2 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.td_layer3 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.td_layer4 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(channels)
        self.td_layer5 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1, stride=1)
        if pooling_type == 'ASP':
            self.pooling = AttentiveStatisticsPooling(channels, attention_channels=128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'SAP':
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TAP':
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == 'TSP':
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentiveStatisticsPooling,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BasicBlockERes2Net,
     lambda: ([], {'expansion': 4, 'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockERes2NetV2,
     lambda: ([], {'expansion': 4, 'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicResBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm1d,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CAMPPlus,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'out_channels': 4, 'kernel_size': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DenseLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FCM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEBlock,
     lambda: ([], {'in_channels': 4, 'se_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttentivePooling,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SpeakerIdentification,
     lambda: ([], {'input_dim': 4, 'num_speakers': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StatsPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TDNNBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TDNNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TemporalAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalStatisticsPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalStatsPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransitLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

