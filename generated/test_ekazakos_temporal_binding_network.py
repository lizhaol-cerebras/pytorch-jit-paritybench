
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


from torch import nn


import torch.nn.functional as F


import torch.utils.data as data


import pandas as pd


import numpy as np


from numpy.random import randint


from torch.nn.init import normal_


from torch.nn.init import constant_


from collections import OrderedDict


import math


from sklearn.metrics import confusion_matrix


import time


import torch.nn.parallel


import torch.optim


from sklearn.metrics import accuracy_score


import torch.utils.model_zoo as model_zoo


import torch.nn as nn


import collections


import tensorflow as tf


import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import MultiStepLR


from torch.nn.utils import clip_grad_norm_


import torchvision


import random


import numbers


class Context_Gating(nn.Module):

    def __init__(self, dimension, add_batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class Gated_Embedding_Unit(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class Multimodal_Gated_Unit(nn.Module):

    def __init__(self, input_dimension, output_dimension, mode=0):
        super().__init__()
        self.mode = mode
        self.fc_h1 = nn.Linear(input_dimension, output_dimension)
        self.fc_h2 = nn.Linear(input_dimension, output_dimension)
        self.fc_h3 = nn.Linear(input_dimension, output_dimension)
        if self.mode == 0:
            self.fc_z1 = nn.Linear(3 * input_dimension, output_dimension)
            self.fc_z2 = nn.Linear(3 * input_dimension, output_dimension)
            self.fc_z3 = nn.Linear(3 * input_dimension, output_dimension)
        elif self.mode == 1:
            self.fc_z1 = nn.Linear(2 * input_dimension, output_dimension)
            self.fc_z2 = nn.Linear(2 * input_dimension, output_dimension)
            self.fc_z3 = nn.Linear(2 * input_dimension, output_dimension)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.sigm1 = nn.Sigmoid()
        self.sigm2 = nn.Sigmoid()
        self.sigm3 = nn.Sigmoid()

    def forward(self, inputs):
        h1 = self.tanh1(self.fc_h1(inputs[0]))
        h2 = self.tanh2(self.fc_h2(inputs[1]))
        h3 = self.tanh3(self.fc_h3(inputs[2]))
        if self.mode == 0:
            concatenated = torch.cat(inputs, dim=1)
            z1 = self.sigm1(self.fc_z1(concatenated))
            z2 = self.sigm2(self.fc_z2(concatenated))
            z3 = self.sigm3(self.fc_z3(concatenated))
        elif self.mode == 1:
            term1 = torch.cat([inputs[0], inputs[1] + inputs[2]], dim=1)
            term2 = torch.cat([inputs[1], inputs[0] + inputs[2]], dim=1)
            term3 = torch.cat([inputs[2], inputs[0] + inputs[1]], dim=1)
            z1 = self.sigm1(self.fc_z1(term1))
            z2 = self.sigm2(self.fc_z2(term2))
            z3 = self.sigm3(self.fc_z3(term3))
        if self.mode == 0:
            h = torch.mul(z1, h1) + torch.mul(z2, h2) + torch.mul(z3, h3)
        elif self.mode == 1:
            h = torch.mul(z1, h1) + torch.mul(1 - z1, h2 + h3) + torch.mul(z2, h2) + torch.mul(1 - z2, h1 + h3) + torch.mul(z3, h3) + torch.mul(1 - z3, h1 + h2)
        return h


class Fusion_Classification_Network(nn.Module):

    def __init__(self, feature_dim, modality, midfusion, num_class, consensus_type, before_softmax, dropout, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.midfusion = midfusion
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.num_segments = num_segments
        if not self.before_softmax:
            self.softmax = nn.Softmax()
        if len(self.modality) > 1:
            if self.midfusion == 'concat':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
                self._add_classification_layer(512)
            elif self.midfusion == 'context_gating':
                self._add_audiovisual_fc_layer(len(self.modality) * feature_dim, 512)
                self.context_gating = Context_Gating(512)
                self._add_classification_layer(512)
            elif self.midfusion == 'multimodal_gating':
                self.multimodal_gated_unit = Multimodal_Gated_Unit(feature_dim, 512)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)
                self._add_classification_layer(512)
        else:
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout(p=self.dropout)
            self._add_classification_layer(feature_dim)

    def _add_classification_layer(self, input_dim):
        std = 0.001
        if isinstance(self.num_class, (list, tuple)):
            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)

    def _add_audiovisual_fc_layer(self, input_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, output_dim)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        std = 0.001
        normal_(self.fc1.weight, 0, std)
        constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        if len(self.modality) > 1:
            if self.midfusion == 'concat':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
            elif self.midfusion == 'context_gating':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
                base_out = self.context_gating(base_out)
            elif self.midfusion == 'multimodal_gating':
                base_out = self.multimodal_gated_unit(inputs)
        else:
            base_out = inputs[0]
        if self.dropout > 0:
            base_out = self.dropout_layer(base_out)
        if isinstance(self.num_class, (list, tuple)):
            base_out_verb = self.fc_verb(base_out)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)
            base_out_noun = self.fc_noun(base_out)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)
            output = output_verb.squeeze(1), output_noun.squeeze(1)
        else:
            base_out = self.fc_action(base_out)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            output = output.squeeze(1)
        return output


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [(self.input_size[1] if abs(x - self.input_size[1]) < 3 else x) for x in crop_sizes]
        crop_w = [(self.input_size[0] if abs(x - self.input_size[0]) < 3 else x) for x in crop_sizes]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4
        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))
            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class TBN(nn.Module):

    def __init__(self, num_class, num_segments, modality, base_model='resnet101', new_length=None, consensus_type='avg', before_softmax=True, dropout=0.8, crop_num=1, midfusion='concat'):
        super(TBN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.midfusion = midfusion
        if not before_softmax and consensus_type != 'avg':
            raise ValueError('Only avg consensus can be used after Softmax')
        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if m == 'RGB' or m == 'Spec' else 5
        else:
            self.new_length = new_length
        None
        self._prepare_base_model(base_model)
        self._prepare_tbn()
        is_flow = any(m == 'Flow' for m in self.modality)
        is_diff = any(m == 'RGBDiff' for m in self.modality)
        is_spec = any(m == 'Spec' for m in self.modality)
        if is_flow:
            None
            self.base_model['Flow'] = self._construct_flow_model(self.base_model['Flow'])
            None
        if is_diff:
            None
            self.base_model['RGBDiff'] = self._construct_diff_model(self.base_model['RGBDiff'])
            None
        if is_spec:
            None
            self.base_model['Spec'] = self._construct_spec_model(self.base_model['Spec'])
            None
        None
        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _remove_last_layer(self):
        for m in self.modality:
            delattr(self.base_model[m], self.base_model[m].last_layer_name)
            for tup in self.base_model[m]._op_list:
                if tup[0] == self.base_model[m].last_layer_name:
                    self.base_model[m]._op_list.remove(tup)

    def _prepare_tbn(self):
        self._remove_last_layer()
        self.fusion_classification_net = Fusion_Classification_Network(self.feature_dim, self.modality, self.midfusion, self.num_class, self.consensus_type, self.before_softmax, self.dropout, self.num_segments)

    def _prepare_base_model(self, base_model):
        if base_model == 'BNInception':
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()
            for m in self.modality:
                self.base_model[m] = getattr(tf_model_zoo, base_model)()
                self.base_model[m].last_layer_name = 'fc'
                self.input_size[m] = 224
                self.input_std[m] = [1]
                if m == 'Flow':
                    self.input_mean[m] = [128]
                elif m == 'RGBDiff':
                    self.input_mean[m] = self.input_mean[m] * (1 + self.new_length[m])
                elif m == 'RGB':
                    self.input_mean[m] = [104, 117, 128]
            self.feature_dim = 1024
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def freeze_fn(self, freeze_mode):
        if freeze_mode == 'modalities':
            for m in self.modality:
                None
                base_model = getattr(self, m.lower())
                for param in base_model.parameters():
                    param.requires_grad_(False)
        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                count = 0
                None
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            m.weight.requires_grad_(False)
                            m.bias.requires_grad_(False)
        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                None
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                None
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    def forward(self, input):
        concatenated = []
        for m in self.modality:
            if m == 'RGB':
                channel = 3
            elif m == 'Flow':
                channel = 2
            elif m == 'Spec':
                channel = 1
            sample_len = channel * self.new_length[m]
            if m == 'RGBDiff':
                sample_len = 3 * self.new_length[m]
                input[m] = self._get_diff(input[m])
            base_model = getattr(self, m.lower())
            base_out = base_model(input[m].view((-1, sample_len) + input[m].size()[-2:]))
            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)
        output = self.fusion_classification_net(concatenated)
        return output

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3
        input_view = input.view((-1, self.num_segments, self.new_length['RGBDiff'] + 1, input_c) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()
        for x in reversed(list(range(1, self.new_length['RGBDiff'] + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return new_data

    def _construct_flow_model(self, base_model):
        modules = list(self.base_model['Flow'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length['Flow'],) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(2 * self.new_length['Flow'], conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_spec_model(self, base_model):
        modules = list(self.base_model['Spec'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).contiguous()
        new_conv = nn.Conv2d(self.new_length['Spec'], conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)
        pool_layer = getattr(self.base_model['Spec'], 'global_pool')
        new_avg_pooling = nn.AvgPool2d(8, stride=pool_layer.stride, padding=pool_layer.padding)
        setattr(self.base_model['Spec'], 'global_pool', new_avg_pooling)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        modules = list(self.base_model['RGBDiff'].modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = torch.cat((params[0].detach(), params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()), 1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length['RGBDiff'],) + kernel_size[2:]
        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: (v * 256 // 224) for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self):
        augmentation = {}
        if 'RGB' in self.modality:
            augmentation['RGB'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['RGB'], [1, 0.875, 0.75, 0.66]), GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in self.modality:
            augmentation['Flow'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['Flow'], [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in self.modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['RGBDiff'], [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)])
        return augmentation


class Identity(torch.nn.Module):

    def forward(self, input):
        return input


LAYER_BUILDER_DICT = dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()
    out, op, in_vars = parse_expr(info['expr'])
    assert len(out) == 1
    assert len(in_vars) == 1
    mod, out_channel = LAYER_BUILDER_DICT[op](attr, channels, conv_bias)
    return id, out[0], mod, out_channel, in_vars[0]


class BNInception(nn.Module):

    def __init__(self, model_path='tf_model_zoo/bninception/bn_inception.yaml', num_classes=101, weight_url='https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'):
        super(BNInception, self).__init__()
        manifest = yaml.load(open(model_path))
        layers = manifest['layers']
        self._channel_dict = dict()
        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(l, 3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]], conv_bias=True)
                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
        state_dict = torch.utils.model_zoo.load_url(weight_url)
        for k, v in state_dict.items():
            state_dict[k] = torch.squeeze(v, dim=0)
        self.load_state_dict(state_dict)

    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                None
            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        None
                    raise
        return data_dict[self._op_list[-1][2]]


class InceptionV3(BNInception):

    def __init__(self, model_path='tf_model_zoo/bninception/inceptionv3.yaml', num_classes=101, weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResnetV2, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C(), nn.AvgPool2d(8, count_include_pad=False))
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {})),
    (Context_Gating,
     lambda: ([], {'dimension': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Gated_Embedding_Unit,
     lambda: ([], {'input_dimension': 4, 'output_dimension': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionResnetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {})),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {})),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {})),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {})),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
]

