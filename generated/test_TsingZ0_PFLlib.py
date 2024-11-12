
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


import random


import torch


import torchvision


import torchvision.transforms as transforms


import torch.utils.data as data


from scipy.io import loadmat


from torch.utils.data import DataLoader


import time


from torch.utils.data import Dataset


from torchvision.datasets import ImageFolder


from torchvision.datasets import DatasetFolder


import copy


from sklearn.preprocessing import label_binarize


from sklearn import metrics


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from collections import defaultdict


from torch.optim import Optimizer


from torch.hub import load_state_dict_from_url


from torch import nn


from torch import Tensor


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


import warnings


import logging


from typing import Tuple


import matplotlib.pyplot as plt


from math import isnan


class Gate(nn.Module):

    def __init__(self, cs) ->None:
        super().__init__()
        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)
        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm


class Ensemble(nn.Module):

    def __init__(self, model, cs, head_g, base) ->None:
        super().__init__()
        self.model = model
        self.head_g = head_g
        self.base = base
        for param in self.head_g.parameters():
            param.requires_grad = False
        for param in self.base.parameters():
            param.requires_grad = False
        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None
        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):
        rep = self.model.base(x)
        gate_in = rep
        if context != None:
            context = F.normalize(context, p=2, dim=1)
            if type(x) == type([]):
                self.context = torch.tile(context, (x[0].shape[0], 1))
            else:
                self.context = torch.tile(context, (x.shape[0], 1))
        if self.context != None:
            gate_in = rep * self.context
        if self.flag == 0:
            rep_p, rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p) + self.head_g(rep_g)
        elif self.flag == 1:
            rep_p = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p)
        else:
            rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.head_g(rep_g)
        if is_rep:
            return output, rep, self.base(x)
        else:
            return output


class ConditionalSelection(nn.Module):

    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, h_dim * 2), nn.LayerNorm([h_dim * 2]), nn.ReLU())

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]


class Generative(nn.Module):

    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) ->None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device
        self.fc1 = nn.Sequential(nn.Linear(noise_dim + num_classes, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device)
        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)
        z = self.fc1(z)
        z = self.fc(z)
        return z


class GCE(nn.Module):

    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = -torch.mean(torch.sum(softmax_loss, dim=1))
        return softmax_loss


class CoV(nn.Module):

    def __init__(self, in_dim):
        super(CoV, self).__init__()
        self.Conditional_gamma = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.LayerNorm([in_dim]))
        self.Conditional_beta = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.LayerNorm([in_dim]))
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)
        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True))
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x


class BiLSTM_TextClassification(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, embedding_dropout, lstm_dropout, attention_dropout, embedding_length, attention=False, embedding_weights=None):
        super(BiLSTM_TextClassification, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.attention_dropout = attention_dropout
        self.attention = attention
        self.embedding_length = embedding_length
        if embedding_weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_weights))
        else:
            self.word_embeddings = nn.Embedding(self.input_size, self.embedding_length)
        self.embedding_dropout_layer = nn.Dropout(p=self.embedding_dropout)
        if self.attention:
            self.attention_layer = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
            self.attention_dropout_layer = nn.Dropout(p=self.attention_dropout)
        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size, self.num_layers, dropout=lstm_dropout, bidirectional=True)
        self.lstm_dropout_layer = nn.Dropout(p=self.lstm_dropout)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def attention_forward(self, lstm_output, state, seq_lens):
        hidden = state.unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        new_hiddens = []
        for i, seq_len in enumerate(seq_lens):
            soft_attn_weights = torch.softmax(attn_weights[i][:seq_len], 0)
            new_hidden = torch.matmul(soft_attn_weights.unsqueeze(0), lstm_output[i, :seq_len, :])
            new_hiddens.append(new_hidden)
        concat_hidden = torch.cat((torch.cat(new_hiddens, 0), state), 1)
        output_hidden = self.attention_layer(concat_hidden)
        output_hidden = self.attention_dropout_layer(output_hidden)
        return output_hidden

    def forward(self, x):
        if type(x) == type([]):
            input_seq, seq_lens = x
        else:
            input_seq, seq_lens = x, [x.shape[1] for _ in range(x.shape[0])]
        batch_size = len(input_seq)
        input_seq = self.word_embeddings(input_seq)
        input_seq = self.embedding_dropout_layer(input_seq)
        h_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        c_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        input_seq = input_seq.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(input_seq, (h_0, c_0))
        output = output.permute(1, 0, 2)
        state = torch.cat([output[i, seq_len - 1, :].unsqueeze(0) for i, seq_len in enumerate(seq_lens)], dim=0)
        state = self.lstm_dropout_layer(state)
        if self.attention:
            output = self.attention_forward(output, state, seq_lens)
        else:
            output = state
        feat = self.fc1(output)
        logits = self.fc(feat)
        return logits


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), norm_layer(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.last_channel, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class BaseHeadSplit(nn.Module):

    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


class HARCNN(nn.Module):

    def __init__(self, in_channels=9, dim_hidden=64 * 26, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size), nn.ReLU(), nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=conv_kernel_size), nn.ReLU(), nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))
        self.fc = nn.Sequential(nn.Linear(dim_hidden, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Digit5CNN(nn.Module):

    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module('bn1', nn.BatchNorm2d(64))
        self.encoder.add_module('relu1', nn.ReLU())
        self.encoder.add_module('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module('conv2', nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module('bn2', nn.BatchNorm2d(64))
        self.encoder.add_module('relu2', nn.ReLU())
        self.encoder.add_module('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module('conv3', nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module('bn3', nn.BatchNorm2d(128))
        self.encoder.add_module('relu3', nn.ReLU())
        self.linear = nn.Sequential()
        self.linear.add_module('fc1', nn.Linear(8192, 3072))
        self.linear.add_module('bn4', nn.BatchNorm1d(3072))
        self.linear.add_module('relu4', nn.ReLU())
        self.linear.add_module('dropout', nn.Dropout())
        self.linear.add_module('fc2', nn.Linear(3072, 2048))
        self.linear.add_module('bn5', nn.BatchNorm1d(2048))
        self.linear.add_module('relu5', nn.ReLU())
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out


class AmazonMLP(nn.Module):

    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(5000, 1000), nn.ReLU(), nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 100), nn.ReLU())
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


class FedAvgCNN(nn.Module):

    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)))
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class FedAvgMLP(nn.Module):

    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


batch_size = 10


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class Mclr_Logistic(nn.Module):

    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class DNN(nn.Module):

    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class CifarNet(nn.Module):

    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LeNet(nn.Module):

    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(20, 50, kernel_size=5), nn.Dropout2d(p=0.5), nn.MaxPool2d(2), nn.ReLU())
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == 'wn':
            self.fc = nn.utils.weight_norm(self.fc, name='weight')
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class LSTMNet(nn.Module):

    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


class fastText(nn.Module):

    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x
        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)
        return out


class TextCNN(nn.Module):

    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.8, padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]), nn.ReLU(), nn.MaxPool1d(max_len - kernel_size[0] + 1))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]), nn.ReLU(), nn.MaxPool1d(max_len - kernel_size[1] + 1))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]), nn.ReLU(), nn.MaxPool1d(max_len - kernel_size[2] + 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x
        embedded_sent = self.embedding(text).permute(0, 2, 1)
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)
        return out


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, groups: 'int'=1, dilation: 'int'=1) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: 'int' = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, has_bn=True) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn3 = norm_layer(planes)
        else:
            self.bn3 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: 'int' = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, has_bn=True) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        if has_bn:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if has_bn:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion)
        if has_bn:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block: 'BasicBlock', layers: 'List[int]', features: 'List[int]'=[64, 128, 256, 512], num_classes: 'int'=1000, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, has_bn=True, bn_block_num=4) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and bn_block_num > 0))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2, dilate=replace_stride_with_dilation[num - 1], has_bn=has_bn and num < bn_block_num))
        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: 'BasicBlock', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False, has_bn=True) ->List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
            else:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.Identity())
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, has_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, has_bn=has_bn))
        return layers

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: 'int', dropout: 'float'=0.1, max_len: 'int'=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: 'int', d_model: 'int', nhead: 'int', nlayers: 'int', num_classes: 'int', dropout: 'float'=0.1, max_len: 'int'=200, d_hid: 'int'=2048):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.hidden_dim = d_model
        self.fc = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self) ->None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: 'Tensor', attn_mask: 'Tensor'=None) ->Tensor:
        if type(x) == type([]):
            src, _ = x
        else:
            src = x
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            attn_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_classes]
        """
        x = self.embedding(src) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)
        x = self.encoder(x, attn_mask)
        x = x[:, 0]
        output = self.fc(x)
        return output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BaseHeadSplit,
     lambda: ([], {'base': torch.nn.ReLU(), 'head': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CifarNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {})),
    (CoV,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConditionalSelection,
     lambda: ([], {'in_dim': 4, 'h_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

