
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


import copy


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.utils.data import DataLoader


from sklearn.preprocessing import label_binarize


from sklearn import metrics


import time


from collections import defaultdict


import random


import torchvision.transforms as transforms


from torch.hub import load_state_dict_from_url


from torch import nn


import math


import torchvision


from torch import Tensor


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


import warnings


import logging


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import OrderedDict


from typing import Tuple


import torchvision.models as zoomodels


from torch.autograd import Function


import types


import re


import scipy.interpolate


import torch.fft


import uuid


import functools


from torch.nn.utils import spectral_norm


from time import perf_counter


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


import inspect


from torch.fft import fftn


import scipy.signal


import scipy.optimize


import torch.nn


import matplotlib.cm


from math import isnan


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


class Feature_Transformer(nn.Module):

    def __init__(self, in_features, out_features, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


class Trainable_prototypes(nn.Module):

    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU())]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
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

    def __init__(self, args, cid, feature_dim=None):
        super().__init__()
        if feature_dim is None:
            feature_dim = args.feature_dim
        self.base = eval(args.models[cid % len(args.models)])
        head = None
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'head'):
            head = self.base.head
            self.base.head = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            self.base.fc = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'classifier'):
            head = self.base.classifier
            self.base.classifier = nn.AdaptiveAvgPool1d(feature_dim)
        else:
            raise 'The base model does not have a classification head.'
        if hasattr(args, 'heads'):
            self.head = eval(args.heads[cid % len(args.heads)])
        elif 'vit' in args.models[cid % len(args.models)]:
            self.head = nn.Sequential(nn.Linear(feature_dim, 768), nn.Tanh(), nn.Linear(768, args.num_classes))
        else:
            self.head = nn.Linear(feature_dim, args.num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


class Head(nn.Module):

    def __init__(self, num_classes=10, hidden_dims=[512]):
        super().__init__()
        hidden_dims.append(num_classes)
        layers = []
        for idx in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[idx - 1], hidden_dims[idx]))
            layers.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*layers)

    def forward(self, rep):
        out = self.fc(rep)
        return out


class CNN(nn.Module):

    def __init__(self, in_features=1, num_classes=10, height=28, num_cov=2, feature_dim=512, hidden_dims=[]):
        super().__init__()
        convs = [nn.Sequential(nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)))]
        height = int(height - 5 + 1)
        height = int((height - 2) / 2 + 1)
        i = -1
        for i in range(num_cov - 1):
            convs.append(nn.Sequential(nn.Conv2d(2 ** (i + 5), 2 ** (i + 6), kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2))))
            height = int(height - 5 + 1)
            height = int((height - 2) / 2 + 1)
        self.conv = nn.Sequential(*convs)
        hidden_dims.append(feature_dim)
        layers = [nn.Flatten()]
        for idx in range(len(hidden_dims)):
            if len(layers) == 1:
                layers.append(nn.Linear(height ** 2 * 2 ** (i + 6), hidden_dims[idx]))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Linear(hidden_dims[idx - 1], hidden_dims[idx]))
                layers.append(nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc1(out)
        out = self.fc(out)
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

    def __init__(self, feature_dim=[500]):
        super(AmazonMLP, self).__init__()
        self.in_features = 5000
        self.out_features = 100
        layers = []
        for idx in range(len(feature_dim)):
            if len(layers) == 0:
                layers.append(nn.Linear(self.in_features, feature_dim[idx]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(feature_dim[idx - 1], feature_dim[idx]))
                layers.append(nn.ReLU())
        try:
            layers.append(nn.Linear(feature_dim[idx], self.out_features))
        except UnboundLocalError:
            layers.append(nn.Linear(self.in_features, self.out_features))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(self.out_features, 2)

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
        self.fc1 = nn.Linear(num_channels * len(kernel_size), hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

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
        feat = self.fc1(final_feature_map)
        out = self.fc(feat)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


_MODELS = {'RN50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt', 'RN101': 'https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt', 'RN50x4': 'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt', 'ViT-B/32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'}


def _transform(n_px):
    return Compose([Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px), lambda image: image.convert('RGB'), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def available_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor'):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int'):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_conv_features(self, features):
        features = nn.AdaptiveAvgPool2d(7)(features)
        return self.visual.attnpool(features)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text

    def forward_features(self, features, text):
        image_features = self.encode_conv_features(features)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def build_model(state_dict: 'dict'):
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: 'list' = [len(set(k.split('.')[2] for k in state_dict if k.startswith(f'visual.layer{b}'))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith(f'transformer.resblocks')))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict)
    return model.eval()


def load(name: 'str', device: 'Union[str, torch.device]'='cuda' if torch.cuda.is_available() else 'cpu', jit=True):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models = {available_models()}')
    try:
        model = torch.jit.load(model_path, map_location=device if jit else 'cpu').eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f'File {model_path} is not a JIT archive. Loading as a state dict instead')
            jit = False
        state_dict = torch.load(model_path, map_location='cpu')
    if not jit:
        model = build_model(state_dict or model.state_dict())
        if str(device) == 'cpu':
            model.float()
        return model, _transform(model.visual.input_resolution)
    device_holder = torch.jit.trace(lambda : torch.ones([]), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes('prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, 'graph') else []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, 'graph') else []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model, _transform(model.input_resolution.item())


def tokenize(texts: 'Union[str, List[str]]', context_length: 'int'=77) ->torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [([sot_token] + _tokenizer.encode(text) + [eot_token]) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f'Input {texts[i]} is too long for context length {context_length}')
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


class ClipHead(nn.Module):

    def __init__(self, prompt, device='cpu'):
        super().__init__()
        self.clip_model = load('RN50', device=device, jit=False)[0].eval()
        self.prompt = prompt

    def calc_loss(self, features):
        dev = features['last'].get_device()
        text_input = tokenize(self.prompt)
        text_features = self.clip_model.encode_text(text_input)
        image_features = self.clip_model.encode_conv_features(features['last'])
        loss = -torch.cosine_similarity(text_features, image_features, dim=1)
        return loss.mean()


class Slice(nn.Module):

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()


def interpolation_checks(t: 'Union[float, np.ndarray]', v0: 'np.ndarray', v1: 'np.ndarray') ->Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:
    """Tests for the interpolation functions"""
    assert np.min(t) >= 0.0 and np.max(t) <= 1.0
    if not isinstance(v0, np.ndarray):
        v0 = np.array(v0)
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    assert v0.shape == v1.shape, f'Incompatible shapes! v0: {v0.shape}, v1: {v1.shape}'
    return t, v0, v1


def lerp(t: 'Union[float, np.ndarray]', v0: 'Union[float, list, tuple, np.ndarray]', v1: 'Union[float, list, tuple, np.ndarray]') ->np.ndarray:
    """
    Linear interpolation between v0 (starting) and v1 (final) vectors; for optimal results,
    use t as an np.ndarray to return all results at once via broadcasting
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    v2 = (1.0 - t) * v0 + t * v1
    return v2


def slerp(t: 'Union[float, np.ndarray]', v0: 'Union[float, list, tuple, np.ndarray]', v1: 'Union[float, list, tuple, np.ndarray]', dot_threshold: 'float'=0.9995) ->np.ndarray:
    """
    Spherical linear interpolation between v0 (starting) and v1 (final) vectors; for optimal
    results, use t as an np.ndarray to return all results at once via broadcasting.

    dot_threshold is the threshold for considering if the two vectors are collinear (not recommended to alter).

    Adapted from the Python code at: https://en.wikipedia.org/wiki/Slerp (at the time, now no longer available).
    Most likely taken from Jonathan Blow's code in C++:
            http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.sum(v0 * v1)
    if np.abs(dot) > dot_threshold:
        return lerp(t, v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


class PPLSampler(torch.nn.Module):

    def __init__(self, G, G_kwargs, epsilon, space, sampling, crop, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self, c):
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)
        if self.space == 'w':
            w0, w1 = self.G.mapping(z=torch.cat([z0, z1]), c=torch.cat([c, c])).chunk(2)
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else:
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0, wt1 = self.G.mapping(z=torch.cat([zt0, zt1]), c=torch.cat([c, c])).chunk(2)
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))
        img = self.G.synthesis(ws=torch.cat([wt0, wt1]), noise_mode='const', force_fp32=True, **self.G_kwargs)
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c * 3:c * 7, c * 2:c * 6]
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])
        img = (img + 1) * (255 / 2)
        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist


class GLU(nn.Module):

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c // 2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


class InitLayer(nn.Module):

    def __init__(self, nz, channel, sz=4):
        super().__init__()
        self.init = nn.Sequential(convTranspose2d(nz, channel * 2, sz, 1, 0, bias=False), NormLayer(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


class CCBN(nn.Module):
    """ conditional batchnorm """

    def __init__(self, output_size, input_size, which_linear, eps=1e-05, momentum=0.1):
        super().__init__()
        self.output_size, self.input_size = output_size, input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
        return out * gain + bias


_conv2d_gradfix_cache = dict()


_null_tensor = torch.empty([0])


def _tuple_of_ints(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


weight_gradients_disabled = False


def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)
    key = transpose, weight_shape, stride, padding, output_padding, dilation, groups
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else:
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [(input_shape[i + 2] - (output_shape[i + 2] - 1) * stride[i] - (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1)) for i in range(ndim)]


    class Conv2d(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            ctx.save_for_backward(input if weight.requires_grad else _null_tensor, weight if input.requires_grad else _null_tensor)
            ctx.input_shape = input.shape
            if weight_shape[2:] == stride == dilation == (1, 1) and padding == (0, 0) and torch.cuda.get_device_capability(input.device) < (8, 0):
                a = weight.reshape(groups, weight_shape[0] // groups, weight_shape[1])
                b = input.reshape(input.shape[0], groups, input.shape[1] // groups, -1)
                c = (a.transpose(1, 2) if transpose else a) @ b.permute(1, 2, 0, 3).flatten(2)
                c = c.reshape(-1, input.shape[0], *input.shape[2:]).transpose(0, 1)
                c = c if bias is None else c + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                return c.contiguous(memory_format=torch.channels_last if input.stride(1) == 1 else torch.contiguous_format)
            if transpose:
                return torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs)
            return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            input_shape = ctx.input_shape
            grad_input = None
            grad_weight = None
            grad_bias = None
            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input_shape, output_shape=grad_output.shape)
                op = _conv2d_gradfix(transpose=not transpose, weight_shape=weight_shape, output_padding=p, **common_kwargs)
                grad_input = op.apply(grad_output, weight, None)
                assert grad_input.shape == input_shape
            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])
            return grad_input, grad_weight, grad_bias


    class Conv2dGradWeight(torch.autograd.Function):

        @staticmethod
        def forward(ctx, grad_output, input):
            ctx.save_for_backward(grad_output if input.requires_grad else _null_tensor, input if grad_output.requires_grad else _null_tensor)
            ctx.grad_output_shape = grad_output.shape
            ctx.input_shape = input.shape
            if weight_shape[2:] == stride == dilation == (1, 1) and padding == (0, 0):
                a = grad_output.reshape(grad_output.shape[0], groups, grad_output.shape[1] // groups, -1).permute(1, 2, 0, 3).flatten(2)
                b = input.reshape(input.shape[0], groups, input.shape[1] // groups, -1).permute(1, 2, 0, 3).flatten(2)
                c = (b @ a.transpose(1, 2) if transpose else a @ b.transpose(1, 2)).reshape(weight_shape)
                return c.contiguous(memory_format=torch.channels_last if input.stride(1) == 1 else torch.contiguous_format)
            name = 'aten::cudnn_convolution_transpose_backward_weight' if transpose else 'aten::cudnn_convolution_backward_weight'
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32]
            return torch._C._jit_get_operation(name)(weight_shape, grad_output, input, padding, stride, dilation, groups, *flags)

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_output_shape = ctx.grad_output_shape
            input_shape = ctx.input_shape
            grad2_grad_output = None
            grad2_input = None
            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output_shape
            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input_shape, output_shape=grad_output_shape)
                op = _conv2d_gradfix(transpose=not transpose, weight_shape=weight_shape, output_padding=p, **common_kwargs)
                grad2_input = op.apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input_shape
            return grad2_grad_output, grad2_input
    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d


enabled = False


def _should_use_custom_op():
    return enabled


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=False, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class UpBlockSmallCond(nn.Module):

    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)
        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn = which_bn(2 * out_planes)
        self.act = GLU()

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c)
        x = self.act(x)
        return x


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width)
        return feat + self.weight * noise


class UpBlockBigCond(nn.Module):

    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False)
        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2 * out_planes)
        self.bn2 = which_bn(2 * out_planes)
        self.act = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)
        x = self.act(x)
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)
        x = self.act(x)
        return x


class SEBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4), conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(), conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class DownBlock(nn.Module):

    def __init__(self, in_planes, out_planes, width=1):
        super().__init__()
        self.main = nn.Sequential(conv2d(in_planes, out_planes * width, 4, 2, 1, bias=True), NormLayer(out_planes * width), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feat):
        return self.main(feat)


_bias_act_cuda_cache = dict()


_plugin = None


def _bias_act_cuda(dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Fast CUDA implementation of `bias_act()` using custom ops.
    """
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)
    key = dim, act, alpha, gain, clamp
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]


    class BiasActCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, b):
            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride(1) == 1 else torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not _null_tensor:
                y = _plugin.bias_act(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor, b if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor, y if 'y' in spec.ref else _null_tensor)
            return y

        @staticmethod
        def backward(ctx, dy):
            dy = dy.contiguous(memory_format=ctx.memory_format)
            x, b, y = ctx.saved_tensors
            dx = None
            db = None
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    dx = BiasActCudaGrad.apply(dy, x, b, y)
            if ctx.needs_input_grad[1]:
                db = dx.sum([i for i in range(dx.ndim) if i != dim])
            return dx, db


    class BiasActCudaGrad(torch.autograd.Function):

        @staticmethod
        def forward(ctx, dy, x, b, y):
            ctx.memory_format = torch.channels_last if dy.ndim > 2 and dy.stride(1) == 1 else torch.contiguous_format
            dx = _plugin.bias_act(dy, b, x, y, _null_tensor, 1, dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(dy if spec.has_2nd_grad else _null_tensor, x, b, y)
            return dx

        @staticmethod
        def backward(ctx, d_dx):
            d_dx = d_dx.contiguous(memory_format=ctx.memory_format)
            dy, x, b, y = ctx.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None
            if ctx.needs_input_grad[0]:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)
            if spec.has_2nd_grad and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                d_x = _plugin.bias_act(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx, alpha, gain, clamp)
            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])
            return d_dy, d_x, d_b, d_y
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda


def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(module_name='upfirdn2d_plugin', sources=['upfirdn2d.cpp', 'upfirdn2d.cu'], headers=['upfirdn2d.h'], source_dir=os.path.dirname(__file__), extra_cuda_cflags=['--use_fast_math'])
    return True


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    """Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)


def _get_weight_shape(w):
    with misc.suppress_tracer_warnings():
        shape = [int(sz) for sz in w.shape]
    misc.assert_shape(w, shape)
    return shape


def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.
    """
    _out_channels, _in_channels_per_group, kh, kw = _get_weight_shape(w)
    if not flip_weight and (kw > 1 or kh > 1):
        w = w.flip([2, 3])
    op = conv2d_gradfix.conv_transpose2d if transpose else conv2d_gradfix.conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


_upfirdn2d_cuda_cache = dict()


def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    key = upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]


    class Upfirdn2dCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, f):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [fw - padx0 - 1, iw * upx - ow * downx + padx0 - upx + 1, fh - pady0 - 1, ih * upy - oh * downy + pady0 - upy + 1]
            dx = None
            df = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=not flip_filter, gain=gain).apply(dy, f)
            assert not ctx.needs_input_grad[1]
            return dx, df
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    """Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


class DownBlockSGBlocks(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        conv_depthwise = Conv2dLayerDepthwise(in_channels, in_channels, kernel_size=3, activation='linear')
        conv_pointwise = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation='lrelu', down=2)
        self.main = nn.Sequential(conv_depthwise, conv_pointwise)

    def forward(self, feat):
        return self.main(feat)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DownBlockSep(nn.Module):

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(SeparableConv2d(in_planes, out_planes, 3), NormLayer(out_planes), nn.LeakyReLU(0.2, inplace=True), nn.AvgPool2d(2, 2))

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(DownBlock(in_planes, out_planes), conv2d(out_planes, out_planes, 1, 1, 0, bias=False), NormLayer(out_planes), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feat):
        return self.main(feat)


class ResidualConvUnit(nn.Module):

    def __init__(self, cin, activation, bn):
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.conv(x), x)


class FeatureFusionBlock(nn.Module):

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class CCBN1D(nn.Module):
    """ conditional batchnorm """

    def __init__(self, output_size, input_size, which_linear, eps=1e-05, momentum=0.1):
        super().__init__()
        self.output_size, self.input_size = output_size, input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1)
        bias = self.bias(y).view(y.size(0), -1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
        return out * gain + bias


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size, mode='bilinear', align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x


class SingleDisc(nn.Module):

    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False):
        super().__init__()
        nfc_midas = {(4): 512, (8): 512, (16): 256, (32): 128, (64): 64, (128): 64, (256): 32, (512): 16, (1024): 8}
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}
        if nc is not None and head is None:
            nfc[start_sz] = nc
        layers = []
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)


class SingleDiscCond(nn.Module):

    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False, c_dim=1000, cmap_dim=64, rand_embedding=False):
        super().__init__()
        self.cmap_dim = cmap_dim
        nfc_midas = {(4): 512, (8): 512, (16): 256, (32): 128, (64): 64, (128): 64, (256): 32, (512): 16, (1024): 8}
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}
        if nc is not None and head is None:
            nfc[start_sz] = nc
        layers = []
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        embed_path = 'in_embeddings/tf_efficientnet_lite0.pkl'
        with open(embed_path, 'rb') as f:
            self.embed = pickle.Unpickler(f).load()['embed']
        None
        if rand_embedding:
            self.embed.__init__(num_embeddings=self.embed.num_embeddings, embedding_dim=self.embed.embedding_dim)
            None
        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.cmap_dim, activation='lrelu')

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class MultiScaleD(nn.Module):

    def __init__(self, channels, resolutions, num_discs=4, proj_type=2, cond=0, patch=False, **kwargs):
        super().__init__()
        assert num_discs in [1, 2, 3, 4, 5]
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc
        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, rec=False):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))
        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_cutout(x, ratio=0.2):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(cutout_size[0], dtype=torch.long, device=x.device), torch.arange(cutout_size[1], dtype=torch.long, device=x.device))
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(torch.arange(x.size(0), dtype=torch.long, device=x.device), torch.arange(x.size(2), dtype=torch.long, device=x.device), torch.arange(x.size(3), dtype=torch.long, device=x.device))
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


AUGMENT_FNS = {'color': [rand_brightness, rand_saturation, rand_contrast], 'translation': [rand_translation], 'cutout': [rand_cutout]}


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


VITS_IMAGENET = ['deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224', 'deit_base_distilled_patch16_224']


VITS_INCEPTION = ['vit_base_patch16_224']


VITS = VITS_IMAGENET + VITS_INCEPTION


EFFNETS_IMAGENET = ['tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b0_ns']


EFFNETS_INCEPTION = ['tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'efficientnet_b1', 'efficientnet_b1_pruned', 'efficientnet_b2_pruned', 'efficientnet_b3_pruned']


EFFNETS = EFFNETS_IMAGENET + EFFNETS_INCEPTION


REGNETS = ['regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064']


TORCHVISION = ['vgg11_bn', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'densenet121', 'densenet169', 'densenet201', 'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'shufflenet_v2_x0_5', 'mobilenet_v2', 'wide_resnet50_2', 'mnasnet0_5', 'mnasnet1_0', 'ghostnet_100', 'cspresnet50', 'fbnetc_100', 'spnasnet_100', 'resnet50d', 'resnet26', 'resnet26d', 'seresnet50', 'resnetblur50', 'resnetrs50', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l', 'ese_vovnet19b_dw', 'ese_vovnet39b', 'res2next50', 'gernet_s', 'gernet_m', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g4', 'revnet', 'dm_nfnet_f1', 'nfnet_l0']


def _feature_splitter(model, idcs):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.features[:idcs[0]])
    pretrained.layer1 = nn.Sequential(model.features[idcs[0]:idcs[1]])
    pretrained.layer2 = nn.Sequential(model.features[idcs[1]:idcs[2]])
    pretrained.layer3 = nn.Sequential(model.features[idcs[2]:idcs[3]])
    return pretrained


def _make_cspresnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.stem, model.stages[0])
    pretrained.layer1 = model.stages[1]
    pretrained.layer2 = model.stages[2]
    pretrained.layer3 = model.stages[3]
    return pretrained


def _make_densenet(model):
    pretrained = nn.Module()
    pretrained.layer0 = model.features[:6]
    pretrained.layer1 = model.features[6:8]
    pretrained.layer1[-1][-1] = nn.Identity()
    pretrained.layer1 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer1)
    pretrained.layer2 = model.features[8:10]
    pretrained.layer2[-1][-1] = nn.Identity()
    pretrained.layer2 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer2)
    pretrained.layer3 = model.features[10:12]
    pretrained.layer3 = nn.Sequential(nn.AvgPool2d(2, 2), pretrained.layer3)
    return pretrained


def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def _make_ghostnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:3])
    pretrained.layer1 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer2 = nn.Sequential(*model.blocks[5:7])
    pretrained.layer3 = nn.Sequential(*model.blocks[7:-1])
    return pretrained


def _make_nfnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.stem, model.stages[0])
    pretrained.layer1 = model.stages[1]
    pretrained.layer2 = model.stages[2]
    pretrained.layer3 = model.stages[3]
    return pretrained


def _make_regnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.stem, model.s1)
    pretrained.layer1 = model.s2
    pretrained.layer2 = model.s3
    pretrained.layer3 = model.s4
    return pretrained


def _make_resnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
    pretrained.layer1 = model.layer2
    pretrained.layer2 = model.layer3
    pretrained.layer3 = model.layer4
    return pretrained


def _make_resnet_clip(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.conv2, model.bn2, model.relu, model.conv3, model.bn3, model.relu, model.avgpool, model.layer1)
    pretrained.layer1 = model.layer2
    pretrained.layer2 = model.layer3
    pretrained.layer3 = model.layer4
    return pretrained


def _make_shufflenet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv1, model.maxpool)
    pretrained.layer1 = model.stage2
    pretrained.layer2 = model.stage3
    pretrained.layer3 = model.stage4
    return pretrained


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = posemb[:, :self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


activations = {}


def forward_flex(self, x):
    b, c, h, w = x.shape
    pos_embed = self._resize_pos_embed(self.pos_embed, h // self.patch_size[1], w // self.patch_size[0])
    B = x.shape[0]
    if hasattr(self.patch_embed, 'backbone'):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if hasattr(self, 'dist_token') and self.dist_token is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x


def get_activation(name):

    def hook(model, input, output):
        activations[name] = output
    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == 'project':
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper


def _make_vit_b16_backbone(model, features=[96, 192, 384, 768], size=[384, 384], hooks=[2, 5, 8, 11], vit_features=768, use_readout='ignore', start_index=1):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))
    pretrained.activations = activations
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    pretrained.layer1 = nn.Sequential(readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
    pretrained.layer2 = nn.Sequential(readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
    pretrained.layer3 = nn.Sequential(readout_oper[2], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0))
    pretrained.layer4 = nn.Sequential(readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained


def _make_vit(model, name):
    if 'tiny' in name:
        features = [24, 48, 96, 192]
        hooks = [2, 5, 8, 11]
        vit_features = 192
    elif 'small' in name:
        features = [48, 96, 192, 384]
        hooks = [2, 5, 8, 11]
        vit_features = 384
    elif 'base' in name:
        features = [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 768
    elif 'large' in name:
        features = [256, 512, 1024, 1024]
        hooks = [5, 11, 17, 23]
        vit_features = 1024
    else:
        raise NotImplementedError('Invalid ViT backbone not available')
    return _make_vit_b16_backbone(model, features=features, size=[224, 224], hooks=hooks, vit_features=vit_features, start_index=2 if 'deit' in name else 1)


def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    _ = pretrained.model.forward_flex(x)
    layer_1 = pretrained.activations['1']
    layer_2 = pretrained.activations['2']
    layer_3 = pretrained.activations['3']
    layer_4 = pretrained.activations['4']
    layer_1 = pretrained.layer1[0:2](layer_1)
    layer_2 = pretrained.layer2[0:2](layer_2)
    layer_3 = pretrained.layer3[0:2](layer_3)
    layer_4 = pretrained.layer4[0:2](layer_4)
    unflatten = nn.Sequential(nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]])))
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    layer_1 = pretrained.layer1[3:len(pretrained.layer1)](layer_1)
    layer_2 = pretrained.layer2[3:len(pretrained.layer2)](layer_2)
    layer_3 = pretrained.layer3[3:len(pretrained.layer3)](layer_3)
    layer_4 = pretrained.layer4[3:len(pretrained.layer4)](layer_4)
    return layer_1, layer_2, layer_3, layer_4


def calc_dims(pretrained, is_vit=False):
    dims = []
    inp_res = 256
    tmp = torch.zeros(1, 3, inp_res, inp_res)
    if not is_vit:
        tmp = pretrained.layer0(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer1(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer2(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer3(tmp)
        dims.append(tmp.shape[1:3])
    else:
        tmp = forward_vit(pretrained, tmp)
        dims = [out.shape[1:3] for out in tmp]
    dims = np.array(dims)
    channels = dims[:, 0]
    res_mult = dims[:, 1] / inp_res
    return channels, res_mult


def _make_pretrained(backbone, verbose=False):
    assert backbone in ALL_MODELS
    if backbone == 'vgg11_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [7, 14, 21, 28]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'vgg13_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 20, 27, 34]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'vgg16_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 23, 33, 43]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'vgg19_bn':
        model = zoomodels.__dict__[backbone](True)
        idcs = [13, 26, 39, 52]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'densenet121':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)
    elif backbone == 'densenet169':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)
    elif backbone == 'densenet201':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_densenet(model)
    elif backbone == 'resnet18':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'resnet34':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'resnet50':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'resnet101':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'resnet152':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'wide_resnet50_2':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'wide_resnet101_2':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_resnet(model)
    elif backbone == 'shufflenet_v2_x0_5':
        model = zoomodels.__dict__[backbone](True)
        pretrained = _make_shufflenet(model)
    elif backbone == 'mobilenet_v2':
        model = zoomodels.__dict__[backbone](True)
        idcs = [4, 7, 14, 18]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'mnasnet0_5':
        model = zoomodels.__dict__[backbone](True)
        model.features = model.layers
        idcs = [9, 10, 12, 14]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'mnasnet1_0':
        model = zoomodels.__dict__[backbone](True)
        model.features = model.layers
        idcs = [9, 10, 12, 14]
        pretrained = _feature_splitter(model, idcs)
    elif backbone == 'ghostnet_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_ghostnet(model)
    elif backbone == 'cspresnet50':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'fbnetc_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone == 'spnasnet_100':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone == 'resnet50d':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'resnet26':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'resnet26d':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'seresnet50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'resnetblur50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'resnetrs50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'tf_mixnet_s':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone == 'tf_mixnet_m':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone == 'tf_mixnet_l':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone == 'dm_nfnet_f0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'dm_nfnet_f1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'ese_vovnet19b_dw':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'ese_vovnet39b':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'res2next50':
        model = timm.create_model(backbone, pretrained=True)
        model.relu = model.act1
        pretrained = _make_resnet(model)
    elif backbone == 'gernet_s':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'gernet_m':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'repvgg_a2':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'repvgg_b0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'repvgg_b1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'repvgg_b1g4':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_cspresnet(model)
    elif backbone == 'dm_nfnet_f1':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_nfnet(model)
    elif backbone == 'nfnet_l0':
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_nfnet(model)
    elif backbone in REGNETS:
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_regnet(model)
    elif backbone in EFFNETS:
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
    elif backbone in VITS:
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_vit(model, backbone)
    elif backbone == 'resnet50_clip':
        model = clip.load('RN50', device='cpu', jit=False)[0].visual
        pretrained = _make_resnet_clip(model)
    else:
        raise NotImplementedError('Wrong model name?')
    pretrained.CHANNELS, pretrained.RES_MULT = calc_dims(pretrained, is_vit=backbone in VITS)
    if verbose:
        None
        None
        None
        None
    return pretrained


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    out_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4
    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.CHANNELS = out_channels
    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))
    scratch.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4
    return scratch


def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], 'Invalid projection type'
    pretrained = _make_pretrained(backbone)
    im_res = 256
    pretrained.RESOLUTIONS = [im_res // 4, im_res // 8, im_res // 16, im_res // 32]
    if proj_type == 0:
        return pretrained, None
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    pretrained.CHANNELS = scratch.CHANNELS
    if proj_type == 1:
        return pretrained, scratch
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)
    pretrained.RESOLUTIONS = [(res * 2) for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS
    return pretrained, scratch


NORMALIZED_CLIP = CLIP


NORMALIZED_IMAGENET = TORCHVISION + REGNETS + EFFNETS_IMAGENET + VITS_IMAGENET


NORMALIZED_INCEPTION = EFFNETS_INCEPTION + VITS_INCEPTION


def get_backbone_normstats(backbone):
    if backbone in NORMALIZED_INCEPTION:
        return {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    elif backbone in NORMALIZED_IMAGENET:
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    elif backbone in NORMALIZED_CLIP:
        return {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    else:
        raise NotImplementedError


class F_RandomProj(nn.Module):

    def __init__(self, backbone='tf_efficientnet_lite3', im_res=256, cout=64, expand=True, proj_type=2, **kwargs):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        self.normstats = get_backbone_normstats(backbone)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone, cout=self.cout, proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        if self.backbone in VITS:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        else:
            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)
        out = {'0': out0, '1': out1, '2': out2, '3': out3}
        if self.proj_type == 0:
            return out
        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])
        out = {'0': out0_channel_mixed, '1': out1_channel_mixed, '2': out2_channel_mixed, '3': out3_channel_mixed}
        if self.proj_type == 1:
            return out
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)
        out = {'0': out0_scale_mixed, '1': out1_scale_mixed, '2': out2_scale_mixed, '3': out3_scale_mixed}
        return out


class ProjectedDiscriminator(torch.nn.Module):

    def __init__(self, backbones, diffaug=True, interp224=True, backbone_kwargs={}, **kwargs):
        super().__init__()
        self.backbones = backbones
        self.diffaug = diffaug
        self.interp224 = interp224
        feature_networks, discriminators = [], []
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(channels=feat.CHANNELS, resolutions=feat.RESOLUTIONS, **backbone_kwargs)
            feature_networks.append([bb_name, feat])
            discriminators.append([bb_name, disc])
        self.feature_networks = nn.ModuleDict(feature_networks)
        self.discriminators = nn.ModuleDict(discriminators)

    def train(self, mode=True):
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c):
        logits = []
        for bb_name, feat in self.feature_networks.items():
            x_aug = DiffAugment(x, policy='color,translation,cutout') if self.diffaug else x
            x_aug = x_aug.add(1).div(2)
            x_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(x_aug)
            if self.interp224 or bb_name in VITS:
                x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)
            features = feat(x_n)
            logits += self.discriminators[bb_name](features, c)
        return logits


class F_Identity(nn.Module):

    def forward(self, x):
        return x


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [(x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device)) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def rotate2d(theta, **kwargs):
    return matrix([torch.cos(theta), torch.sin(-theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1], **kwargs)


def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    return matrix([vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0], [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0], [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0], [0, 0, 0, 1], **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix([sx, 0, 0], [0, sy, 0], [0, 0, 1], **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


def scale3d(sx, sy, sz, **kwargs):
    return matrix([sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1], **kwargs)


def translate2d(tx, ty, **kwargs):
    return matrix([1, 0, tx], [0, 1, ty], [0, 0, 1], **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def translate3d(tx, ty, tz, **kwargs):
    return matrix([1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1], **kwargs)


wavelets = {'haar': [0.7071067811865476, 0.7071067811865476], 'db1': [0.7071067811865476, 0.7071067811865476], 'db2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'db3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'db4': [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523], 'db5': [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125], 'db6': [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017], 'db7': [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236], 'db8': [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161], 'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427], 'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728], 'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148], 'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255], 'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609]}


class DummyMapping(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z, c=None, **kwargs):
        return z.unsqueeze(1)


def UpBlockBig(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False), NoiseInjection(), NormLayer(out_planes * 2), GLU(), conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False), NoiseInjection(), NormLayer(out_planes * 2), GLU())
    return block


def UpBlockSmall(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False), NormLayer(out_planes * 2), GLU())
    return block


def normalize_second_moment(x, dim=1, eps=1e-08):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class FastganSynthesis(nn.Module):

    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        nfc_multi = {(2): 16, (4): 16, (8): 8, (16): 4, (32): 2, (64): 2, (128): 1, (256): 0.5, (512): 0.25, (1024): 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)
        UpBlock = UpBlockSmall if lite else UpBlockBig
        self.feat_8 = UpBlock(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlock(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])
        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])
        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)
        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input, c=None, **kwargs):
        input = normalize_second_moment(input[:, 0])
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        if self.img_resolution >= 64:
            feat_last = feat_64
        if self.img_resolution >= 128:
            feat_last = self.se_128(feat_8, self.feat_128(feat_last))
        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))
        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))
        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)
        return self.to_big(feat_last)


class FastganSynthesisCond(nn.Module):

    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, num_classes=1000, lite=False):
        super().__init__()
        self.z_dim = z_dim
        nfc_multi = {(2): 16, (4): 16, (8): 8, (16): 4, (32): 2, (64): 2, (128): 1, (256): 0.5, (512): 0.25, (1024): 0.125, (2048): 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)
        self.img_resolution = img_resolution
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)
        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond
        self.feat_8 = UpBlock(nfc[4], nfc[8], z_dim)
        self.feat_16 = UpBlock(nfc[8], nfc[16], z_dim)
        self.feat_32 = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64 = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)
        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])
        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)
        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])
        self.embed = nn.Embedding(num_classes, z_dim)

    def forward(self, input, c, update_emas=False):
        c = self.embed(c.argmax(1))
        input = normalize_second_moment(input[:, 0])
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4, c)
        feat_16 = self.feat_16(feat_8, c)
        feat_32 = self.feat_32(feat_16, c)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, c))
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64, c))
        if self.img_resolution >= 128:
            feat_last = feat_128
        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c))
        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c))
        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c)
        return self.to_big(feat_last)


_filtered_lrelu_cuda_cache = dict()


def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or clamp == float(clamp) and clamp >= 0
    clamp = float(clamp if clamp is not None else 'inf')
    key = up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]


    class FilteredLReluCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]
            if si is None:
                si = torch.empty([0])
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)
            write_signs = si.numel() == 0 and (x.requires_grad or b.requires_grad)
            x = x.contiguous()
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn('low-performance memory layout detected in filtered_lrelu input', RuntimeWarning)
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                    warnings.warn('filtered_lrelu called with non-default cuda stream but concurrent execution is not supported', RuntimeWarning)
                y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1
            if return_code < 0:
                warnings.warn('filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback', RuntimeWarning)
                y = x.add(b.unsqueeze(-1).unsqueeze(-1))
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs)
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter)
            ctx.save_for_backward(fu, fd, si if si.numel() else so)
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = None
            dfu = None
            assert not ctx.needs_input_grad[1]
            dfd = None
            assert not ctx.needs_input_grad[2]
            db = None
            dsi = None
            assert not ctx.needs_input_grad[4]
            dsx = None
            assert not ctx.needs_input_grad[5]
            dsy = None
            assert not ctx.needs_input_grad[6]
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [fu.shape[-1] - 1 + (fd.shape[-1] - 1) - px0, xw * up - yw * down + px0 - (up - 1), fu.shape[0] - 1 + (fd.shape[0] - 1) - py0, xh * up - yh * down + py0 - (up - 1)]
                gg = gain * up ** 2 / down ** 2
                ff = not flip_filter
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)
            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])
            return dx, dfu, dfd, db, dsi, dsx, dsy
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda


def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    """Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AddReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DummyMapping,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (F_Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Feature_Transformer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InitLayer,
     lambda: ([], {'nz': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Interpolate,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ProjectReadout,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResidualConvUnit,
     lambda: ([], {'cin': 4, 'activation': 4, 'bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Slice,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transpose,
     lambda: ([], {'dim0': 4, 'dim1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

