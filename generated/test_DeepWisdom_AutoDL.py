import sys
_module = sys.modules[__name__]
del sys
AutoDL_ingestion_program = _module
algorithm = _module
data_converter = _module
data_io = _module
data_pb2 = _module
dataset = _module
dataset_utils = _module
ingestion = _module
architectures = _module
resnet = _module
model = _module
skeleton = _module
data = _module
augmentations = _module
dataloader = _module
dataset = _module
transforms = _module
nn = _module
modules = _module
hooks = _module
loss = _module
wrappers = _module
optim = _module
optimizers = _module
scheduler = _module
sgdw = _module
projects = _module
api = _module
logic = _module
others = _module
utils = _module
timer = _module
Auto_NLP = _module
second_stage_models = _module
ac = _module
ac_new = _module
get_embedding = _module
model_iter_second_stage = _module
set_up = _module
tf_model = _module
log_utils = _module
time_utils = _module
CONSTANT = _module
Auto_Tabular = _module
data_space = _module
explore = _module
feat_engine = _module
feat_gen = _module
feat_namer = _module
feat_opt = _module
model_lib = _module
cb = _module
dnn = _module
dnn_n = _module
emb_nn = _module
lgb = _module
logistic_regression = _module
meta_model = _module
my_emb_nn = _module
xgb = _module
model_space = _module
data_utils = _module
eda = _module
sample = _module
mc3 = _module
model = _module
dali_data = _module
dataloader = _module
dataset = _module
stratified_sampler = _module
transforms = _module
hooks = _module
loss = _module
profile = _module
wrappers = _module
optimizers = _module
sgdw = _module
logic = _module
data_manager = _module
data_sampler = _module
feature_config = _module
feature_utils = _module
preprocess_utils = _module
sample_config = _module
sample_utils = _module
evaluator = _module
generators = _module
data_generator = _module
ensemble_manager = _module
feature_generator = _module
model_generator = _module
cnn_models = _module
model_utils = _module
rnn_models = _module
model_manager = _module
emb_utils = _module
run_model = _module
at_speech = _module
at_speech_config = _module
at_speech_cons = _module
backbones = _module
thinresnet34 = _module
tr34_bb = _module
classifier = _module
sklearn_lr = _module
thinresnet34_cls = _module
data_augment = _module
examples_gen_maker = _module
feats_data_space = _module
feats_engine = _module
raw_data_space = _module
policy_space = _module
decision_making = _module
ensemble_learning = _module
meta_learning = _module
model_executor = _module
at_toolkit = _module
at_cons = _module
at_evalator = _module
at_sampler = _module
at_tfds_convertor = _module
at_utils = _module
interface = _module
adl_classifier = _module
adl_feats_maker = _module
adl_metadata = _module
adl_tfds_convertor = _module
model_nlp = _module
nlp_autodl_config = _module
nlp_dataset_convertor = _module
AutoDL_scoring_program = _module
libscores = _module
score = _module
download_public_datasets = _module
run_local_test = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import logging


from collections import OrderedDict


import copy


import torch


import torchvision.models as models


from torch.utils import model_zoo


from torchvision.models.resnet import model_urls


import torch.nn as nn


import random


import tensorflow as tf


import torchvision as tv


import numpy as np


from torch.utils.data import Dataset


from torch import nn


from functools import wraps


from functools import reduce


from tensorflow.keras.layers import Dense


from tensorflow.keras.layers import Input


from tensorflow.keras.layers import BatchNormalization


from tensorflow.keras.layers import Dropout


from tensorflow.keras.layers import Activation


from tensorflow.keras.layers import Add


from sklearn.metrics import roc_auc_score


from torch.utils.data import DataLoader


import torch.optim as optim


import torchvision.models.video.resnet as models


from torchvision.models.video.resnet import model_urls


from itertools import chain


import math


import time


import inspect


import types


import collections


from random import shuffle


from abc import ABC


from collections import defaultdict


from torch.utils.data import Sampler


from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import StratifiedShuffleSplit


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = inplanes * planes * 3 * 3 * 3 // (inplanes * 3 * 3 + 3 * planes)
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResLayer(nn.Module):

    def __init__(self, in_c, out_c, groups=1):
        super(ResLayer, self).__init__()
        self.act = nn.CELU(0.075, inplace=False)
        conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=groups)
        norm = nn.BatchNorm2d(num_features=out_c)
        pool = nn.MaxPool2d(2)
        self.pre_conv = nn.Sequential(OrderedDict([('conv', conv), ('pool', pool), ('norm', norm), ('act', nn.CELU(0.075, inplace=False))]))
        self.res1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)), ('act', nn.CELU(0.075, inplace=False))]))
        self.res2 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)), ('act', nn.CELU(0.075, inplace=False))]))

    def forward(self, x):
        x = self.pre_conv(x)
        out = self.res1(x)
        out = self.res2(out)
        out = out + x
        return out


class ResNet9(nn.Module):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        super(ResNet9, self).__init__()
        channels = [64, 128, 256, 512]
        group = 1
        self.in_channels = in_channels
        if in_channels == 3:
            self.stem = torch.nn.Sequential(skeleton.nn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False))
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(skeleton.nn.Normalize(0.5, 0.25, inplace=False), skeleton.nn.CopyChannels(3))
        else:
            self.stem = torch.nn.Sequential(skeleton.nn.Normalize(0.5, 0.25, inplace=False), torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(3))
        conv1 = nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        norm1 = nn.BatchNorm2d(num_features=channels[0])
        act = nn.CELU(0.075, inplace=False)
        pool = nn.MaxPool2d(2)
        self.prep = nn.Sequential(OrderedDict([('conv', conv1), ('bn', norm1), ('act', act)]))
        self.layer1 = ResLayer(channels[0], channels[1], groups=group)
        conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=group)
        norm2 = nn.BatchNorm2d(num_features=channels[2])
        self.layer2 = nn.Sequential(OrderedDict([('conv', conv2), ('pool', pool), ('bn', norm2), ('act', act)]))
        self.layer3 = ResLayer(channels[2], channels[3], groups=group)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Linear(channels[3], num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.0):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url('https://github.com/DeepWisdom/AutoDL/releases/download/opensource/r9-70e4b5c2.pth.tar', model_dir=self.model_dir)
        new_sd = copy.deepcopy(sd['state_dict'])
        for key, value in sd['state_dict'].items():
            new_sd[key[7:]] = sd['state_dict'][key]
        self.load_state_dict(new_sd, strict=False)

    def forward_origin(self, x, targets):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau
        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets
        loss = self.loss_fn(input=logits, target=targets)
        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = targets == 1
            neg = targets < 1
            npos = pos.sum(dim=0)
            nneg = neg.sum(dim=0)
            positive_ratio = torch.clamp(npos / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            negative_ratio = torch.clamp(nneg / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            normalized_loss = loss * pos / positive_ratio
            normalized_loss += loss * neg / negative_ratio
            loss = normalized_loss
        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue
            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self


class MoveToHook(nn.Module):

    @staticmethod
    def to(tensors, device, half=False):
        for t in tensors:
            if isinstance(t, (tuple, list)):
                MoveToHook
            if not isinstance(t, torch.Tensor):
                continue
            t.data = t.data
            if half:
                if t.is_floating_point():
                    t.data = t.data.half()

    @staticmethod
    def get_forward_pre_hook(device, half=False):

        def hook(module, inputs):
            _ = module
            MoveToHook
        return hook


class CrossEntropyLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon=0.1, sparse_target=True, reduction='avg'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.sparse_target = sparse_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        if self.sparse_target:
            targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        else:
            targets = target
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = -targets * log_probs
        if self.reduction == 'avg':
            loss = loss.mean(0).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryCrossEntropyLabelSmooth(torch.nn.BCEWithLogitsLoss):

    def __init__(self, num_classes, epsilon=0.1, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BinaryCrossEntropyLabelSmooth, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, input, target):
        target = (1 - self.epsilon) * target + self.epsilon
        return super(BinaryCrossEntropyLabelSmooth, self).forward(input, target)


class ToDevice(torch.nn.Module):

    def __init__(self):
        super(ToDevice, self).__init__()
        self.register_buffer('buf', torch.zeros(1, dtype=torch.float32))

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (tuple, list)):
            xs = xs[0]
        device = self.buf.device
        out = []
        for x in xs:
            if x is not None and x.device != device:
                out.append(x)
            else:
                out.append(x)
        return out[0] if len(xs) == 1 else tuple(out)


class CopyChannels(torch.nn.Module):

    def __init__(self, multiple=3, dim=1):
        super(CopyChannels, self).__init__()
        self.multiple = multiple
        self.dim = dim

    def forward(self, x):
        return torch.cat([x for _ in range(self.multiple)], dim=self.dim)


class Normalize(torch.nn.Module):

    def __init__(self, mean, std, mode, inplace=False):
        super(Normalize, self).__init__()
        if mode == 'conv3d':
            if isinstance(mean, list):
                self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32)[None, :, None, None, None])
                self.register_buffer('std', torch.tensor(std, dtype=torch.float32)[None, :, None, None, None])
            else:
                self.register_buffer('mean', torch.tensor([mean], dtype=torch.float32)[None, :, None, None, None])
                self.register_buffer('std', torch.tensor([std], dtype=torch.float32)[None, :, None, None, None])
        else:
            self.register_buffer('mean', torch.tensor([mean], dtype=torch.float32)[None, :, None, None])
            self.register_buffer('std', torch.tensor([std], dtype=torch.float32)[None, :, None, None])
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            x = x.clone()
        x.sub_(self.mean).div_(self.std)
        return x


class Reshape(torch.nn.Module):

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class SplitTime(torch.nn.Module):

    def __init__(self, times):
        super(SplitTime, self).__init__()
        self.times = times

    def forward(self, x):
        batch, channels, height, width = x.shape
        return x.view(-1, self.times, channels, height, width)


class Permute(torch.nn.Module):

    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Cutout(torch.nn.Module):

    def __init__(self, ratio=0.0):
        super(Cutout, self).__init__()
        self.ratio = ratio

    def forward(self, input):
        batch, channel, height, width = input.shape
        w = int(width * self.ratio)
        h = int(height * self.ratio)
        if self.training and w > 0 and h > 0:
            x = np.random.randint(width, size=(batch,))
            y = np.random.randint(height, size=(batch,))
            x1s = np.clip(x - w // 2, 0, width)
            x2s = np.clip(x + w // 2, 0, width)
            y1s = np.clip(y - h // 2, 0, height)
            y2s = np.clip(y + h // 2, 0, height)
            mask = torch.ones_like(input)
            for idx, (x1, x2, y1, y2) in enumerate(zip(x1s, x2s, y1s, y2s)):
                mask[idx, :, y1:y2, x1:x2] = 0.0
            input = input * mask
        return input


class Mul(torch.nn.Module):

    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


def decorator_tuple_to_args(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            args[1:] = list(args[1])
        return func(*args, **kwargs)
    return wrapper


class Concat(torch.nn.Module):

    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class MergeSum(torch.nn.Module):

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)


class MergeProd(torch.nn.Module):

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[0] * xs[1]


class Choice(torch.nn.Module):

    def __init__(self, idx=0):
        super(Choice, self).__init__()
        self.idx = idx

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[self.idx]


class Toggle(torch.nn.Module):

    def __init__(self, module):
        super(Toggle, self).__init__()
        self.module = module
        self.on = True

    def forward(self, x):
        return self.module(x) if self.on else x


class Split(torch.nn.Module):

    def __init__(self, *modules):
        super(Split, self).__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def forward(self, x):
        return tuple([m(x) for m in self._modules.values()])


class DropPath(torch.nn.Module):

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self._half = False

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            shape = list(x.shape[:1]) + [(1) for _ in x.shape[1:]]
            keep_prob = 1.0 - self.drop_prob
            mask = torch.FloatTensor(*shape).bernoulli_(keep_prob)
            if self._half:
                mask = mask.half()
            x.div_(keep_prob)
            x.mul_(mask)
        return x

    def half(self):
        self._half = True

    def float(self):
        self._half = False


class DelayedPass(torch.nn.Module):

    def __init__(self):
        super(DelayedPass, self).__init__()
        self.register_buffer('keep', None)

    def forward(self, x):
        rv = self.keep
        self.keep = x
        return rv


class Reader(torch.nn.Module):

    def __init__(self, x=None):
        super(Reader, self).__init__()
        self.x = x

    def forward(self, x):
        return self.x


class KeepByPass(torch.nn.Module):

    def __init__(self):
        super(KeepByPass, self).__init__()
        self._reader = Reader()
        self.info = {}

    @property
    def x(self):
        return self._reader.x

    def forward(self, x):
        self._reader.x = x
        return x

    def reader(self):
        return self._reader


class TabularModel(nn.Module):
    """Basic model for tabular data."""

    def __init__(self, emb_szs, n_cont, out_sz, layers, emb_drop=0.2, use_bn=True, bn_final=False):
        super(TabularModel, self).__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        ps = [0.2] * len(layers)
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [None]
        layers = []
        for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.0] + ps, actns)):
            layers += self.bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
        if bn_final:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x

    def bn_drop_lin(self, n_in: int, n_out: int, bn: bool=True, p: float=0.0, actn=None):
        """Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."""
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers


class Conv3DSimple(nn.Conv3d):

    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3, 3), stride=stride, padding=padding, bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv2Plus1D(nn.Sequential):

    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv2Plus1D, self).__init__(nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, padding, padding), bias=False), nn.BatchNorm3d(midplanes), nn.ReLU(inplace=False), nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DNoTemporal, self).__init__(in_channels=in_planes, out_channels=out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, padding, padding), bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = inplanes * planes * 3 * 3 * 3 // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm3d(planes), nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False), nn.BatchNorm3d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicStem(nn.Sequential):

    def __init__(self):
        super(BasicStem, self).__init__(nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=False))


class R2Plus1dStem(nn.Sequential):

    def __init__(self):
        super(R2Plus1dStem, self).__init__(nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False), nn.BatchNorm3d(45), nn.ReLU(inplace=False), nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=False))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, stem, num_classes=400, zero_init_residual=False):
        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ResNet(VideoResNet):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet, self).__init__(Block, num_classes=num_classes, conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3, layers=[2, 2, 2, 2], stem=BasicStem, **kwargs)
        self._class_normalize = True
        self._is_video = True
        self._half = False
        self.init_hyper_params()
        self.checkpoints = []
        self.predict_prob_list = dict()
        self.round_idx = 0
        self.single_ensemble = False
        self.use_test_time_augmentation = False
        self.update_transforms = False
        self.history_predictions = dict()
        self.g_his_eval_dict = dict()
        self.last_y_pred_round = 0
        self.ensemble_scores = dict()
        self.ensemble_predictions = dict()
        self.ensemble_test_index = 0
        if in_channels == 3:
            self.preprocess = torch.nn.Sequential(skeleton.nn.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989], mode='conv3d', inplace=False))
        elif in_channels == 1:
            self.preprocess = torch.nn.Sequential(skeleton.nn.Normalize(0.5, 0.25, mode='conv3d', inplace=False), skeleton.nn.CopyChannels(3))
        else:
            self.preprocess = torch.nn.Sequential(skeleton.nn.Normalize(0.5, 0.25, mode='conv3d', inplace=False), torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(3))
        self.last_channels = 512 * Block.expansion
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)

    def init_hyper_params(self):
        self.info = {'loop': {'epoch': 0, 'test': 0, 'best_score': 0.0}, 'condition': {'first': {'train': True, 'valid': True, 'test': True}}, 'terminate': False}
        self.hyper_params = {'optimizer': {'lr': 0.15, 'warmup_multiplier': 2.0, 'warmup_epoch': 3}, 'dataset': {'train_info_sample': 256, 'cv_valid_ratio': 0.1, 'max_valid_count': 256, 'max_size': 64, 'base': 16, 'max_times': 3, 'enough_count': {'image': 10000, 'video': 1000}, 'batch_size': 32, 'steps_per_epoch': 30, 'max_epoch': 1000, 'batch_size_test': 256}, 'checkpoints': {'keep': 30}, 'conditions': {'score_type': 'auc', 'early_epoch': 1, 'skip_valid_score_threshold': 0.9, 'test_after_at_least_seconds': 1, 'test_after_at_least_seconds_max': 90, 'test_after_at_least_seconds_step': 2, 'threshold_valid_score_diff': 0.001, 'threshold_valid_best_score': 0.997, 'decide_threshold_valid_best_score': 0.93, 'max_inner_loop_ratio': 0.1, 'min_lr': 1e-06, 'use_fast_auto_aug': True}}

    def init(self, model_dir=None, gain=1.0):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['mc3_18'], model_dir=self.model_dir)
        del sd['fc.weight']
        del sd['fc.bias']
        for m in self.layer1.modules():
            for p in m.parameters():
                p.requires_grad_(False)
        for m in self.stem.modules():
            for p in m.parameters():
                p.requires_grad_(False)
        self.load_state_dict(sd, strict=False)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)

    def init_opt(self, steps_per_epoch, batch_size, init_lr, warmup_multiplier, warm_up_epoch):
        lr_multiplier = max(0.5, batch_size / 32)
        params = [p for p in self.parameters() if p.requires_grad]
        params_fc = [p for n, p in self.named_parameters() if p.requires_grad and 'fc' == n[:2] or 'conv1d' == n[:6]]
        scheduler_lr = skeleton.optim.get_change_scale(skeleton.optim.gradual_warm_up(skeleton.optim.get_reduce_on_plateau_scheduler(init_lr * lr_multiplier / warmup_multiplier, patience=10, factor=0.5, metric_name='train_loss'), warm_up_epoch=warm_up_epoch, multiplier=warmup_multiplier), init_scale=1.0)
        self.optimizer_fc = skeleton.optim.ScheduledOptimizer(params_fc, torch.optim.SGD, steps_per_epoch=steps_per_epoch, clip_grad_max_norm=None, lr=scheduler_lr, momentum=0.9, weight_decay=0.00025, nesterov=True)
        self.optimizer = skeleton.optim.ScheduledOptimizer(params, torch.optim.SGD, steps_per_epoch=steps_per_epoch, clip_grad_max_norm=None, lr=scheduler_lr, momentum=0.9, weight_decay=0.00025, nesterov=True)

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(skeleton.nn.SplitTime(times), skeleton.nn.Permute(0, 2, 1, 3, 4))
            self.conv1d_post = torch.nn.Sequential()

    def forward_origin(self, x):
        x = self.preprocess(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        dims = len(inputs.shape)
        logits = self.forward_origin(inputs)
        logits /= tau
        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets
        loss = self.loss_fn(input=logits, target=targets)
        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = targets == 1
            neg = targets < 1
            npos = pos.sum()
            nneg = neg.sum()
            positive_ratio = max(0.1, min(0.9, npos / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, nneg / (npos + nneg)))
            normalized_loss = loss * pos / positive_ratio
            normalized_loss += loss * neg / negative_ratio
            loss = normalized_loss
        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

        def half(self):
            for module in self.modules():
                if len([c for c in module.children()]) > 0:
                    continue
                if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.half()
                else:
                    module.float()
            self._half = True
            return self


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicStem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     True),
    (BinaryCrossEntropyLabelSmooth,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2Plus1D,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'midplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3DNoTemporal,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3DSimple,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CopyChannels,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cutout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DelayedPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeepByPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mul,
     lambda: ([], {'weight': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {'mean': 4, 'std': 4, 'mode': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (R2Plus1dStem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {}),
     True),
    (Reader,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResLayer,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Split,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SplitTime,
     lambda: ([], {'times': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ToDevice,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Toggle,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DeepWisdom_AutoDL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

