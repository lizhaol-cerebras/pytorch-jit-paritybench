import sys
_module = sys.modules[__name__]
del sys
dataloaders = _module
custom_transforms = _module
davis_2016 = _module
helpers = _module
layers = _module
osvos_layers = _module
mypath = _module
networks = _module
vgg_osvos = _module
train_online = _module
train_parent = _module
util = _module
path_abstract = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
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


import random


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.autograd import Variable


from torch.nn import functional as F


import math


from copy import deepcopy


import scipy.io


import torch.nn as nn


import torch.nn.modules as modules


import torch.optim as optim


from torchvision import transforms


from torch.utils.data import DataLoader


import scipy.misc as sm


from torchvision import models


class PathAbstract(object):

    @staticmethod
    def db_root_dir():
        raise NotImplementedError

    @staticmethod
    def save_root_dir():
        raise NotImplementedError

    @staticmethod
    def models_dir():
        raise NotImplementedError


class Path(PathAbstract):

    @staticmethod
    def db_root_dir():
        return '/path/to/DAVIS-2016'

    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return './models'


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


def find_conv_layers(_vgg):
    inds = []
    for i in range(len(_vgg.features)):
        if isinstance(_vgg.features[i], nn.Conv2d):
            inds.append(i)
    return inds


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def interp_surgery(lay):
    m, k, h, w = lay.weight.data.size()
    if m != k:
        None
        raise ValueError
    if h != w:
        None
        raise ValueError
    filt = upsample_filt(h)
    for i in range(m):
        lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))
    return lay.weight.data


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_osvos(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class OSVOS(nn.Module):

    def __init__(self, pretrained=1):
        super(OSVOS, self).__init__()
        lay_list = [[64, 64], ['M', 128, 128], ['M', 256, 256, 256], ['M', 512, 512, 512], ['M', 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]
        None
        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()
        for i in range(0, len(lay_list)):
            stages.append(make_layers_osvos(lay_list[i], in_channels[i]))
            if i > 0:
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn
        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        None
        self._initialize_weights(pretrained)

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.stages[0](x)
        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))
        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)
        if pretrained == 1:
            None
            vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            _vgg = VGG(make_layers(vgg_structure))
            _vgg.load_state_dict(torch.load(os.path.join(Path.models_dir(), 'vgg_pytorch.pth'), map_location=lambda storage, loc: storage))
            inds = find_conv_layers(_vgg)
            k = 0
            for i in range(len(self.stages)):
                for j in range(len(self.stages[i])):
                    if isinstance(self.stages[i][j], nn.Conv2d):
                        self.stages[i][j].weight = deepcopy(_vgg.features[inds[k]].weight)
                        self.stages[i][j].bias = deepcopy(_vgg.features[inds[k]].bias)
                        k += 1
        elif pretrained == 2:
            None
            caffe_weights = scipy.io.loadmat(os.path.join(Path.models_dir(), 'vgg_caffe.mat'))
            caffe_ind = 0
            for ind, layer in enumerate(self.stages.parameters()):
                if ind % 2 == 0:
                    c_w = torch.from_numpy(caffe_weights['weights'][0][caffe_ind].transpose())
                    assert layer.data.shape == c_w.shape
                    layer.data = c_w
                else:
                    c_b = torch.from_numpy(caffe_weights['biases'][0][caffe_ind][:, 0])
                    assert layer.data.shape == c_b.shape
                    layer.data = c_b
                    caffe_ind += 1

