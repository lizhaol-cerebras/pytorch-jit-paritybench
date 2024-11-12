
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


from torch.utils.data import Dataset


from torchvision import transforms


import numpy as np


from torch import nn


import torch.nn as nn


import math


from torch.nn.parameter import Parameter


from torch.nn.functional import interpolate


from torch.nn import functional as F


from torch.utils import model_zoo


from torch.optim import SGD


from torch.utils.data import DataLoader


from torch.autograd import Variable


from torch.optim.lr_scheduler import StepLR


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-06):
        super(FRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.t = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        miu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(miu2 + self.eps)
        return torch.max(self.gamma * x + self.beta, self.t)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MS_Block(nn.Module):

    def __init__(self, in_channel, out_channel, pool_level, txt_length):
        super(MS_Block, self).__init__()
        self.txt_length = txt_length
        pool_kernel = 5 * pool_level, 1
        pool_stride = 5 * pool_level, 1
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = interpolate(x, size=(self.txt_length, 1))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Release_Block(nn.Module):

    def __init__(self, kernel_size, in_channel=4096, out_channel=4096):
        super(Release_Block, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.batch_norm1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size, bias=False)
        self.batch_norm_down = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        residual = self.downsample(residual)
        residual = self.batch_norm_down(residual)
        out += residual
        out = self.relu(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class WeightAttention(nn.Module):

    def __init__(self, bit, ms_num=4, use_gpu=True):
        super(WeightAttention, self).__init__()
        self.weight = torch.empty([ms_num, bit])
        nn.init.normal_(self.weight, 0.25, 1 / bit)
        None
        if use_gpu:
            self.weight = self.weight
        self.weight = torch.nn.Parameter(self.weight)

    def forward(self, *input):
        hash_list = []
        for x in input:
            hash_list.append(x.unsqueeze(1))
        out = torch.cat(hash_list, dim=1)
        out = out * self.weight
        out = torch.sum(out, dim=1)
        out = out.squeeze()
        return out


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def init_pretrained_weights(self, model_url):
        """
        Initializes model with pretrained weights.
         Layers that don't match with pretrained layers in name or size are kept unchanged.
        :param model_url: a http url or local file path.
        :return:
        """
        if model_url[:4] == 'http':
            pretrain_dict = model_zoo.load_url(model_url)
        else:
            try:
                pretrain_dict = torch.load(model_url)
            except FileExistsError:
                None
                return
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        None

    def save_entire(self, model_path):
        torch.save(self, model_path)

    def save_dict(self, model_path):
        torch.save(self.state_dict(), model_path)

    def save_state(self, model_path, epoch):
        torch.save({'state_dict': self.state_dict(), 'epoch': epoch}, model_path)

    def resume_state(self, model_path):
        model_CKPT = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(model_CKPT['state_dict'])
        return model_CKPT['epoch']

    def load_dict(self, model_path):
        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(model_dict)

    @staticmethod
    def glorot(tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GraphConvolution):
                self.glorot(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *x):
        pass


class CNNF(BasicModule):

    def __init__(self, bit, leakRelu=None, bn=False):
        super(CNNF, self).__init__()
        self.module_name = 'CNNF'
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.BatchNorm2d(64) if bn else nn.LocalResponseNorm(size=2, k=2), nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(kernel_size=(3, 3), stride=2), nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.BatchNorm2d(256) if bn else nn.LocalResponseNorm(size=2, k=2), nn.MaxPool2d(kernel_size=(3, 3), stride=2), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)))
        self.fc = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True), nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1), nn.LeakyReLU(negative_slope=leakRelu, inplace=True) if leakRelu else nn.ReLU(inplace=True))
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        out = self.fc(x)
        feature = out.squeeze()
        out = self.classifier(feature)
        if out_feature:
            return out, feature
        return out


class ResNet(BasicModule):
    """Residual network.
    
    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers, last_stride=2, fc_dims=None, dropout_p=None, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, out_feature=False):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        y = self.classifier(v)
        if out_feature:
            return y, v
        return y


class VGG(BasicModule):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicModule,
     lambda: ([], {}),
     lambda: ([], {})),
    (FRN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MS_Block,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'pool_level': 4, 'txt_length': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
]

