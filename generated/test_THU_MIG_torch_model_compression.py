import sys
_module = sys.modules[__name__]
del sys
classification = _module
detection = _module
segmentation = _module
model_tools_exmple = _module
prune_and_recovery = _module
prune_by_class = _module
prune_by_name = _module
acnet = _module
imagenet_train = _module
prune = _module
prune_mobilenet = _module
acnet = _module
main = _module
models = _module
densenet = _module
dla = _module
dla_simple = _module
dpn = _module
efficientnet = _module
googlenet = _module
lenet = _module
mobilenet = _module
mobilenetv2 = _module
pnasnet = _module
preact_resnet = _module
regnet = _module
resnet = _module
resnext = _module
senet = _module
shufflenet = _module
shufflenetv2 = _module
unet = _module
vgg = _module
prune = _module
prune_csgd = _module
qat = _module
utils = _module
setup = _module
torchpruner = _module
function_module = _module
graph = _module
init = _module
module_pruner_regist = _module
onnx_op_regist = _module
mask_utils = _module
model_pruner = _module
model_tools = _module
module_pruner = _module
prune_function = _module
pruners = _module
operator = _module
onnx_operator = _module
operator = _module
register = _module
torchslim = _module
modules = _module
acb_corner_rep_module = _module
acnet_rep_modules = _module
base_rep_module = _module
cnc_rep_module = _module
pruning = _module
csgd = _module
resrep = _module
quantizing = _module
qat = _module
qat_tools = _module
quantizer_test = _module
reparameter = _module
reparam = _module
slim_solver = _module

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


import torchvision


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


import torchvision.transforms as transforms


import torchvision.models as models


import torchvision.datasets as datasets


import torch.optim as optim


import torch.backends.cudnn as cudnn


import math


import torch.nn.quantized.modules.functional_modules as qf


import time


import torch.nn.init as init


import torch.onnx


import torch.onnx.symbolic_helper


import torch.onnx.utils


from collections import OrderedDict


from typing import Dict


from typing import List


from typing import Set


import copy


import collections


import torch.nn.intrinsic.qat as nniqat


from torch.utils.data import DataLoader


from collections import defaultdict


from sklearn.cluster import KMeans


import torch.nn.intrinsic as intrinsic


import torch.nn.qat as nnqat


import torch.quantization as q


from torch.onnx import OperatorExportTypes


class ShuffleBlock(nn.Module):

    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_planes = out_planes / 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SplitBlock(nn.Module):

    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):

    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):

    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2 * out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels, out_channels, level=level - 1, stride=stride)
            self.right_tree = Tree(block, out_channels, out_channels, level=level - 1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class DLA(nn.Module):

    def __init__(self, block=BasicBlock, num_classes=10):
        super(DLA, self).__init__()
        self.base = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.layer3 = Tree(block, 32, 64, level=1, stride=1)
        self.layer4 = Tree(block, 64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SimpleDLA(nn.Module):

    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        self.base = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.layer3 = Tree(block, 32, 64, level=1, stride=1)
        self.layer4 = Tree(block, 64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DPN(nn.Module):

    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SE(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * group_width))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def swish(x):
    return x * x.sigmoid()


class EfficientNet(nn.Module):

    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size', 'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(Block(in_channels, out_channels, kernel_size, stride, expansion, se_ratio=0.25, drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1), nn.BatchNorm2d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1), nn.BatchNorm2d(n3x3red), nn.ReLU(True), nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), nn.BatchNorm2d(n3x3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1), nn.BatchNorm2d(n5x5red), nn.ReLU(True), nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True), nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_planes, pool_planes, kernel_size=1), nn.BatchNorm2d(pool_planes), nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Conv2d(16 * 5 * 5, 120, 1)
        self.fc2 = nn.Conv2d(120, 84, 1)
        self.fc3 = nn.Conv2d(84, 10, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1, 1, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MobileNet(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SepConv(nn.Module):
    """Separable Convolution."""

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1 + y2)


class CellB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(CellB, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(2 * out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        b1 = F.relu(y1 + y2)
        b2 = F.relu(y3 + y4)
        y = torch.cat([b1, b2], 1)
        return F.relu(self.bn2(self.conv2(y)))


class PNASNet(nn.Module):

    def __init__(self, cell_type, num_cells, num_planes):
        super(PNASNet, self).__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes * 2)
        self.layer3 = self._make_layer(num_planes * 2, num_cells=6)
        self.layer4 = self._downsample(num_planes * 4)
        self.layer5 = self._make_layer(num_planes * 4, num_cells=6)
        self.linear = nn.Linear(num_planes * 4, 10)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out


class PreActBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        out = out * w
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RegNet(nn.Module):

    def __init__(self, cfg, num_classes=10):
        super(RegNet, self).__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.linear = nn.Linear(self.cfg['widths'][-1], num_classes)

    def _make_layer(self, idx):
        depth = self.cfg['depths'][idx]
        width = self.cfg['widths'][idx]
        stride = self.cfg['strides'][idx]
        group_width = self.cfg['group_width']
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']
        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(Block(self.in_planes, width, s, group_width, bottleneck_ratio, se_ratio))
            self.in_planes = width
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.base_channel = 16
        self.in_planes = self.base_channel
        self.conv1 = nn.Conv2d(3, self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channel)
        self.layer1 = self._make_layer(block, self.base_channel, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.base_channel * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.base_channel * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.base_channel * 8, num_blocks[3], stride=2)
        self.linear = nn.Conv2d(self.base_channel * 8 * block.expansion, self.base_channel * 8 * block.expansion, 1, 1, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.linear(out)
        return out


class ResNet2(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet2, self).__init__()
        self.base_channel = 16
        self.in_planes = self.base_channel
        self.conv1 = nn.Conv2d(3, self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channel)
        self.layer1 = self._make_layer(block, self.base_channel, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.base_channel * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.base_channel * 4, num_blocks[2], stride=2)
        self.linear = nn.Conv2d(self.base_channel * 4 * block.expansion, self.base_channel * 4 * block.expansion, 1, 1, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.linear(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ShuffleNet(nn.Module):

    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']
        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 10)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


configs = {(0.5): {'out_channels': (48, 96, 192, 1024), 'num_blocks': (3, 7, 3)}, (1): {'out_channels': (116, 232, 464, 1024), 'num_blocks': (3, 7, 3)}, (1.5): {'out_channels': (176, 352, 704, 1024), 'num_blocks': (3, 7, 3)}, (2): {'out_channels': (224, 488, 976, 2048), 'num_blocks': (3, 7, 3)}}


class ShuffleNetV2(nn.Module):

    def __init__(self, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 32, 1, 1)
        self.conv5 = nn.Conv2d(32, 10, 1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(F.avg_pool2d(out1, 2)))
        out3 = F.relu(self.conv3(nn.functional.interpolate(out2, size=(int(32), int(32)), mode='nearest')))
        out = self.conv4(out3)
        out = F.avg_pool2d(out, 32)
        out = self.conv5(out)
        return out


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class FunctionModule(nn.Module):

    def __init__(self):
        super(FunctionModule, self).__init__()


_relu_backup = F.relu


class Relu(FunctionModule):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, *args, **kwargs):
        return _relu_backup(*args, **kwargs)


_add_backup = torch.add


class Add(FunctionModule):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *args, **kwargs):
        return _add_backup(*args, **kwargs)


_cat_backup = torch.cat


class Cat(FunctionModule):

    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, *args, **kwargs):
        return _cat_backup(*args, **kwargs)


_mul_backup = torch.mul


class Mul(FunctionModule):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, *args, **kwargs):
        return _mul_backup(*args, **kwargs)


class weight_norm(nn.Module):

    def __init__(self, channels, init_value=0.1, dim=(1, 2, 3)):
        super(weight_norm, self).__init__()
        self.scale = nn.Parameter(torch.ones(channels) * init_value)
        self.dim = dim

    def forward(self, kernel):
        var = torch.norm(kernel, p=2, dim=self.dim)
        return (self.scale / var).view(-1, 1, 1, 1) * kernel


class RepModule(nn.Module):

    def __init__(self):
        super(RepModule, self).__init__()

    @staticmethod
    def deploy(name, module=None):
        module = module if module is not None and isinstance(module, RepModule) else name
        return module.convert()

    def convert(self):
        raise NotImplementedError('Lack RepModule::convert')

    def forward(self, *args):
        raise NotImplementedError('Lack RepModule::forward')


class Compactor(nn.Module):

    def __init__(self, num_features):
        super(Compactor, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, stride=1, padding=0, bias=False)
        identity_mat = np.eye(num_features, dtype=np.float32)
        self.conv.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))

    def forward(self, x):
        return self.conv(x)


class ModuleCompactor(nn.Module):

    def __init__(self, module):
        super(ModuleCompactor, self).__init__()
        self.module = module
        if isinstance(module, nn.BatchNorm2d):
            self.compactor = Compactor(self.module.num_features)
            return
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            self.compactor = Compactor(self.module.out_channels)
            return
        raise RuntimeError('Unsupport type for compactor ' + str(type(self.module)))

    def forward(self, x):
        x = self.module(x)
        x = self.compactor(x)
        return x


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


def return_arg0(x):
    return x


class CnCRep(RepModule):

    def __init__(self, conv: nn.Conv2d=None, in_channels=-1, out_channels=-1, kernel_size=-1, stride=1, padding=0, dilation=1, groups=1, bias=True, deploy=False, non_linear=None, with_bn=True):
        super(CnCRep, self).__init__()
        self.with_bn = with_bn
        if conv is not None:
            in_channels = conv.in_channels
            out_channels = conv.out_channels
            kernel_size = conv.kernel_size
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]
            stride = conv.stride
            padding = conv.padding
            if isinstance(padding, tuple):
                for i in padding[1:]:
                    assert padding[0] == i
                padding = padding[0]
            dilation = conv.dilation
            if isinstance(dilation, tuple):
                for i in dilation[1:]:
                    assert dilation[0] == i
                dilation = dilation[0]
            groups = conv.groups
            bias = conv.bias is not None
            padding_mode = conv.padding_mode
        else:
            assert in_channels > 0
            assert out_channels > 0
            assert kernel_size > 0
            bias = bias is not None and bias is not False
        if non_linear is None:
            self.non_linear = return_arg0
        else:
            self.non_linear = non_linear
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert kernel_size % 2 == 1
        self.fused_conv = None
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            use_bias_in_conv = bias and not with_bn
            if padding >= 1:
                assert padding_mode == 'zeros'
                self.pad = nn.ZeroPad2d(padding=(padding, padding, padding, padding))
            else:
                self.pad = nn.Identity()
            if self.with_bn:
                self.cnc_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
            else:
                self.cnc_origin = nn.Sequential()
                self.cnc_origin.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            self.cnc_left_up = nn.Sequential()
            self.cnc_left_up.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=int((kernel_size + 1) / 2), stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            if self.with_bn:
                self.cnc_left_up.add_module('bn', nn.BatchNorm2d(out_channels))
            self.cnc_left_down = nn.Sequential()
            self.cnc_left_down.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=int((kernel_size + 1) / 2), stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            if self.with_bn:
                self.cnc_left_down.add_module('bn', nn.BatchNorm2d(out_channels))
            self.cnc_right_up = nn.Sequential()
            self.cnc_right_up.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=int((kernel_size + 1) / 2), stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            if self.with_bn:
                self.cnc_right_up.add_module('bn', nn.BatchNorm2d(out_channels))
            self.cnc_right_down = nn.Sequential()
            self.cnc_right_down.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=int((kernel_size + 1) / 2), stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            if self.with_bn:
                self.cnc_right_down.add_module('bn', nn.BatchNorm2d(out_channels))
            self.cnc_center = nn.Sequential()
            self.cnc_center.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=use_bias_in_conv, padding_mode=padding_mode))
            if self.with_bn:
                self.cnc_center.add_module('bn', nn.BatchNorm2d(out_channels))
            self.cnc_center_crop = (kernel_size - 1) // 2 * dilation
            if not self.with_bn:
                self.reduce_init_values()

    def reduce_init_values(self):
        with torch.no_grad():
            for i in [self.cnc_origin, self.cnc_center, self.cnc_left_up, self.cnc_left_down, self.cnc_right_up, self.cnc_right_down]:
                i.conv.weight.data *= 1 / 6
                if i.conv.bias is not None:
                    i.conv.bias.data *= 1 / 6

    def forward(self, inputs):
        if self.fused_conv is not None:
            return self.non_linear(self.fused_conv(inputs))
        inputs_pad = self.pad(inputs)
        out = self.cnc_origin(inputs)
        out += self.cnc_left_up(inputs_pad[:, :, :-self.cnc_center_crop, :-self.cnc_center_crop])
        out += self.cnc_left_down(inputs_pad[:, :, self.cnc_center_crop:, :-self.cnc_center_crop])
        out += self.cnc_right_up(inputs_pad[:, :, :-self.cnc_center_crop, self.cnc_center_crop:])
        out += self.cnc_right_down(inputs_pad[:, :, self.cnc_center_crop:, self.cnc_center_crop:])
        out += self.cnc_center(inputs_pad[:, :, self.cnc_center_crop:-self.cnc_center_crop, self.cnc_center_crop:-self.cnc_center_crop])
        return self.non_linear(out)

    def get_equivalent_kernel_bias(self):
        merge_dict = {'cnc_origin': [0, 0], 'cnc_left_up': [0, 0], 'cnc_left_down': [self.kernel_size // 2, 0], 'cnc_right_up': [0, self.kernel_size // 2], 'cnc_right_down': [self.kernel_size // 2, self.kernel_size // 2], 'cnc_center': [self.kernel_size // 2, self.kernel_size // 2]}
        kernel = torch.zeros_like(self.cnc_origin.conv.weight.data)
        bias = torch.zeros(self.cnc_origin.conv.weight.data.shape[0])
        if self.with_bn:
            for k, v in merge_dict.items():
                t = self.__getattr__(k)
                d, _, k1, k2 = t.conv.weight.data.shape
                bn_mean, bn_sigma, bn_gamma, bn_beta = t.bn.running_mean, (t.bn.running_var + t.bn.eps).sqrt(), t.bn.weight, t.bn.bias
                kernel[:, :, v[0]:v[0] + k1, v[1]:v[1] + k2] += t.conv.weight.data * bn_gamma.reshape(d, 1, 1, 1) / bn_sigma.reshape(d, 1, 1, 1)
                bias += bn_beta - bn_gamma * bn_mean / bn_sigma
        else:
            for k, v in merge_dict.items():
                t = self.__getattr__(k)
                d, _, k1, k2 = t.conv.weight.data.shape
                kernel[:, :, v[0]:v[0] + k1, v[1]:v[1] + k2] += t.conv.weight.data
                if t.conv.bias is not None:
                    bias += t.conv.bias.data
        return kernel, bias

    def convert(self):
        if self.fused_conv is not None:
            return self.fused_conv
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(in_channels=self.cnc_origin.conv.in_channels, out_channels=self.cnc_origin.conv.out_channels, kernel_size=self.cnc_origin.conv.kernel_size, stride=self.cnc_origin.conv.stride, padding=self.cnc_origin.conv.padding, dilation=self.cnc_origin.conv.dilation, groups=self.cnc_origin.conv.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('cnc_origin')
        self.__delattr__('cnc_left_up')
        self.__delattr__('cnc_left_down')
        self.__delattr__('cnc_right_up')
        self.__delattr__('cnc_right_down')
        self.__delattr__('cnc_center')
        return self.fused_conv


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CellA,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellB,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Compactor,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1': 4, 'n3x3red': 4, 'n3x3': 4, 'n5x5red': 4, 'n5x5': 4, 'pool_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Relu,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (SE,
     lambda: ([], {'in_planes': 4, 'se_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SplitBlock,
     lambda: ([], {'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (weight_norm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_THU_MIG_torch_model_compression(_paritybench_base):
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

