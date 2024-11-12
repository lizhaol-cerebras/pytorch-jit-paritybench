
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


import torch.optim as optim


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn as nn


import math


import random


import torchvision


from torchvision.models import resnet


import matplotlib


import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


import torch.optim.lr_scheduler as lrs


from torch.utils.data.sampler import SubsetRandomSampler


from torchvision import transforms as T


class NLLMultiLabelSmooth(nn.Module):

    def __init__(self, smoothing=0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Vanilla block does not support bm, gw, and se_r options'
        super(VanillaBlock, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Basic transform does not support bm, gw, and se_r options'
        super(ResBasicBlock, self).__init__()
        self._construct(w_in, w_out, stride)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)

    def _construct(self, w_in, w_out, stride):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, kernel_size=1, bias=True), Swish(), nn.Conv2d(w_se, w_in, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        num_gs = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-05, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-05, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self._construct(in_w, out_w)

    def _construct(self, in_w, out_w):
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_w, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            self.add_module('b{}'.format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {'vanilla_block': VanillaBlock, 'res_basic_block': ResBasicBlock, 'res_bottleneck_block': ResBottleneckBlock}
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(block_type)
    return block_funs[block_type]


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {'res_stem_cifar': ResStemCifar, 'res_stem_in': ResStemIN, 'simple_stem_in': SimpleStemIN}
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(stem_type)
    return stem_funs[stem_type]


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn and False
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='Network', **kwargs):
        super(AnyNet, self).__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')
        if kwargs:
            self._construct(stem_type=kwargs['stem_type'], stem_w=kwargs['stem_w'], block_type=kwargs['block_type'], ds=kwargs['ds'], ws=kwargs['ws'], ss=kwargs['ss'], bms=kwargs['bms'], gws=kwargs['gws'], se_r=kwargs['se_r'], nc=self.num_classes)
        else:
            self._construct(stem_type='plain_block', stem_w=32, block_type='plain_block', ds=[], ws=[], ss=[], bms=[], gws=[], se_r=0.25 if True else None, nc=self.num_classes)
        self.apply(init_weights)

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        bms = bms if bms else [(1.0) for _d in ds]
        gws = gws if gws else [(1) for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.prev_w = prev_w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            self.load_state_dict(torch.load(checkpoint_path))


class EffHead(nn.Module):
    """EfficientNet head."""

    def __init__(self, w_in, w_out, nc):
        super(EffHead, self).__init__()
        self._construct(w_in, w_out, nc)

    def _construct(self, w_in, w_out, nc):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.conv_swish = Swish()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if 0.0 > 0.0:
            self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, 'dropout') else x
        x = self.fc(x)
        return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        super(MBConv, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out):
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, kernel_size=1, stride=1, padding=0, bias=False)
            self.exp_bn = nn.BatchNorm2d(w_exp, eps=1e-05, momentum=0.1)
            self.exp_swish = Swish()
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel_size=kernel, stride=stride, groups=w_exp, bias=False, padding=1 if kernel == 3 else 2)
        self.dwise_bn = nn.BatchNorm2d(w_exp, eps=1e-05, momentum=0.1)
        self.dwise_swish = Swish()
        w_se = int(w_in * se_r)
        self.se = SE(w_exp, w_se)
        self.lin_proj = nn.Conv2d(w_exp, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and 0.0 > 0.0:
                f_x = nu.drop_connect(f_x, 0.0)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out, d)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            self.add_module('b{}'.format(i + 1), MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-05, momentum=0.1)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class EfficientNet(nn.Module):
    """EfficientNet-B2 model."""

    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='Network'):
        super(EfficientNet, self).__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        STEM_W = 32
        STRIDES = [1, 2, 2, 2, 1, 2, 1]
        DEPTHS = [2, 3, 3, 4, 4, 5, 2]
        WIDTHS = [16, 24, 48, 88, 120, 208, 352]
        EXP_RATIOS = [1, 6, 6, 6, 6, 6, 6]
        KERNELS = [3, 3, 5, 3, 5, 5, 3]
        HEAD_W = 1408
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')
        self._construct(stem_w=STEM_W, ds=DEPTHS, ws=WIDTHS, exp_rs=EXP_RATIOS, se_r=0.25, ss=STRIDES, ks=KERNELS, head_w=HEAD_W, nc=1000)
        self.apply(init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), EffStage(prev_w, exp_r, kernel, stride, se_r, w, d))
            prev_w = w
        self.prev_w = prev_w
        self.head_w = head_w
        self.head = EffHead(prev_w, head_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            self.load_state_dict(torch.load(checkpoint_path))


class EvoNorm(nn.Module):

    def __init__(self, input, non_linear=True, version='S0', momentum=0.9, eps=1e-05, training=True):
        super(EvoNorm, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError('Invalid EvoNorm version')
        self.insize = input
        self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
        if self.non_linear:
            self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True).reshape(1, x.size(1), 1, 1)
                with torch.no_grad():
                    self.running_var.copy_(self.momentum * self.running_var + (1 - self.momentum) * var)
            else:
                var = self.running_var
            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [(w != wp or r != rp) for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


class RegNet(AnyNet):
    """RegNetY-1.6GF model."""

    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='RegNet'):
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')
        SE_ON = True
        DEPTH = 27
        W0 = 48
        WA = 20.71
        WM = 2.65
        GROUP_W = 24
        b_ws, num_s, _, _ = generate_regnet(w_a=WA, w_0=W0, w_m=WM, d=DEPTH)
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        gws = [GROUP_W for _ in range(num_s)]
        bms = [(1.0) for _ in range(num_s)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [(2) for _ in range(num_s)]
        se_r = 0.25 if SE_ON else None
        kwargs = {'stem_type': 'simple_stem_in', 'stem_w': 32, 'block_type': 'res_bottleneck_block', 'ss': ss, 'ds': ds, 'ws': ws, 'bms': bms, 'gws': gws, 'se_r': se_r, 'nc': self.num_classes}
        super(RegNet, self).__init__(shape, num_classes, checkpoint_dir, checkpoint_name, **kwargs)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(BasicBlock, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.norm1 = norm(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.norm2 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(Bottleneck, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d
        mid_channel = out_channel // self.expansion
        self.conv1 = conv1x1(in_channel, mid_channel)
        self.norm1 = norm(mid_channel)
        self.conv2 = conv3x3(mid_channel, mid_channel, stride)
        self.norm2 = norm(mid_channel)
        self.conv3 = conv1x1(mid_channel, out_channel)
        self.norm3 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layer_config, num_classes=2, norm='batch', zero_init_residual=False):
        super(ResNet, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d
        self.in_channel = 64
        self.conv = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = norm(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64 * block.expansion, layer_config[0], stride=1, norm=norm)
        self.layer2 = self.make_layer(block, 128 * block.expansion, layer_config[1], stride=2, norm=norm)
        self.layer3 = self.make_layer(block, 256 * block.expansion, layer_config[2], stride=2, norm=norm)
        self.layer4 = self.make_layer(block, 512 * block.expansion, layer_config[3], stride=2, norm=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.dense = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)

    def make_layer(self, block, out_channel, num_blocks, stride=1, norm=None):
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(conv1x1(self.in_channel, out_channel, stride), norm(out_channel))
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, norm))
        self.in_channel = out_channel
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.dense(out)
        return out


class ResNet50(nn.Module):

    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='ResNet50', pretrained=False, pretrained_path=None, norm='batch', zero_init_residual=False):
        super(ResNet50, self).__init__()
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.H, self.W, self.C = shape
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, norm, zero_init_residual)
        if pretrained:
            None
            if pretrained_path is None:
                None
                model = resnet.resnet50(pretrained=True)
                if norm == 'evonorm':
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m = EvoNorm
                        if isinstance(m, nn.ReLU):
                            m = nn.Identity
                if zero_init_residual:
                    for m in model.modules():
                        if isinstance(m, resnet.Bottleneck):
                            nn.init.constant_(m.bn3.weight, 0)
            else:
                checkpoint = torch.load(pretrained_path)
                model.load_state_dict(checkpoint)
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_features = 512 * Bottleneck.expansion
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(self.num_features, num_classes))

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + '.pt')
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x):
        out = x
        out = self.features(out)
        out = self.classifier(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AnyHead,
     lambda: ([], {'w_in': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'd': 4, 'block_fun': torch.nn.ReLU, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicTransform,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleneckTransform,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EffHead,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLLMultiLabelSmooth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ResBasicBlock,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBottleneckBlock,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResStemCifar,
     lambda: ([], {'w_in': 4, 'w_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResStemIN,
     lambda: ([], {'w_in': 4, 'w_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SE,
     lambda: ([], {'w_in': 4, 'w_se': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleStemIN,
     lambda: ([], {'in_w': 4, 'out_w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemIN,
     lambda: ([], {'w_in': 4, 'w_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VanillaBlock,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

